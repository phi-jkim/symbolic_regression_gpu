// run_evogp.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <cctype>
#include <algorithm>
#include <limits>
#include <string>
#include <math.h>   // ensure C float forms like ::sinf, ::powf are visible too

// Bring in the evogp kernel host API (we link forward.cu and generate.cu from evogp)
#include "../evogp/src/evogp/cuda/kernel.h"

// eval_tree single-expression evaluators from eval_tree.cu (GPU)
extern "C" void eval_tree_gpu(const int* tokens,
                               const float* values,
                               const float* features,
                               int len,
                               int num_features,
                               float* out_host);

// batch evaluator (datapoint-parallel) from eval_tree.cu
extern "C" void eval_tree_gpu_batch(const int* tokens,
                                    const float* values,
                                    const float* X,
                                    int len,
                                    int num_features,
                                    int dataPoints,
                                    float* out_dev,
                                    int blocks,
                                    int threads);

// population x datapoints async evaluator
extern "C" void eval_tree_gpu_pop_dp_async(const int* tokens_all,
                                            const float* values_all,
                                            const int* offsets,
                                            const int* lengths,
                                            int popSize,
                                            const float* X,
                                            int num_features,
                                            int dataPoints,
                                            float* out_dev,
                                            int blocks_y,
                                            int threads);

// Map EVOGP Function enum (defs.h) to eval_tree operator codes
static const auto map_evogp_to_eval = [](int f) -> int {
    switch (f) {
        case 0:  return 29; // IF
        case 1:  return 1;  // ADD
        case 2:  return 2;  // SUB
        case 3:  return 3;  // MUL
        case 4:  return 4;  // DIV
        case 5:  return 9;  // LOOSE_DIV
        case 6:  return 10; // POW
        case 7:  return 11; // LOOSE_POW
        case 8:  return 12; // MAX
        case 9:  return 13; // MIN
        case 10: return 14; // LT
        case 11: return 15; // GT
        case 12: return 16; // LE
        case 13: return 17; // GE
        case 14: return 5;  // SIN
        case 15: return 6;  // COS
        case 16: return 18; // TAN
        case 17: return 19; // SINH
        case 18: return 20; // COSH
        case 19: return 21; // TANH
        case 20: return 8;  // LOG
        case 21: return 22; // LOOSE_LOG
        case 22: return 7;  // EXP
        case 23: return 23; // INV
        case 24: return 24; // LOOSE_INV
        case 25: return 25; // NEG
        case 26: return 26; // ABS
        case 27: return 27; // SQRT
        case 28: return 28; // LOOSE_SQRT
        default: return 0;  // CONST 0 if unknown
    }
};

static void gpu_check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(e));
        std::exit(1);
    }
}

// ---------- Canonical prefix helpers (avoid drift across places) ----------
// Types: 0=VAR, 1=CONST, 2=UFUNC (arity 1), 3=BFUNC (arity 2), 4=TFUNC (arity 3)

static inline bool valid_prefix_types_canon(const int16_t* tptr, int len) {
    if (len <= 0) return false;
    int need = 0;
    for (int i = 0; i < len; ++i) {
        int16_t t = (int16_t)(tptr[i] & 0x7F);
        int ar = (t == 0 || t == 1) ? 0 : (t == 2 ? 1 : (t == 3 ? 2 : (t == 4 ? 3 : -1)));
        if (ar < 0) return false;
        if (t == 0 || t == 1) {      // VAR/CONST
            need += 1;
        } else if (t == 2) {         // UFUNC
            if (need < 1) return false;
            // net 0
        } else if (t == 3) {         // BFUNC
            if (need < 2) return false;
            need -= 1;               // net -1
        } else {                     // TFUNC
            if (need < 3) return false;
            need -= 2;               // net -2
        }
    }
    return need == 1;
}

static inline int prefix_effective_len_canon(const int16_t* tptr, int len) {
    // Scan right->left and find the minimal suffix that leaves one value on stack.
    int depth = 0;
    for (int i = len - 1; i >= 0; --i) {
        int16_t t = (int16_t)(tptr[i] & 0x7F);
        if (t == 0 || t == 1) {
            depth += 1;
        } else if (t == 2) {
            if (depth < 1) return len;
            // depth unchanged
        } else if (t == 3) {
            if (depth < 2) return len;
            depth -= 1;
        } else if (t == 4) {
            if (depth < 3) return len;
            depth -= 2;
        } else {
            return len;
        }
        if (depth == 1) {
            return len - i;
        }
    }
    return len;
}

// ---------- Tiny name helpers for debug ----------

static inline const char* op_name_short(int op) {
    switch (op) {
        case 1: return "ADD"; case 2: return "SUB"; case 3: return "MUL"; case 4: return "DIV";
        case 5: return "SIN"; case 6: return "COS"; case 7: return "EXP"; case 8: return "LOG";
        case 9: return "L-DIV"; case 10: return "POW"; case 11: return "L-POW"; case 12: return "MAX"; case 13: return "MIN";
        case 14: return "LT"; case 15: return "GT"; case 16: return "LE"; case 17: return "GE";
        case 18: return "TAN"; case 19: return "SINH"; case 20: return "COSH"; case 21: return "TANH";
        case 22: return "L-LOG"; case 23: return "INV"; case 24: return "L-INV"; case 25: return "NEG"; case 26: return "ABS";
        case 27: return "SQRT"; case 28: return "L-SQRT"; case 29: return "IF"; default: return "?";
    }
}

static inline int arity_of_eval_op(int op) {
    if (op == 29) return 3;
    switch (op) {
        case 5: case 6: case 7: case 8:
        case 18: case 19: case 20: case 21:
        case 22: case 23: case 24: case 25:
        case 26: case 27: case 28:
            return 1;
        default: return 2;
    }
}

// ---------- Config for batch benchmark stabilization ----------
struct BatchBenchCfg {
    int threads = 256;
    int min_len_pad = 8;      // ensure at least this many ops via semantic-NOP padding
    bool pad_with_neg_pairs = true; // even count of NEG around the expression (NEG(NEG(x)) == x)
};

// ---------- Helpers to build concatenated tokens/values ----------
struct Workset {
    std::vector<int>   offsets;
    std::vector<int>   lengths;
    std::vector<int>   tokens;
    std::vector<float> values;
    size_t tokensN = 0;
};

static Workset build_workset_for_async(unsigned int popSize,
                                       unsigned int maxGPLen,
                                       const std::vector<float>& h_val,
                                       const std::vector<int16_t>& h_type,
                                       const std::vector<int16_t>& h_size) {
    Workset w;
    w.offsets.resize(popSize);
    w.lengths.resize(popSize);
    w.tokens.reserve(popSize * maxGPLen);
    w.values.reserve(popSize * maxGPLen);

    for (unsigned int n = 0; n < popSize; ++n) {
        const size_t base = n * maxGPLen;
        int len_all = (int)h_size[base + 0];
        int len = len_all; if (len < 0) len = 0;
        w.offsets[n] = (int)w.tokens.size();
        if (!valid_prefix_types_canon(&h_type[base], len)) {
            w.lengths[n] = 0; // invalid -> skip
            continue;
        }
        int eff_len = prefix_effective_len_canon(&h_type[base], len);
        int start = len_all - eff_len; if (start < 0) start = 0;
        w.lengths[n] = eff_len;
        for (int i = start; i < len_all; ++i) {
            int16_t t_full = h_type[base + i];
            int16_t t = t_full & 0x7F;
            if (t == 0) { w.tokens.push_back(-1); w.values.push_back(h_val[base + i]); }
            else if (t == 1) { w.tokens.push_back(0);  w.values.push_back(h_val[base + i]); }
            else {
                int f;
                if (t_full & 0x80) {
                    uint32_t bits; std::memcpy(&bits, &h_val[base + i], sizeof(uint32_t));
                    f = (int)(bits & 0xFFFFu);
                } else {
                    f = (int)h_val[base + i];
                }
                w.tokens.push_back(map_evogp_to_eval(f));
                w.values.push_back(0.0f);
            }
        }
    }
    w.tokensN = w.tokens.size();
    return w;
}

// Build a stabilized workset for batch timing (pads too-short or invalid trees)
static Workset build_workset_for_batch(unsigned int popSize,
                                       unsigned int maxGPLen,
                                       const std::vector<float>& h_val,
                                       const std::vector<int16_t>& h_type,
                                       const std::vector<int16_t>& h_size,
                                       const BatchBenchCfg& cfg) {
    Workset w;
    w.offsets.resize(popSize);
    w.lengths.resize(popSize);
    w.tokens.reserve(popSize * maxGPLen);
    w.values.reserve(popSize * maxGPLen);

    const int pad_target = std::max(0, cfg.min_len_pad);

    for (unsigned int n = 0; n < popSize; ++n) {
        const size_t base = n * maxGPLen;
        const int len_all = (int)h_size[base + 0];
        w.offsets[n] = (int)w.tokens.size();

        bool valid = (len_all > 0) && valid_prefix_types_canon(&h_type[base], len_all);
        if (!valid) {
            // Replace with default VAR0, then pad with even NEG to reach pad_target
            // Prefix: [ ... many NEG ... , VAR, 0 ]
            int cur_len = 1; // VAR
            int pad_needed = std::max(0, pad_target - cur_len);
            int neg_to_add = cfg.pad_with_neg_pairs ? ((pad_needed + 1) / 2) * 2 : pad_needed;
            for (int i = 0; i < neg_to_add; ++i) { w.tokens.push_back(25); w.values.push_back(0.0f); } // OP_NEG
            w.tokens.push_back(-1); w.values.push_back(0.0f); // VAR 0
            w.lengths[n] = neg_to_add + 1;
            continue;
        }

        int eff_len = prefix_effective_len_canon(&h_type[base], len_all);
        int start = len_all - eff_len; if (start < 0) start = 0;

        // Pad: add even number of NEG at the *front* so semantics preserved (NEG(NEG(x)) == x)
        int pad_needed = std::max(0, pad_target - eff_len);
        int neg_to_add = cfg.pad_with_neg_pairs ? ((pad_needed + 1) / 2) * 2 : pad_needed;
        for (int i = 0; i < neg_to_add; ++i) { w.tokens.push_back(25); w.values.push_back(0.0f); } // OP_NEG

        for (int i = start; i < len_all; ++i) {
            int16_t t_full = h_type[base + i];
            int16_t t = t_full & 0x7F;
            if (t == 0) { w.tokens.push_back(-1); w.values.push_back(h_val[base + i]); }
            else if (t == 1) { w.tokens.push_back(0);  w.values.push_back(h_val[base + i]); }
            else {
                int f;
                if (t_full & 0x80) {
                    uint32_t bits; std::memcpy(&bits, &h_val[base + i], sizeof(uint32_t));
                    f = (int)(bits & 0xFFFFu);
                } else {
                    f = (int)h_val[base + i];
                }
                w.tokens.push_back(map_evogp_to_eval(f));
                w.values.push_back(0.0f);
            }
        }
        w.lengths[n] = neg_to_add + eff_len;
    }
    w.tokensN = w.tokens.size();
    return w;
}

// ---------- Bench: population x datapoints async evaluator ----------
static float bench_eval_tree_popdp_async(unsigned int popSize,
                                         unsigned int dataPoints,
                                         unsigned int maxGPLen,
                                         unsigned int varLen,
                                         unsigned int et_threads,
                                         float* cur_val, int16_t* cur_type, int16_t* cur_size,
                                         float* d_Xdp) {
    // Pull current population encodings to host
    std::vector<float> h_val(popSize * maxGPLen);
    std::vector<int16_t> h_type(popSize * maxGPLen);
    std::vector<int16_t> h_size(popSize * maxGPLen);
    gpu_check(cudaMemcpy(h_val.data(),  cur_val,  sizeof(float)   * h_val.size(),  cudaMemcpyDeviceToHost),  "D2H cur_val (bench)");
    gpu_check(cudaMemcpy(h_type.data(), cur_type, sizeof(int16_t) * h_type.size(), cudaMemcpyDeviceToHost), "D2H cur_type (bench)");
    gpu_check(cudaMemcpy(h_size.data(), cur_size, sizeof(int16_t) * h_size.size(), cudaMemcpyDeviceToHost), "D2H cur_size (bench)");

    Workset ws = build_workset_for_async(popSize, maxGPLen, h_val, h_type, h_size);

    // Device buffers
    int *d_offsets_all=nullptr, *d_lengths_all=nullptr, *d_tokens_all=nullptr;
    float *d_values_all=nullptr, *d_out_all=nullptr;
    gpu_check(cudaMalloc(&d_offsets_all, sizeof(int) * ws.offsets.size()), "cudaMalloc offsets_all");
    gpu_check(cudaMalloc(&d_lengths_all, sizeof(int) * ws.lengths.size()), "cudaMalloc lengths_all");
    if (ws.tokensN > 0) {
        gpu_check(cudaMalloc(&d_tokens_all, sizeof(int) * ws.tokensN), "cudaMalloc tokens_all");
        gpu_check(cudaMalloc(&d_values_all, sizeof(float) * ws.tokensN), "cudaMalloc values_all");
    }
    gpu_check(cudaMalloc(&d_out_all, sizeof(float) * (size_t)popSize * (size_t)dataPoints), "cudaMalloc out_all");

    gpu_check(cudaMemcpy(d_offsets_all, ws.offsets.data(), sizeof(int) * ws.offsets.size(), cudaMemcpyHostToDevice), "H2D offsets_all");
    gpu_check(cudaMemcpy(d_lengths_all, ws.lengths.data(), sizeof(int) * ws.lengths.size(), cudaMemcpyHostToDevice), "H2D lengths_all");
    if (ws.tokensN > 0) {
        gpu_check(cudaMemcpy(d_tokens_all, ws.tokens.data(), sizeof(int) * ws.tokensN, cudaMemcpyHostToDevice), "H2D tokens_all");
        gpu_check(cudaMemcpy(d_values_all, ws.values.data(), sizeof(float) * ws.tokensN, cudaMemcpyHostToDevice), "H2D values_all");
    }

    // Warmup + timed
    cudaEvent_t te0, te1;
    gpu_check(cudaEventCreate(&te0), "cudaEventCreate te0");
    gpu_check(cudaEventCreate(&te1), "cudaEventCreate te1");

    int threads = (int)et_threads;
    int blocks_y = (int)((dataPoints + threads - 1) / threads);

    eval_tree_gpu_pop_dp_async(d_tokens_all, d_values_all, d_offsets_all, d_lengths_all,
                               (int)popSize, d_Xdp, (int)varLen, (int)dataPoints,
                               d_out_all, blocks_y, threads);
    gpu_check(cudaDeviceSynchronize(), "warmup async sync");

    gpu_check(cudaEventRecord(te0), "cudaEventRecord te0");
    eval_tree_gpu_pop_dp_async(d_tokens_all, d_values_all, d_offsets_all, d_lengths_all,
                               (int)popSize, d_Xdp, (int)varLen, (int)dataPoints,
                               d_out_all, blocks_y, threads);
    gpu_check(cudaEventRecord(te1), "cudaEventRecord te1");
    gpu_check(cudaEventSynchronize(te1), "cudaEventSynchronize te1");

    float ms = 0.0f; gpu_check(cudaEventElapsedTime(&ms, te0, te1), "cudaEventElapsedTime");
    cudaEventDestroy(te0); cudaEventDestroy(te1);

    cudaFree(d_offsets_all); cudaFree(d_lengths_all);
    if (d_tokens_all) cudaFree(d_tokens_all);
    if (d_values_all) cudaFree(d_values_all);
    cudaFree(d_out_all);
    return ms;
}

// ---------- Bench: batch evaluator across the entire population ----------
static float bench_eval_tree_batch(unsigned int popSize,
                                   unsigned int dataPoints,
                                   unsigned int maxGPLen,
                                   unsigned int varLen,
                                   unsigned int et_threads,
                                   float* cur_val, int16_t* cur_type, int16_t* cur_size,
                                   float* d_Xdp,
                                   const BatchBenchCfg& cfg) {
    // Pull population encodings
    std::vector<float> h_val(popSize * maxGPLen);
    std::vector<int16_t> h_type(popSize * maxGPLen);
    std::vector<int16_t> h_size(popSize * maxGPLen);
    gpu_check(cudaMemcpy(h_val.data(),  cur_val,  sizeof(float)   * h_val.size(),  cudaMemcpyDeviceToHost),  "D2H cur_val (batch)");
    gpu_check(cudaMemcpy(h_type.data(), cur_type, sizeof(int16_t) * h_type.size(), cudaMemcpyDeviceToHost), "D2H cur_type (batch)");
    gpu_check(cudaMemcpy(h_size.data(), cur_size, sizeof(int16_t) * h_size.size(), cudaMemcpyDeviceToHost), "D2H cur_size (batch)");

    // Build a stabilized workset (pads invalid/short to min_len_pad with even NEG pairs)
    Workset ws = build_workset_for_batch(popSize, maxGPLen, h_val, h_type, h_size, cfg);

    // Compute max program length to size device token/value staging buffers once
    int max_len = 0;
    for (unsigned int n = 0; n < popSize; ++n) max_len = std::max(max_len, ws.lengths[n]);
    if (max_len <= 0) {
        // Nothing to run; return ~0 time
        return 0.0f;
    }

    // Device staging buffers for one tree's tokens/values, reused for all trees
    int   *d_tokens = nullptr;
    float *d_values = nullptr;
    gpu_check(cudaMalloc(&d_tokens, sizeof(int)   * max_len), "cudaMalloc d_tokens(batch all)");
    gpu_check(cudaMalloc(&d_values, sizeof(float) * max_len), "cudaMalloc d_values(batch all)");

    // Output buffer for all trees [popSize, dataPoints] — primarily to keep parity with async path
    float *d_out_all = nullptr;
    gpu_check(cudaMalloc(&d_out_all, sizeof(float) * (size_t)popSize * (size_t)dataPoints), "cudaMalloc d_out_all(batch)");

    // Launch geometry (same for every tree)
    int threads = cfg.threads > 0 ? cfg.threads : (int)et_threads;
    int blocks  = (dataPoints + threads - 1) / threads;

    // -------- Warmup pass over ALL trees --------
    for (unsigned int n = 0; n < popSize; ++n) {
        const int len = ws.lengths[n];
        if (len <= 0) continue;
        const int off = ws.offsets[n];

        gpu_check(cudaMemcpy(d_tokens, ws.tokens.data() + off, sizeof(int)   * len, cudaMemcpyHostToDevice), "H2D tokens(batch warmup)");
        gpu_check(cudaMemcpy(d_values, ws.values.data() + off, sizeof(float) * len, cudaMemcpyHostToDevice), "H2D values(batch warmup)");

        float* out_slice = d_out_all + (size_t)n * (size_t)dataPoints;
        eval_tree_gpu_batch(d_tokens, d_values, d_Xdp, len, (int)varLen, (int)dataPoints, out_slice, blocks, threads);
    }
    gpu_check(cudaDeviceSynchronize(), "warmup batch(all) sync");

    // -------- Timed pass over ALL trees --------
    cudaEvent_t te0, te1;
    gpu_check(cudaEventCreate(&te0), "cudaEventCreate te0(batch all)");
    gpu_check(cudaEventCreate(&te1), "cudaEventCreate te1(batch all)");
    gpu_check(cudaEventRecord(te0), "cudaEventRecord te0(batch all)");

    for (unsigned int n = 0; n < popSize; ++n) {
        const int len = ws.lengths[n];
        if (len <= 0) continue;
        const int off = ws.offsets[n];

        gpu_check(cudaMemcpy(d_tokens, ws.tokens.data() + off, sizeof(int)   * len, cudaMemcpyHostToDevice), "H2D tokens(batch timed)");
        gpu_check(cudaMemcpy(d_values, ws.values.data() + off, sizeof(float) * len, cudaMemcpyHostToDevice), "H2D values(batch timed)");

        float* out_slice = d_out_all + (size_t)n * (size_t)dataPoints;
        eval_tree_gpu_batch(d_tokens, d_values, d_Xdp, len, (int)varLen, (int)dataPoints, out_slice, blocks, threads);
    }

    gpu_check(cudaEventRecord(te1), "cudaEventRecord te1(batch all)");
    gpu_check(cudaEventSynchronize(te1), "cudaEventSynchronize te1(batch all)");
    float ms = 0.0f; gpu_check(cudaEventElapsedTime(&ms, te0, te1), "cudaEventElapsedTime(batch all)");
    cudaEventDestroy(te0); cudaEventDestroy(te1);

    // Cleanup
    cudaFree(d_tokens);
    cudaFree(d_values);
    cudaFree(d_out_all);

    return ms;
}

// ---------- SR_fitness timing wrapper ----------
static float sr_avg(unsigned int popSize,
                    unsigned int dataPoints,
                    unsigned int maxGPLen,
                    unsigned int varLen,
                    unsigned int outLen,
                    float* vv, int16_t* tt, int16_t* ss,
                    float* d_Xdp, float* d_labels, float* d_fitness) {
    cudaEvent_t e0, e1; gpu_check(cudaEventCreate(&e0), "cudaEventCreate e0"); gpu_check(cudaEventCreate(&e1), "cudaEventCreate e1");
    gpu_check(cudaEventRecord(e0), "cudaEventRecord e0");
    SR_fitness(popSize, dataPoints, maxGPLen, varLen, outLen, true, vv, tt, ss, d_Xdp, d_labels, d_fitness, 4);
    gpu_check(cudaEventRecord(e1), "cudaEventRecord e1");
    gpu_check(cudaEventSynchronize(e1), "cudaEventSynchronize e1");
    {
        auto err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::fprintf(stderr, "SR_fitness kernel error: %s\n", cudaGetErrorString(err));
        }
    }
    float ms = 0.0f; gpu_check(cudaEventElapsedTime(&ms, e0, e1), "cudaEventElapsedTime SR");
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return ms;
}

// ---------- CLI + driver ----------

struct CmdCfg {
    unsigned int popSize = (1u << 15);
    unsigned int dataPoints = 256;
    unsigned int gens = 2;
    unsigned int et_threads = 256; // threads per block
    unsigned int enable_check = 0; // 1 to enable correctness checks
    int          pad_min_len = 8;  // batch pad target
};

static inline bool starts_with(const char* s, const char* pfx) {
    return std::strncmp(s, pfx, std::strlen(pfx)) == 0;
}

static void parse_args(int argc, char** argv, CmdCfg& cfg) {
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (std::strcmp(a, "--help") == 0 || std::strcmp(a, "-h") == 0) {
            std::printf("Usage: %s [--pop N] [--points N] [--gens N] [--threads N] [--check 0|1] [--pad_min_len N]\n", argv[0]);
            std::printf("Also supports positional: %s <pop> <points> <gens> <threads> <check>\n", argv[0]);
            std::exit(0);
        } else if (starts_with(a, "--pop")) {
            const char* v = std::strchr(a, '='); if (!v && i + 1 < argc) v = argv[++i]; else if (v) ++v; if (v) cfg.popSize = (unsigned)std::strtoul(v, nullptr, 10);
        } else if (starts_with(a, "--points")) {
            const char* v = std::strchr(a, '='); if (!v && i + 1 < argc) v = argv[++i]; else if (v) ++v; if (v) cfg.dataPoints = (unsigned)std::strtoul(v, nullptr, 10);
        } else if (starts_with(a, "--gens")) {
            const char* v = std::strchr(a, '='); if (!v && i + 1 < argc) v = argv[++i]; else if (v) ++v; if (v) cfg.gens = (unsigned)std::strtoul(v, nullptr, 10);
        } else if (starts_with(a, "--threads")) {
            const char* v = std::strchr(a, '='); if (!v && i + 1 < argc) v = argv[++i]; else if (v) ++v; if (v) cfg.et_threads = (unsigned)std::strtoul(v, nullptr, 10);
        } else if (starts_with(a, "--check")) {
            const char* v = std::strchr(a, '='); if (!v && i + 1 < argc) v = argv[++i]; else if (v) ++v; if (v) cfg.enable_check = (unsigned)std::strtoul(v, nullptr, 10);
        } else if (starts_with(a, "--pad_min_len")) {
            const char* v = std::strchr(a, '='); if (!v && i + 1 < argc) v = argv[++i]; else if (v) ++v; if (v) cfg.pad_min_len = (int)std::strtol(v, nullptr, 10);
        }
    }
    if (argc > 1 && std::isdigit((unsigned char)argv[1][0])) cfg.popSize    = (unsigned)std::strtoul(argv[1], nullptr, 10);
    if (argc > 2 && std::isdigit((unsigned char)argv[2][0])) cfg.dataPoints = (unsigned)std::strtoul(argv[2], nullptr, 10);
    if (argc > 3 && std::isdigit((unsigned char)argv[3][0])) cfg.gens       = (unsigned)std::strtoul(argv[3], nullptr, 10);
    if (argc > 4 && std::isdigit((unsigned char)argv[4][0])) cfg.et_threads = (unsigned)std::strtoul(argv[4], nullptr, 10);
    if (argc > 5 && std::isdigit((unsigned char)argv[5][0])) cfg.enable_check = (unsigned)std::strtoul(argv[5], nullptr, 10);
}

int main(int argc, char** argv) {
    CmdCfg cfg; parse_args(argc, argv, cfg);
    unsigned int popSize = cfg.popSize;
    const unsigned int varLen = 4;
    const unsigned int outLen = 1;
    const unsigned int maxGPLen = 64;

    // Generation config
    const unsigned int constSamplesLen = 8;
    const float outProb = 0.0f;
    const float constProb = 0.25f;

    // Host generation params
    std::vector<unsigned int> h_keys = {123456789u, 362436069u};
    std::vector<float> h_depth2leaf(MAX_FULL_DEPTH, 0.5f);
    h_depth2leaf[0] = 0.0f;
    if (MAX_FULL_DEPTH > 1) h_depth2leaf[1] = 0.6f;
    for (int d = 2; d < MAX_FULL_DEPTH; ++d) h_depth2leaf[d] = 0.85f;

    std::vector<float> h_roulette(Function::END);
    for (int i = 0; i < Function::END; ++i) {
        h_roulette[i] = (float)i / (float)Function::END;
    }
    std::vector<float> h_consts(constSamplesLen);
    if (constSamplesLen >= 8) {
        h_consts[0] = 0.5f; h_consts[1] = 1.0f; h_consts[2] = 2.0f; h_consts[3] = 3.0f;
        h_consts[4] = 4.0f; h_consts[5] = 6.0f; h_consts[6] = 8.0f; h_consts[7] = 10.0f;
    } else {
        for (unsigned int i = 0; i < constSamplesLen; ++i) h_consts[i] = 1.0f + (float)i;
    }

    // Device buffers for encodings and params
    float *d_val = nullptr, *d_vars = nullptr, *d_res = nullptr;
    int16_t *d_type = nullptr, *d_size = nullptr;
    unsigned int *d_keys = nullptr;
    float *d_depth2leaf = nullptr, *d_roulette = nullptr, *d_consts = nullptr;

    gpu_check(cudaMalloc(&d_val,  sizeof(float)   * popSize * maxGPLen), "cudaMalloc d_val");
    gpu_check(cudaMalloc(&d_type, sizeof(int16_t) * popSize * maxGPLen), "cudaMalloc d_type");
    gpu_check(cudaMalloc(&d_size, sizeof(int16_t) * popSize * maxGPLen), "cudaMalloc d_size");

    gpu_check(cudaMalloc(&d_keys, sizeof(unsigned int) * h_keys.size()), "cudaMalloc d_keys");
    gpu_check(cudaMalloc(&d_depth2leaf, sizeof(float) * MAX_FULL_DEPTH), "cudaMalloc d_depth2leaf");
    gpu_check(cudaMalloc(&d_roulette, sizeof(float) * Function::END), "cudaMalloc d_roulette");
    gpu_check(cudaMalloc(&d_consts, sizeof(float) * constSamplesLen), "cudaMalloc d_consts");

    gpu_check(cudaMemcpy(d_keys, h_keys.data(), sizeof(unsigned int) * h_keys.size(), cudaMemcpyHostToDevice), "H2D keys");
    gpu_check(cudaMemcpy(d_depth2leaf, h_depth2leaf.data(), sizeof(float) * MAX_FULL_DEPTH, cudaMemcpyHostToDevice), "H2D depth2leaf");
    gpu_check(cudaMemcpy(d_roulette, h_roulette.data(), sizeof(float) * Function::END, cudaMemcpyHostToDevice), "H2D roulette");
    gpu_check(cudaMemcpy(d_consts, h_consts.data(), sizeof(float) * constSamplesLen, cudaMemcpyHostToDevice), "H2D consts");

    // Generate initial population
    generate(popSize, maxGPLen, varLen, outLen, constSamplesLen, outProb, constProb,
             d_keys, d_depth2leaf, d_roulette, d_consts,
             d_val, d_type, d_size);
    gpu_check(cudaDeviceSynchronize(), "generate sync");

    // Dataset
    unsigned int dataPoints = cfg.dataPoints;
    unsigned int et_threads = cfg.et_threads;
    unsigned int enable_check = cfg.enable_check;
    std::srand(1234);
    std::vector<float> X(dataPoints * varLen);
    for (unsigned int k = 0; k < dataPoints; ++k) {
        for (unsigned int j = 0; j < varLen; ++j) {
            X[k * varLen + j] = 0.01f * (float)((k + 1) * (j + 1));
        }
    }

    // Buffers for EVOGP SR_fitness & eval benchmarks
    std::vector<float> h_vars(popSize * varLen);
    gpu_check(cudaMalloc(&d_vars, sizeof(float) * h_vars.size()), "cudaMalloc d_vars");
    gpu_check(cudaMalloc(&d_res, sizeof(float) * popSize * outLen), "cudaMalloc d_res");

    float *d_Xdp = nullptr;
    gpu_check(cudaMalloc(&d_Xdp, sizeof(float) * X.size()), "cudaMalloc d_Xdp");
    gpu_check(cudaMemcpy(d_Xdp, X.data(), sizeof(float) * X.size(), cudaMemcpyHostToDevice), "H2D Xdp");

    // Warmup SR_fitness
    float *d_labels = nullptr, *d_fitness = nullptr;
    std::vector<float> h_labels(dataPoints * outLen, 0.0f);
    gpu_check(cudaMalloc(&d_labels, sizeof(float) * h_labels.size()), "cudaMalloc d_labels");
    gpu_check(cudaMemcpy(d_labels, h_labels.data(), sizeof(float) * h_labels.size(), cudaMemcpyHostToDevice), "H2D labels");
    gpu_check(cudaMalloc(&d_fitness, sizeof(float) * popSize), "cudaMalloc d_fitness");
    SR_fitness(popSize, dataPoints, maxGPLen, varLen, outLen, true, d_val, d_type, d_size, d_Xdp, d_labels, d_fitness, 4);
    gpu_check(cudaDeviceSynchronize(), "SR_fitness warmup sync");

    // Optional deterministic correctness check (augmented counters)
    auto valid_prefix_types = [&](const int16_t* tptr, int len){ return valid_prefix_types_canon(tptr, len); };

    auto check_correctness = [&](const char* tag, float* cur_val, int16_t* cur_type, int16_t* cur_size) {
        if (!enable_check) return;
        const int sTrees = 200; const int sPoints = 1000; const float tol = 1e-4f;
        std::vector<float> h_val(popSize * maxGPLen);
        std::vector<int16_t> h_type(popSize * maxGPLen);
        std::vector<int16_t> h_size(popSize * maxGPLen);
        gpu_check(cudaMemcpy(h_val.data(),  cur_val,  sizeof(float)   * h_val.size(),  cudaMemcpyDeviceToHost),  "D2H cur_val (chk)");
        gpu_check(cudaMemcpy(h_type.data(), cur_type, sizeof(int16_t) * h_type.size(), cudaMemcpyDeviceToHost), "D2H cur_type (chk)");
        gpu_check(cudaMemcpy(h_size.data(), cur_size, sizeof(int16_t) * h_size.size(), cudaMemcpyDeviceToHost), "D2H cur_size (chk)");

        std::vector<unsigned int> trees, points;
        for (int i = 0; i < sTrees; ++i) trees.push_back((unsigned int)(std::rand() % (int)popSize));
        for (int i = 0; i < sPoints; ++i) points.push_back((unsigned int)(std::rand() % (int)dataPoints));

        size_t attempted_pairs = (size_t)trees.size() * (size_t)points.size();
        size_t checked_pairs   = 0;
        size_t finite_pairs    = 0;
        size_t skipped_pairs   = 0;
        std::vector<uint8_t> tree_had_valid(popSize, 0);

        double sum_err = 0.0, max_err = 0.0; unsigned int worst_t = 0, worst_k = 0; size_t pairs = 0;
        size_t nf_one_side = 0, nf_both = 0;
        size_t finite_logged = 0, finite_cap = 6;
        size_t skip_logged = 0, skip_cap = 16;
        std::vector<float> out_host(popSize);

        auto prefix_effective_len = [&](const int16_t* tptr, int len){ return prefix_effective_len_canon(tptr, len); };

        for (unsigned int pk = 0; pk < points.size(); ++pk) {
            unsigned int k = points[pk];
            const float* xk = &X[k * varLen];
            for (unsigned int n = 0; n < popSize; ++n) {
                float* dst = &h_vars[n * varLen];
                for (unsigned int j = 0; j < varLen; ++j) dst[j] = xk[j];
            }
            gpu_check(cudaMemcpy(d_vars, h_vars.data(), sizeof(float) * h_vars.size(), cudaMemcpyHostToDevice), "H2D vars chk");
            evaluate(popSize, maxGPLen, varLen, outLen, cur_val, cur_type, cur_size, d_vars, d_res);
            gpu_check(cudaDeviceSynchronize(), "chk eval sync");
            gpu_check(cudaMemcpy(out_host.data(), d_res, sizeof(float) * popSize, cudaMemcpyDeviceToHost), "D2H res chk");

            for (unsigned int ti = 0; ti < trees.size(); ++ti) {
                unsigned int n = trees[ti];
                size_t base = n * maxGPLen; int len_total = (int)h_size[base + 0];
                if (len_total <= 0 || !valid_prefix_types(&h_type[base], len_total)) {
                    ++skipped_pairs;
                    if (skip_logged < skip_cap) {
                        ++skip_logged;
                        std::fprintf(stderr, "[check %s] INVALID prefix tree=%u len_total=%d (k=%u)\n", tag, n, len_total, k);
                    }
                    continue;
                }

                int eff_len = prefix_effective_len(&h_type[base], len_total);
                int start = len_total - eff_len; if (start < 0) start = 0;

                std::vector<int> tok(eff_len); std::vector<float> val(eff_len);
                for (int j = 0; j < eff_len; ++j) {
                    int idx = start + j;
                    int16_t t_full = h_type[base + idx];
                    int16_t t = t_full & 0x7F;
                    if (t == 0) { tok[j] = -1; val[j] = h_val[base + idx]; } // VAR
                    else if (t == 1) { tok[j] = 0;  val[j] = h_val[base + idx]; } // CONST
                    else {
                        int f;
                        if (t_full & 0x80) {
                            uint32_t bits; std::memcpy(&bits, &h_val[base + idx], sizeof(uint32_t));
                            f = (int)(bits & 0xFFFFu);
                        } else {
                            f = (int)h_val[base + idx];
                        }
                        tok[j] = map_evogp_to_eval(f);
                        val[j] = 0.0f;
                    }
                }

                checked_pairs++;
                float out_et = 0.0f; eval_tree_gpu(tok.data(), val.data(), xk, eff_len, (int)varLen, &out_et);

                double a = (double)out_host[n];
                double b = (double)out_et;
                if (std::isfinite(a) && std::isfinite(b)) {
                    finite_pairs++; tree_had_valid[n] = 1;
                    double err = std::abs(a - b);
                    sum_err += err; ++pairs; if (err > max_err) { max_err = err; worst_t = n; worst_k = k; }
                    if (err > tol && finite_logged < finite_cap) {
                        ++finite_logged;
                        std::fprintf(stderr, "[debug %s finite-mismatch] tree=%u k=%u len=%d EVOGP=%.7g eval_tree=%.7g err=%.7g\n",
                                     tag, n, k, eff_len, a, b, err);
                    }
                } else {
                    if (std::isfinite(a) ^ std::isfinite(b)) {
                        ++nf_one_side; double err_nf = 1e9; sum_err += err_nf;
                        if (err_nf > max_err) { max_err = err_nf; worst_t = n; worst_k = k; }
                    } else {
                        ++nf_both;
                    }
                }
            }
        }
        size_t unique_trees_attempted = 200;
        size_t unique_trees_checked   = 0;
        for (unsigned int n = 0; n < popSize; ++n) unique_trees_checked += (tree_had_valid[n] ? 1u : 0u);

        double mean_err = pairs > 0 ? (sum_err / (double)pairs) : 0.0;
        if (nf_one_side + nf_both > 0) {
            std::printf("[check %s] non-finite: one-side=%zu both=%zu (penalized one-side with 1e9)\n", tag, nf_one_side, nf_both);
        }
        std::printf("[check %s] abs err mean=%.6e max=%.6e (worst: tree=%u, k=%u) tol=%.1e\n",
                    tag, mean_err, max_err, worst_t, worst_k, tol);
        std::printf("[check %s] pairs: attempted=%zu, checked=%zu, finite=%zu, skipped=%zu\n",
                    tag, attempted_pairs, checked_pairs, finite_pairs, skipped_pairs);
        std::printf("[check %s] trees: attempted_unique=%zu, checked_unique=%zu\n",
                    tag, unique_trees_attempted, unique_trees_checked);
    };

    unsigned int gens = cfg.gens;

    // Explicit timing buffers for ALL stages
    std::vector<float> sr_times; sr_times.reserve(gens + 1);
    std::vector<float> et_async_times; et_async_times.reserve(gens + 1);
    std::vector<float> et_batch_times; et_batch_times.reserve(gens + 1);

    BatchBenchCfg batch_cfg;
    batch_cfg.threads     = (int)cfg.et_threads;
    batch_cfg.min_len_pad = cfg.pad_min_len;

    std::printf("Config: pop=%u, points=%u, gens=%u, threads=%u, check=%u, pad_min_len=%d\n",
                popSize, dataPoints, gens, et_threads, enable_check, cfg.pad_min_len);
    if (et_threads % 32 != 0) {
        std::fprintf(stderr, "Warning: threads (%u) is not a multiple of warp size (32).\n", et_threads);
    }

    // initial timings
    sr_times.push_back(sr_avg(popSize, dataPoints, maxGPLen, varLen, outLen, d_val, d_type, d_size, d_Xdp, d_labels, d_fitness));
    et_async_times.push_back(bench_eval_tree_popdp_async(popSize, dataPoints, maxGPLen, varLen, et_threads, d_val, d_type, d_size, d_Xdp));
    et_batch_times.push_back(bench_eval_tree_batch(popSize, dataPoints, maxGPLen, varLen, et_threads, d_val, d_type, d_size, d_Xdp, batch_cfg));
    check_correctness("init", d_val, d_type, d_size);

    // Evolution buffers
    float *d_val_mut=nullptr, *d_val_cx=nullptr, *d_val_new=nullptr;
    int16_t *d_type_mut=nullptr, *d_size_mut=nullptr, *d_type_cx=nullptr, *d_size_cx=nullptr, *d_type_new=nullptr, *d_size_new=nullptr;
    gpu_check(cudaMalloc(&d_val_mut,  sizeof(float)   * popSize * maxGPLen), "cudaMalloc d_val_mut");
    gpu_check(cudaMalloc(&d_type_mut, sizeof(int16_t) * popSize * maxGPLen), "cudaMalloc d_type_mut");
    gpu_check(cudaMalloc(&d_size_mut, sizeof(int16_t) * popSize * maxGPLen), "cudaMalloc d_size_mut");
    gpu_check(cudaMalloc(&d_val_cx,   sizeof(float)   * popSize * maxGPLen), "cudaMalloc d_val_cx");
    gpu_check(cudaMalloc(&d_type_cx,  sizeof(int16_t) * popSize * maxGPLen), "cudaMalloc d_type_cx");
    gpu_check(cudaMalloc(&d_size_cx,  sizeof(int16_t) * popSize * maxGPLen), "cudaMalloc d_size_cx");
    gpu_check(cudaMalloc(&d_val_new,  sizeof(float)   * popSize * maxGPLen), "cudaMalloc d_val_new");
    gpu_check(cudaMalloc(&d_type_new, sizeof(int16_t) * popSize * maxGPLen), "cudaMalloc d_type_new");
    gpu_check(cudaMalloc(&d_size_new, sizeof(int16_t) * popSize * maxGPLen), "cudaMalloc d_size_new");

    std::vector<int> h_left(popSize), h_right(popSize), h_left_pos(popSize), h_right_pos(popSize), h_mut_idx(popSize);
    int *d_left=nullptr, *d_right=nullptr, *d_left_pos=nullptr, *d_right_pos=nullptr, *d_mut_idx=nullptr;
    gpu_check(cudaMalloc(&d_left,      sizeof(int) * popSize), "cudaMalloc d_left");
    gpu_check(cudaMalloc(&d_right,     sizeof(int) * popSize), "cudaMalloc d_right");
    gpu_check(cudaMalloc(&d_left_pos,  sizeof(int) * popSize), "cudaMalloc d_left_pos");
    gpu_check(cudaMalloc(&d_right_pos, sizeof(int) * popSize), "cudaMalloc d_right_pos");
    gpu_check(cudaMalloc(&d_mut_idx,   sizeof(int) * popSize), "cudaMalloc d_mut_idx");

    float *cur_val = d_val; int16_t *cur_type = d_type, *cur_size = d_size;

    for (unsigned int g = 0; g < gens; ++g) {
        // Select donors (copy-as-new before mutation)
        std::vector<int> h_donor(popSize);
        for (unsigned int i = 0; i < popSize; ++i) h_donor[i] = std::rand() % (int)popSize;
        for (unsigned int i = 0; i < popSize; ++i) {
            const size_t fbytes = sizeof(float)   * maxGPLen;
            const size_t ibytes = sizeof(int16_t) * maxGPLen;
            const size_t sbytes = sizeof(int16_t) * maxGPLen;
            gpu_check(cudaMemcpy(d_val_new  + i * maxGPLen, cur_val  + h_donor[i] * maxGPLen, fbytes, cudaMemcpyDeviceToDevice), "D2D donor val");
            gpu_check(cudaMemcpy(d_type_new + i * maxGPLen, cur_type + h_donor[i] * maxGPLen, ibytes, cudaMemcpyDeviceToDevice), "D2D donor type");
            gpu_check(cudaMemcpy(d_size_new + i * maxGPLen, cur_size + h_donor[i] * maxGPLen, sbytes, cudaMemcpyDeviceToDevice), "D2D donor size");
        }
        gpu_check(cudaDeviceSynchronize(), "gather donor trees");

        // Build mutate indices bounded by size
        {
            std::vector<int16_t> h_size_buf(popSize * maxGPLen);
            gpu_check(cudaMemcpy(h_size_buf.data(), cur_size, sizeof(int16_t) * h_size_buf.size(), cudaMemcpyDeviceToHost), "D2H cur_size for mutate");
            for (unsigned int i = 0; i < popSize; ++i) {
                int sz = (int)h_size_buf[i * maxGPLen + 0];
                if (sz <= 0) sz = 1;
                h_mut_idx[i] = std::rand() % sz;
            }
        }
        gpu_check(cudaMemcpy(d_mut_idx, h_mut_idx.data(), sizeof(int) * popSize, cudaMemcpyHostToDevice), "H2D mut_idx");

        mutate((int)popSize, (int)maxGPLen,
               cur_val, cur_type, cur_size,
               d_mut_idx,
               d_val_new, d_type_new, d_size_new,
               d_val_mut, d_type_mut, d_size_mut);
        gpu_check(cudaDeviceSynchronize(), "mutation sync");

        // Crossover parent indices/positions bounded by tree sizes
        {
            std::vector<int16_t> h_size_buf(popSize * maxGPLen);
            gpu_check(cudaMemcpy(h_size_buf.data(), d_size_mut, sizeof(int16_t) * h_size_buf.size(), cudaMemcpyDeviceToHost), "D2H size for crossover");
            for (unsigned int i = 0; i < popSize; ++i) {
                int li = std::rand() % (int)popSize;
                int ri = std::rand() % (int)popSize;
                int lsz = (int)h_size_buf[li * maxGPLen + 0];
                int rsz = (int)h_size_buf[ri * maxGPLen + 0];
                if (lsz <= 0) lsz = 1;
                if (rsz <= 0) rsz = 1;
                h_left[i] = li;
                h_right[i] = ri;
                h_left_pos[i] = std::rand() % lsz;
                h_right_pos[i] = std::rand() % rsz;
            }
        }
        gpu_check(cudaMemcpy(d_left,      h_left.data(),      sizeof(int) * popSize, cudaMemcpyHostToDevice), "H2D left");
        gpu_check(cudaMemcpy(d_right,     h_right.data(),     sizeof(int) * popSize, cudaMemcpyHostToDevice), "H2D right");
        gpu_check(cudaMemcpy(d_left_pos,  h_left_pos.data(),  sizeof(int) * popSize, cudaMemcpyHostToDevice), "H2D left_pos");
        gpu_check(cudaMemcpy(d_right_pos, h_right_pos.data(), sizeof(int) * popSize, cudaMemcpyHostToDevice), "H2D right_pos");

        crossover((int)popSize, (int)popSize, (int)maxGPLen,
                  d_val_mut, d_type_mut, d_size_mut,
                  d_left, d_right, d_left_pos, d_right_pos,
                  d_val_cx, d_type_cx, d_size_cx);
        gpu_check(cudaDeviceSynchronize(), "crossover sync");

        // Timings and checks
        float sr_ms = sr_avg(popSize, dataPoints, maxGPLen, varLen, outLen, d_val_cx, d_type_cx, d_size_cx, d_Xdp, d_labels, d_fitness);
        sr_times.push_back(sr_ms);

        float et_async_ms = bench_eval_tree_popdp_async(popSize, dataPoints, maxGPLen, varLen, et_threads, d_val_cx, d_type_cx, d_size_cx, d_Xdp);
        et_async_times.push_back(et_async_ms);

        float et_batch_ms = bench_eval_tree_batch(popSize, dataPoints, maxGPLen, varLen, et_threads, d_val_cx, d_type_cx, d_size_cx, d_Xdp, batch_cfg);
        et_batch_times.push_back(et_batch_ms);

        check_correctness("gen", d_val_cx, d_type_cx, d_size_cx);

        cur_val = d_val_cx; cur_type = d_type_cx; cur_size = d_size_cx;
    }

    // Print per-stage and totals
    std::printf("Per-stage times (ms):\n");
    std::printf("  idx | SR_fitness (ms) | et_async (ms) | et_batch (ms)\n");
    float total_sr = 0.0f, total_async = 0.0f, total_batch = 0.0f;
    for (size_t i = 0; i < et_async_times.size(); ++i) {
        float sr = (i < sr_times.size()) ? sr_times[i] : 0.0f;
        float ea = et_async_times[i];
        float eb = (i < et_batch_times.size()) ? et_batch_times[i] : 0.0f;
        std::printf("  [%zu] | %14.6f | %12.6f | %12.6f\n", i, sr, ea, eb);
        total_sr += sr; total_async += ea; total_batch += eb;
    }
    std::printf("Overall totals (ms): SR_fitness=%.6f et_async=%.6f et_batch=%.6f\n", total_sr, total_async, total_batch);

    // Cleanup
    cudaFree(d_val); cudaFree(d_type); cudaFree(d_size);
    cudaFree(d_keys); cudaFree(d_depth2leaf); cudaFree(d_roulette); cudaFree(d_consts);
    cudaFree(d_vars); cudaFree(d_res);
    cudaFree(d_val_mut); cudaFree(d_type_mut); cudaFree(d_size_mut);
    cudaFree(d_val_cx); cudaFree(d_type_cx); cudaFree(d_size_cx);
    cudaFree(d_val_new); cudaFree(d_type_new); cudaFree(d_size_new);
    cudaFree(d_left); cudaFree(d_right); cudaFree(d_left_pos); cudaFree(d_right_pos);
    cudaFree(d_mut_idx);
    cudaFree(d_Xdp);
    cudaFree(d_labels); cudaFree(d_fitness);

    return 0;
}

// Build & Run
// nvcc -O3 -std=c++17 -arch=sm_80 run_evogp.cu \
//   ../evogp/src/evogp/cuda/forward.cu ../evogp/src/evogp/cuda/generate.cu ../evogp/src/evogp/cuda/mutation.cu \
//   eval_tree.cu -I ../evogp/src/evogp/cuda -o run_evogp
// ./run_evogp --pop 512 --points 512 --gens 1 --threads 256 --check 1 --pad_min_len 8
