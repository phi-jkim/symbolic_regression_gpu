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

// parallelism over both population and data points and async loading
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

// ---------- Bench: population x datapoints async evaluator ----------

static float bench_eval_tree_popdp_async(unsigned int popSize,
                                         unsigned int dataPoints,
                                         unsigned int maxGPLen,
                                         unsigned int varLen,
                                         unsigned int et_threads,
                                         float* cur_val, int16_t* cur_type, int16_t* cur_size,
                                         float* d_Xdp) {
    auto valid_prefix_types = [&](const int16_t* tptr, int len){ return valid_prefix_types_canon(tptr, len); };

    // Pull current population encodings to host
    std::vector<float> h_val(popSize * maxGPLen);
    std::vector<int16_t> h_type(popSize * maxGPLen);
    std::vector<int16_t> h_size(popSize * maxGPLen);
    gpu_check(cudaMemcpy(h_val.data(),  cur_val,  sizeof(float)   * h_val.size(),  cudaMemcpyDeviceToHost),  "D2H cur_val (bench)");
    gpu_check(cudaMemcpy(h_type.data(), cur_type, sizeof(int16_t) * h_type.size(), cudaMemcpyDeviceToHost), "D2H cur_type (bench)");
    gpu_check(cudaMemcpy(h_size.data(), cur_size, sizeof(int16_t) * h_size.size(), cudaMemcpyDeviceToHost), "D2H cur_size (bench)");

    // Build concatenated prefix arrays
    std::vector<int>   h_offsets(popSize);
    std::vector<int>   h_lengths(popSize);
    std::vector<int>   h_tokens; h_tokens.reserve(popSize * maxGPLen);
    std::vector<float> h_values; h_values.reserve(popSize * maxGPLen);

    auto prefix_effective_len = [&](const int16_t* tptr, int len){ return prefix_effective_len_canon(tptr, len); };

    for (unsigned int n = 0; n < popSize; ++n) {
        const size_t base = n * maxGPLen;
        int len_all = (int)h_size[base + 0];
        int len = len_all;
        if (len < 0) len = 0;
        h_offsets[n] = (int)h_tokens.size();
        if (!valid_prefix_types(&h_type[base], len)) {
            h_lengths[n] = 0; // skip invalid
            continue;
        }
        // Trim to minimal valid prefix that yields a single value
        int eff_len = prefix_effective_len(&h_type[base], len);
        int start = len_all - eff_len;
        if (start < 0) start = 0;
        h_lengths[n] = eff_len;
        for (int i = start; i < len_all; ++i) {
            int16_t t_full = h_type[base + i];
            int16_t t = t_full & 0x7F;
            if (t == 0) { h_tokens.push_back(-1); h_values.push_back(h_val[base + i]); }
            else if (t == 1) { h_tokens.push_back(0);  h_values.push_back(h_val[base + i]); }
            else {
                int f;
                if (t_full & 0x80) {
                    uint32_t bits; std::memcpy(&bits, &h_val[base + i], sizeof(uint32_t));
                    f = (int)(bits & 0xFFFFu);
                } else {
                    f = (int)h_val[base + i];
                }
                h_tokens.push_back(map_evogp_to_eval(f));
                h_values.push_back(0.0f);
            }
        }
    }

    // Device buffers
    int *d_offsets_all=nullptr, *d_lengths_all=nullptr, *d_tokens_all=nullptr;
    float *d_values_all=nullptr, *d_out_all=nullptr;
    const size_t tokensN = h_tokens.size();
    gpu_check(cudaMalloc(&d_offsets_all, sizeof(int) * h_offsets.size()), "cudaMalloc offsets_all");
    gpu_check(cudaMalloc(&d_lengths_all, sizeof(int) * h_lengths.size()), "cudaMalloc lengths_all");
    if (tokensN > 0) {
        gpu_check(cudaMalloc(&d_tokens_all, sizeof(int) * tokensN), "cudaMalloc tokens_all");
        gpu_check(cudaMalloc(&d_values_all, sizeof(float) * tokensN), "cudaMalloc values_all");
    }
    gpu_check(cudaMalloc(&d_out_all, sizeof(float) * (size_t)popSize * (size_t)dataPoints), "cudaMalloc out_all");

    gpu_check(cudaMemcpy(d_offsets_all, h_offsets.data(), sizeof(int) * h_offsets.size(), cudaMemcpyHostToDevice), "H2D offsets_all");
    gpu_check(cudaMemcpy(d_lengths_all, h_lengths.data(), sizeof(int) * h_lengths.size(), cudaMemcpyHostToDevice), "H2D lengths_all");
    if (tokensN > 0) {
        gpu_check(cudaMemcpy(d_tokens_all, h_tokens.data(), sizeof(int) * tokensN, cudaMemcpyHostToDevice), "H2D tokens_all");
        gpu_check(cudaMemcpy(d_values_all, h_values.data(), sizeof(float) * tokensN, cudaMemcpyHostToDevice), "H2D values_all");
    }

    // Launch and time
    cudaEvent_t te0, te1;
    gpu_check(cudaEventCreate(&te0), "cudaEventCreate te0");
    gpu_check(cudaEventCreate(&te1), "cudaEventCreate te1");

    int threads = (int)et_threads;
    int blocks_y = (int)((dataPoints + threads - 1) / threads);

    // Warmup
    eval_tree_gpu_pop_dp_async(d_tokens_all, d_values_all, d_offsets_all, d_lengths_all,
                               (int)popSize, d_Xdp, (int)varLen, (int)dataPoints,
                               d_out_all, blocks_y, threads);
    gpu_check(cudaDeviceSynchronize(), "warmup sync");

    gpu_check(cudaEventRecord(te0), "cudaEventRecord te0");
    eval_tree_gpu_pop_dp_async(d_tokens_all, d_values_all, d_offsets_all, d_lengths_all,
                               (int)popSize, d_Xdp, (int)varLen, (int)dataPoints,
                               d_out_all, blocks_y, threads);
    gpu_check(cudaEventRecord(te1), "cudaEventRecord te1");
    gpu_check(cudaEventSynchronize(te1), "cudaEventSynchronize te1");

    {
        auto err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::fprintf(stderr, "eval_tree_gpu_pop_dp_async kernel error: %s\n", cudaGetErrorString(err));
        }
    }
    float ms = 0.0f; gpu_check(cudaEventElapsedTime(&ms, te0, te1), "cudaEventElapsedTime");
    cudaEventDestroy(te0); cudaEventDestroy(te1);

    cudaFree(d_offsets_all); cudaFree(d_lengths_all);
    if (d_tokens_all) cudaFree(d_tokens_all);
    if (d_values_all) cudaFree(d_values_all);
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
};

static inline bool starts_with(const char* s, const char* pfx) {
    return std::strncmp(s, pfx, std::strlen(pfx)) == 0;
}

static void parse_args(int argc, char** argv, CmdCfg& cfg) {
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (std::strcmp(a, "--help") == 0 || std::strcmp(a, "-h") == 0) {
            std::printf("Usage: %s [--pop N] [--points N] [--gens N] [--threads N] [--check 0|1]\n", argv[0]);
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

    // Buffers for EVOGP SR_fitness & async eval benchmark
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

    // Optional deterministic correctness check
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

        double sum_err = 0.0, max_err = 0.0; unsigned int worst_t = 0, worst_k = 0; size_t pairs = 0;
        size_t nf_one_side = 0, nf_both = 0;
        size_t finite_logged = 0, finite_cap = 6;
        size_t skipped = 0, skip_logged = 0, skip_cap = 16;
        std::vector<float> out_host(popSize);
        bool traced = false;

        auto prefix_effective_len = [&](const int16_t* tptr, int len){ return prefix_effective_len_canon(tptr, len); };

        for (unsigned int pk = 0; pk < points.size(); ++pk) {
            unsigned int k = points[pk];
            const float* xk = &X[k * varLen];
            for (unsigned int n = 0; n < popSize; ++n) {
                float* dst = &h_vars[n * varLen];
                for (unsigned int j = 0; j < varLen; ++j) dst[j] = xk[j];
            }
            (void)cudaDeviceSynchronize();
            (void)cudaGetLastError();
            gpu_check(cudaMemcpy(d_vars, h_vars.data(), sizeof(float) * h_vars.size(), cudaMemcpyHostToDevice), "H2D vars chk");
            evaluate(popSize, maxGPLen, varLen, outLen, cur_val, cur_type, cur_size, d_vars, d_res);
            gpu_check(cudaDeviceSynchronize(), "chk eval sync");
            gpu_check(cudaMemcpy(out_host.data(), d_res, sizeof(float) * popSize, cudaMemcpyDeviceToHost), "D2H res chk");

            for (unsigned int ti = 0; ti < trees.size(); ++ti) {
                unsigned int n = trees[ti];
                size_t base = n * maxGPLen; int len_total = (int)h_size[base + 0];
                if (len_total <= 0) continue;

                if (!valid_prefix_types(&h_type[base], len_total)) {
                    ++skipped;
                    if (skip_logged < skip_cap) {
                        ++skip_logged;
                        std::fprintf(stderr, "[check %s] INVALID prefix tree=%u len_total=%d (k=%u)\n", tag, n, len_total, k);
                        std::fprintf(stderr, "  types: ");
                        for (int ii = 0; ii < std::min(len_total, 64); ++ii) std::fprintf(stderr, "%s%d", ii?", ":"", (int)(h_type[base+ii] & 0x7F));
                        std::fprintf(stderr, "\n");
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

                float out_et = 0.0f; eval_tree_gpu(tok.data(), val.data(), xk, eff_len, (int)varLen, &out_et);
                double a = (double)out_host[n];
                double b = (double)out_et;
                if (!std::isfinite(a) || !std::isfinite(b)) {
                    if (std::isfinite(a) ^ std::isfinite(b)) {
                        ++nf_one_side;
                        double err_nf = 1e9;
                        sum_err += err_nf;
                        if (err_nf > max_err) { max_err = err_nf; worst_t = n; worst_k = k; }
                        if (nf_one_side <= 6) {
                            std::fprintf(stderr, "[debug %s one-side-nonfinite] tree=%u k=%u len=%d EVOGP=%.7g eval_tree=%.7g\n",
                                         tag, n, k, eff_len, a, b);
                        }
                    } else {
                        ++nf_both;
                    }
                    continue;
                }

                double err = std::abs(a - b);
                sum_err += err; ++pairs; if (err > max_err) { max_err = err; worst_t = n; worst_k = k; }
                if (err > tol && finite_logged < finite_cap) {
                    ++finite_logged;
                    std::fprintf(stderr, "[debug %s finite-mismatch] tree=%u k=%u len=%d EVOGP=%.7g eval_tree=%.7g err=%.7g\n",
                                 tag, n, k, eff_len, a, b, err);
                    std::fprintf(stderr, "  tok: "); for (int ii=0; ii<std::min(eff_len, 64); ++ii) std::fprintf(stderr, "%s%d", ii?", ":"", tok[ii]); std::fprintf(stderr, "\n");
                    std::fprintf(stderr, "  val: "); for (int ii=0; ii<std::min(eff_len, 64); ++ii) std::fprintf(stderr, "%s%.7g", ii?", ":"", val[ii]); std::fprintf(stderr, "\n");
                    std::fprintf(stderr, "  x: "); for (unsigned int jj=0;jj<varLen;++jj) std::fprintf(stderr, "%s%.7g", jj?", ":"", xk[jj]); std::fprintf(stderr, "\n");

                    // Per-token step trace (using host math float forms from <math.h>)
                    std::vector<float> st; st.reserve(eff_len);
                    std::fprintf(stderr, "[trace %s] begin per-token evaluation tree=%u k=%u len=%d\n", tag, n, k, eff_len);
                    for (int ii = eff_len - 1; ii >= 0; --ii) {
                        int ttok = tok[ii];
                        if (ttok == 0) {
                            float cv = val[ii]; st.push_back(cv);
                            std::fprintf(stderr, "  i=%d CONST v=%.7g -> sp=%zu\n", ii, cv, st.size());
                        } else if (ttok == -1) {
                            int idx2 = (int)val[ii];
                            float vx = (idx2>=0 && idx2<(int)varLen) ? xk[idx2] : 0.0f; st.push_back(vx);
                            std::fprintf(stderr, "  i=%d VAR idx=%d v=%.7g -> sp=%zu\n", ii, idx2, vx, st.size());
                        } else {
                            int ar = arity_of_eval_op(ttok);
                            if (ar == 1) {
                                float A = st.back(); st.pop_back();
                                float R;
                                switch (ttok) {
                                    case 5:  R = ::sinf(A); break;
                                    case 6:  R = ::cosf(A); break;
                                    case 7:  R = ::expf(A); break;
                                    case 8:  R = ::logf(A); break;
                                    case 18: R = ::tanf(A); break;
                                    case 19: R = ::sinhf(A); break;
                                    case 20: R = ::coshf(A); break;
                                    case 21: R = ::tanhf(A); break;
                                    case 22: R = (A==0.0f) ? -1e9f : ::logf(::fabsf(A)); break;
                                    case 23: R = (A==0.0f) ? NAN   : 1.0f/A; break;
                                    case 24: { float d = (::fabsf(A)<=1e-9f) ? (A<0?-1e-9f:1e-9f) : A; R = 1.0f/d; } break;
                                    case 25: R = -A; break;
                                    case 26: R = ::fabsf(A); break;
                                    case 27: R = ::sqrtf(A); break;
                                    case 28: R = ::sqrtf(::fabsf(A)); break;
                                    default: R = 0.0f; break;
                                }
                                st.push_back(R);
                                std::fprintf(stderr, "  i=%d %s a=%.7g -> r=%.7g sp=%zu\n", ii, op_name_short(ttok), A, R, st.size());
                            } else if (ar == 2) {
                                float A = st.back(); st.pop_back();
                                float B2 = st.back(); st.pop_back();
                                float R;
                                switch (ttok) {
                                    case 1:  R = A + B2; break;
                                    case 2:  R = A - B2; break;
                                    case 3:  R = A * B2; break;
                                    case 4:  R = (B2 == 0.0f) ? NAN : (A / B2); break;
                                    case 9:  { float denom = (::fabsf(B2)<=1e-9f) ? (B2<0?-1e-9f:1e-9f) : B2; R = A / denom; } break;
                                    case 10: R = ::powf(A, B2); break;
                                    case 11: R = (A==0.0f && B2==0.0f) ? 0.0f : ::powf(::fabsf(A), B2); break;
                                    case 12: R = (A >= B2) ? A : B2; break;
                                    case 13: R = (A <= B2) ? A : B2; break;
                                    case 14: R = (A <  B2) ? 1.0f : -1.0f; break;
                                    case 15: R = (A >  B2) ? 1.0f : -1.0f; break;
                                    case 16: R = (A <= B2) ? 1.0f : -1.0f; break;
                                    case 17: R = (A >= B2) ? 1.0f : -1.0f; break;
                                    default: R = 0.0f; break;
                                }
                                st.push_back(R);
                                std::fprintf(stderr, "  i=%d %s a=%.7g b=%.7g -> r=%.7g sp=%zu\n", ii, op_name_short(ttok), A, B2, R, st.size());
                            } else { // IF
                                float C = st.back(); st.pop_back();
                                float B2 = st.back(); st.pop_back();
                                float A = st.back(); st.pop_back();
                                float R = (A > 0.0f) ? B2 : C; st.push_back(R);
                                std::fprintf(stderr, "  i=%d IF cond=%.7g then=%.7g else=%.7g -> r=%.7g sp=%zu\n", ii, A, B2, C, R, st.size());
                            }
                        }
                    }
                    float final_r = st.empty() ? 0.0f : st.back();
                    std::fprintf(stderr, "[trace %s] end result=%.7g EVOGP=%.7g eval_tree=%.7g\n", tag, final_r, a, b);
                }
            }
        }
        double mean_err = pairs > 0 ? (sum_err / (double)pairs) : 0.0;
        if (skipped > 0) {
            std::printf("[check %s] skipped %zu invalid prefixes (logged %zu), checked %zu pairs\n", tag, skipped, skip_logged, pairs);
        }
        if (nf_one_side + nf_both > 0) {
            std::printf("[check %s] non-finite: one-side=%zu both=%zu (penalized one-side with 1e9)\n", tag, nf_one_side, nf_both);
        }
        std::printf("[check %s] abs err mean=%.6e max=%.6e (worst: tree=%u, k=%u) tol=%.1e\n", tag, mean_err, max_err, worst_t, worst_k, tol);
    };

    unsigned int gens = cfg.gens;

    // Explicit timing buffers for BOTH stages
    std::vector<float> sr_times; sr_times.reserve(gens + 1);
    std::vector<float> et_times; et_times.reserve(gens + 1);

    std::printf("Config: pop=%u, points=%u, gens=%u, threads=%u, check=%u\n",
                popSize, dataPoints, gens, et_threads, enable_check);
    if (et_threads % 32 != 0) {
        std::fprintf(stderr, "Warning: threads (%u) is not a multiple of warp size (32).\n", et_threads);
    }

    // initial timings
    sr_times.push_back(sr_avg(popSize, dataPoints, maxGPLen, varLen, outLen, d_val, d_type, d_size, d_Xdp, d_labels, d_fitness));
    et_times.push_back(bench_eval_tree_popdp_async(popSize, dataPoints, maxGPLen, varLen, et_threads, d_val, d_type, d_size, d_Xdp));
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
        // ---- FIX: add 4th argument kind for right_pos ----
        gpu_check(cudaMemcpy(d_right_pos, h_right_pos.data(), sizeof(int) * popSize, cudaMemcpyHostToDevice), "H2D right_pos");

        crossover((int)popSize, (int)popSize, (int)maxGPLen,
                  d_val_mut, d_type_mut, d_size_mut,
                  d_left, d_right, d_left_pos, d_right_pos,
                  d_val_cx, d_type_cx, d_size_cx);
        gpu_check(cudaDeviceSynchronize(), "crossover sync");

        // Timings and checks
        float sr_ms = sr_avg(popSize, dataPoints, maxGPLen, varLen, outLen, d_val_cx, d_type_cx, d_size_cx, d_Xdp, d_labels, d_fitness);
        sr_times.push_back(sr_ms);
        float et_ms = bench_eval_tree_popdp_async(popSize, dataPoints, maxGPLen, varLen, et_threads, d_val_cx, d_type_cx, d_size_cx, d_Xdp);
        et_times.push_back(et_ms);
        check_correctness("gen", d_val_cx, d_type_cx, d_size_cx);

        cur_val = d_val_cx; cur_type = d_type_cx; cur_size = d_size_cx;
    }

    // Print per-stage and totals clearly
    std::printf("Per-stage times (ms):\n");
    std::printf("  idx | SR_fitness (ms) | eval_tree_async (ms)\n");
    float total_sr = 0.0f, total_et = 0.0f;
    for (size_t i = 0; i < et_times.size(); ++i) {
        float sr = (i < sr_times.size()) ? sr_times[i] : 0.0f;
        float et = et_times[i];
        std::printf("  [%zu] | %14.6f | %18.6f\n", i, sr, et);
        total_sr += sr;
        total_et += et;
    }
    std::printf("Overall totals (ms): SR_fitness=%.6f eval_tree_async=%.6f\n", total_sr, total_et);

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
// nvcc -O3 -std=c++17 -arch=sm_80 run_evogp.cu ../evogp/src/evogp/cuda/forward.cu ../evogp/src/evogp/cuda/generate.cu ../evogp/src/evogp/cuda/mutation.cu eval_tree.cu -I ../evogp/src/evogp/cuda -o run_evogp
// ./run_evogp --pop 512 --points 512 --gens 1 --threads 256 --check 1
