#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <cmath>

// Bring in the evogp kernel host API (we link forward.cu and generate.cu from evogp)
#include "../evogp/src/evogp/cuda/kernel.h"

// eval_tree single-expression evaluators from eval_tree.cu (GPU)
extern "C" void eval_tree_gpu(const int* tokens,
                               const float* values,
                               const float* features,
                               int len,
                               int num_features,
                               float* out_host);
extern "C" void eval_tree_gpu_batch(const int* tokens,
                                     const float* values,
                                     const float* X, // device pointer [dataPoints, num_features]
                                     int len,
                                     int num_features,
                                     int dataPoints,
                                     float* out_dev, // device pointer [dataPoints]
                                     int blocks,
                                     int threads);

static void gpu_check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(e));
        std::exit(1);
    }
}

struct CmdCfg {
    unsigned int popSize = (1u << 15);
    unsigned int dataPoints = 256;
    unsigned int gens = 2;
    unsigned int et_blocks = 0;   // 0 = auto
    unsigned int et_threads = 256; // threads per block
    unsigned int enable_check = 0; // 1 to enable correctness checks
};

static inline bool starts_with(const char* s, const char* pfx) {
    return std::strncmp(s, pfx, std::strlen(pfx)) == 0;
}

static void parse_args(int argc, char** argv, CmdCfg& cfg) {
    // Named flags override defaults
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (std::strcmp(a, "--help") == 0 || std::strcmp(a, "-h") == 0) {
            std::printf("Usage: %s [--pop N] [--points N] [--gens N] [--blocks N] [--threads N] [--check 0|1]\n", argv[0]);
            std::printf("Also supports positional: %s <pop> <points> <gens> <blocks> <threads> <check>\n", argv[0]);
            std::exit(0);
        } else if (starts_with(a, "--pop")) {
            const char* v = std::strchr(a, '='); if (!v && i + 1 < argc) v = argv[++i]; else if (v) ++v; if (v) cfg.popSize = (unsigned)std::strtoul(v, nullptr, 10);
        } else if (starts_with(a, "--points")) {
            const char* v = std::strchr(a, '='); if (!v && i + 1 < argc) v = argv[++i]; else if (v) ++v; if (v) cfg.dataPoints = (unsigned)std::strtoul(v, nullptr, 10);
        } else if (starts_with(a, "--gens")) {
            const char* v = std::strchr(a, '='); if (!v && i + 1 < argc) v = argv[++i]; else if (v) ++v; if (v) cfg.gens = (unsigned)std::strtoul(v, nullptr, 10);
        } else if (starts_with(a, "--blocks")) {
            const char* v = std::strchr(a, '='); if (!v && i + 1 < argc) v = argv[++i]; else if (v) ++v; if (v) cfg.et_blocks = (unsigned)std::strtoul(v, nullptr, 10);
        } else if (starts_with(a, "--threads")) {
            const char* v = std::strchr(a, '='); if (!v && i + 1 < argc) v = argv[++i]; else if (v) ++v; if (v) cfg.et_threads = (unsigned)std::strtoul(v, nullptr, 10);
        } else if (starts_with(a, "--check")) {
            const char* v = std::strchr(a, '='); if (!v && i + 1 < argc) v = argv[++i]; else if (v) ++v; if (v) cfg.enable_check = (unsigned)std::strtoul(v, nullptr, 10);
        }
    }
    // Positional fallback if provided
    // argv[1]=pop, [2]=points, [3]=gens, [4]=blocks, [5]=threads, [6]=check
    if (argc > 1 && std::isdigit(argv[1][0])) cfg.popSize    = (unsigned)std::strtoul(argv[1], nullptr, 10);
    if (argc > 2 && std::isdigit(argv[2][0])) cfg.dataPoints = (unsigned)std::strtoul(argv[2], nullptr, 10);
    if (argc > 3 && std::isdigit(argv[3][0])) cfg.gens       = (unsigned)std::strtoul(argv[3], nullptr, 10);
    if (argc > 4 && std::isdigit(argv[4][0])) cfg.et_blocks  = (unsigned)std::strtoul(argv[4], nullptr, 10);
    if (argc > 5 && std::isdigit(argv[5][0])) cfg.et_threads = (unsigned)std::strtoul(argv[5], nullptr, 10);
    if (argc > 6 && std::isdigit(argv[6][0])) cfg.enable_check = (unsigned)std::strtoul(argv[6], nullptr, 10);
}

int main(int argc, char** argv) {
    // Parameters
    CmdCfg cfg; parse_args(argc, argv, cfg);
    unsigned int popSize = cfg.popSize; // default 32768
    const unsigned int varLen = 4;      // number of variables per individual
    const unsigned int outLen = 1;      // single-output for this demo
    const unsigned int maxGPLen = 64;   // max nodes per tree

    // Generation config
    const unsigned int constSamplesLen = 8;
    const float outProb = 0.0f;   // no multi-output in this demo
    const float constProb = 0.5f; // 50% constants vs variables at leaves

    // Host arrays for generation params
    std::vector<unsigned int> h_keys = {123456789u, 362436069u};
    std::vector<float> h_depth2leaf(MAX_FULL_DEPTH, 0.5f); // 50% leaf prob at all depths
    // Encourage more functions near root if desired
    h_depth2leaf[0] = 0.0f; // never leaf at root

    std::vector<float> h_roulette(Function::END);
    for (int i = 0; i < Function::END; ++i) {
        h_roulette[i] = (float)i / (float)Function::END; // simple ascending thresholds in [0,1)
    }
    std::vector<float> h_consts(constSamplesLen);
    for (unsigned int i = 0; i < constSamplesLen; ++i) h_consts[i] = (float)((int)i - 3); // [-3,-2,-1,0,1,2,3,4]

    // Device buffers for encodings and generation params
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

    // Generate population encodings on device
    generate(popSize, maxGPLen, varLen, outLen, constSamplesLen, outProb, constProb,
             d_keys, d_depth2leaf, d_roulette, d_consts,
             d_val, d_type, d_size);
    gpu_check(cudaDeviceSynchronize(), "generate sync");

    // Create a dataset of dataPoints, each with varLen features.
    unsigned int dataPoints = cfg.dataPoints;
    unsigned int et_blocks = cfg.et_blocks;
    unsigned int et_threads = cfg.et_threads;
    unsigned int enable_check = cfg.enable_check; // 1 to enable deterministic checks
    std::srand(1234);
    std::vector<float> X(dataPoints * varLen);
    for (unsigned int k = 0; k < dataPoints; ++k) {
        for (unsigned int j = 0; j < varLen; ++j) {
            // deterministic pattern per datapoint
            X[k * varLen + j] = 0.01f * (float)((k + 1) * (j + 1));
        }
    }
    // Buffer to hold variables replicated across population for each datapoint
    std::vector<float> h_vars(popSize * varLen);
    gpu_check(cudaMalloc(&d_vars, sizeof(float) * h_vars.size()), "cudaMalloc d_vars");
    gpu_check(cudaMalloc(&d_res, sizeof(float) * popSize * outLen), "cudaMalloc d_res");
    // Buffers for eval_tree_gpu_batch
    float *d_Xdp = nullptr, *d_outdp = nullptr, *d_et_vals = nullptr;
    int *d_et_tok = nullptr;
    gpu_check(cudaMalloc(&d_Xdp, sizeof(float) * X.size()), "cudaMalloc d_Xdp");
    gpu_check(cudaMemcpy(d_Xdp, X.data(), sizeof(float) * X.size(), cudaMemcpyHostToDevice), "H2D Xdp");
    gpu_check(cudaMalloc(&d_outdp, sizeof(float) * dataPoints), "cudaMalloc d_outdp");
    gpu_check(cudaMalloc(&d_et_tok, sizeof(int) * maxGPLen), "cudaMalloc d_et_tok");
    gpu_check(cudaMalloc(&d_et_vals, sizeof(float) * maxGPLen), "cudaMalloc d_et_vals");

    // Warmup SR_fitness once (dataset-parallel)
    // Prepare labels (zeros) and fitness buffer
    float *d_labels = nullptr, *d_fitness = nullptr;
    std::vector<float> h_labels(dataPoints * outLen, 0.0f);
    gpu_check(cudaMalloc(&d_labels, sizeof(float) * h_labels.size()), "cudaMalloc d_labels");
    gpu_check(cudaMemcpy(d_labels, h_labels.data(), sizeof(float) * h_labels.size(), cudaMemcpyHostToDevice), "H2D labels");
    gpu_check(cudaMalloc(&d_fitness, sizeof(float) * popSize), "cudaMalloc d_fitness");
    // Copy dataset to device (already in d_Xdp)
    SR_fitness(popSize, dataPoints, maxGPLen, varLen, outLen, true, d_val, d_type, d_size, d_Xdp, d_labels, d_fitness, 4);
    gpu_check(cudaDeviceSynchronize(), "SR_fitness warmup sync");

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    auto sr_avg = [&](float* vv, int16_t* tt, int16_t* ss) {
        // Time a single SR_fitness call over the full dataset and all trees
        cudaEventRecord(e0);
        SR_fitness(popSize, dataPoints, maxGPLen, varLen, outLen, true, vv, tt, ss, d_Xdp, d_labels, d_fitness, 4);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        // catch any kernel error immediately
        {
            auto err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::fprintf(stderr, "SR_fitness kernel error: %s\n", cudaGetErrorString(err));
            }
        }
        float ms = 0.0f; cudaEventElapsedTime(&ms, e0, e1);
        return ms / (float)dataPoints; // report ms per datapoint for consistency
    };

    // Map EVOGP Function enum (defs.h) to eval_tree operator codes
    auto map_evogp_to_eval = [](int f) -> int {
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
    // Validate prefix using EVOGP node types (more robust than opcode mapping)
    auto valid_prefix_types = [&](const int16_t* tptr, int len) -> bool {
        int count = 0;
        for (int i = len - 1; i >= 0; --i) {
            int16_t t = tptr[i] & 0x7F;
            if (t == 0 || t == 1) { // VAR or CONST
                ++count;
            } else if (t == 2) { // UFUNC
                if (count < 1) return false; count = count - 1 + 1;
            } else if (t == 3) { // BFUNC
                if (count < 2) return false; count = count - 2 + 1;
            } else if (t == 4) { // TFUNC
                if (count < 3) return false; count = count - 3 + 1;
            } else {
                return false;
            }
        }
        return count == 1;
    };

    // Optional deterministic correctness check (excluded from timing)
    auto check_correctness = [&](const char* tag, float* cur_val, int16_t* cur_type, int16_t* cur_size) {
        if (!enable_check) return;
        // sampled random trees and random datapoints 
        const int sTrees = 100; const int sPoints = 100; const float tol = 1e-4f;
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
        size_t skipped = 0, skip_logged = 0; const size_t skip_cap = 16;
        std::vector<float> out_host(popSize);

        for (unsigned int pk = 0; pk < points.size(); ++pk) {
            unsigned int k = points[pk];
            const float* xk = &X[k * varLen];
            for (unsigned int n = 0; n < popSize; ++n) {
                float* dst = &h_vars[n * varLen];
                for (unsigned int j = 0; j < varLen; ++j) dst[j] = xk[j];
            }
            // clear any sticky error from prior kernels
            {
                auto pre = cudaDeviceSynchronize();
                if (pre != cudaSuccess) {
                    std::fprintf(stderr, "[check %s] pre-sync kernel error: %s\n", tag, cudaGetErrorString(pre));
                    return;
                }
                (void)cudaGetLastError();
            }
            gpu_check(cudaMemcpy(d_vars, h_vars.data(), sizeof(float) * h_vars.size(), cudaMemcpyHostToDevice), "H2D vars chk");
            evaluate(popSize, maxGPLen, varLen, outLen, cur_val, cur_type, cur_size, d_vars, d_res);
            gpu_check(cudaDeviceSynchronize(), "chk eval sync");
            gpu_check(cudaMemcpy(out_host.data(), d_res, sizeof(float) * popSize, cudaMemcpyDeviceToHost), "D2H res chk");

            // build eval tree compatible prefix expression 
            for (unsigned int ti = 0; ti < trees.size(); ++ti) {
                unsigned int n = trees[ti];
                size_t base = n * maxGPLen; int len = (int)h_size[base + 0];
                if (len <= 0) continue;
                std::vector<int> tok(len); std::vector<float> val(len);
                for (int i = 0; i < len; ++i) {
                    int16_t t_full = h_type[base + i];
                    int16_t t = t_full & 0x7F;
                    // EVOGP leaf type: 0=VAR, 1=CONST
                    if (t == 0) { tok[i] = -1; val[i] = h_val[base + i]; } // (leaf mapping)
                    else if (t == 1) { tok[i] = 0;  val[i] = h_val[base + i]; }
                    else {
                        int f;
                        if (t_full & 0x80) {
                            // OUT_NODE: function id is in low 16 bits of the float bit-pattern
                            uint32_t bits; std::memcpy(&bits, &h_val[base + i], sizeof(uint32_t));
                            f = (int)(bits & 0xFFFFu);
                        } else {
                            f = (int)h_val[base + i];
                        }
                        tok[i] = map_evogp_to_eval(f);
                        val[i] = 0.0f;
                    }
                }
                // validate prefix by EVOGP node types before launching
                if (!valid_prefix_types(&h_type[base], len)) {
                    ++skipped; if (skip_logged < skip_cap) { std::fprintf(stderr, "[check %s] skip invalid prefix tree=%u len=%d\n", tag, n, len); ++skip_logged; }
                    continue;
                }
                float out_et = 0.0f; eval_tree_gpu(tok.data(), val.data(), xk, len, (int)varLen, &out_et);
                double a = (double)out_host[n];
                double b = (double)out_et;
                if (!std::isfinite(a) || !std::isfinite(b)) {
                    // count as infinite error if only one side is finite; skip if both non-finite
                    if (std::isfinite(a) ^ std::isfinite(b)) { sum_err += INFINITY; max_err = INFINITY; worst_t = n; worst_k = k; }
                    continue;
                }
                double err = std::abs(a - b);
                sum_err += err; ++pairs; if (err > max_err) { max_err = err; worst_t = n; worst_k = k; }
            }
        }
        double mean_err = pairs > 0 ? (sum_err / (double)pairs) : 0.0;
        if (skipped > 0) {
            std::printf("[check %s] skipped %zu invalid prefixes (logged %zu), checked %zu pairs\n", tag, skipped, skip_logged, pairs);
        }
        std::printf("[check %s] abs err mean=%.6e max=%.6e (worst: tree=%u, k=%u) tol=%.1e\n", tag, mean_err, max_err, worst_t, worst_k, tol);
    };

    unsigned int gens = cfg.gens;
    std::vector<float> times; times.reserve(gens + 1);

    // Print effective configuration
    std::printf("Config: pop=%u, points=%u, gens=%u, blocks=%u, threads=%u, check=%u\n",
                popSize, dataPoints, gens, et_blocks, et_threads, enable_check);
    if (et_threads % 32 != 0) {
        std::fprintf(stderr, "Warning: threads (%u) is not a multiple of warp size (32).\n", et_threads);
    }

    // helper: benchmark eval_tree_gpu across the whole population (convert EVOGP -> tokens/values, then batched per tree)
    auto bench_eval_tree_pop_avg = [&](float* cur_val, int16_t* cur_type, int16_t* cur_size) {
        // Pull current population encodings to host
        std::vector<float> h_val(popSize * maxGPLen);
        std::vector<int16_t> h_type(popSize * maxGPLen);
        std::vector<int16_t> h_size(popSize * maxGPLen);
        gpu_check(cudaMemcpy(h_val.data(),  cur_val,  sizeof(float)   * h_val.size(),  cudaMemcpyDeviceToHost),  "D2H cur_val");
        gpu_check(cudaMemcpy(h_type.data(), cur_type, sizeof(int16_t) * h_type.size(), cudaMemcpyDeviceToHost), "D2H cur_type");
        gpu_check(cudaMemcpy(h_size.data(), cur_size, sizeof(int16_t) * h_size.size(), cudaMemcpyDeviceToHost), "D2H cur_size");

        // Benchmark: for each tree, build tokens/values for that tree, copy to device, and launch batched eval over datapoints
        cudaEvent_t te0, te1; cudaEventCreate(&te0); cudaEventCreate(&te1);
        float total_ms = 0.0f;
        std::vector<int>   tok(maxGPLen);
        std::vector<float> val(maxGPLen);
        for (unsigned int n = 0; n < popSize; ++n) {
            const size_t base = n * maxGPLen;
            const int len = (int)h_size[base + 0];
            // Build prefix tokens for this tree
            for (int i = 0; i < len; ++i) {
                int16_t t_full = h_type[base + i];
                int16_t t = t_full & 0x7F;
                // EVOGP leaf type: 0=VAR, 1=CONST
                if (t == 0) { tok[i] = -1; val[i] = h_val[base + i]; }
                else if (t == 1) { tok[i] = 0;  val[i] = h_val[base + i]; }
                else {
                    int f;
                    if (t_full & 0x80) {
                        uint32_t bits; std::memcpy(&bits, &h_val[base + i], sizeof(uint32_t));
                        f = (int)(bits & 0xFFFFu);
                    } else {
                        f = (int)h_val[base + i];
                    }
                    tok[i] = map_evogp_to_eval(f);
                    val[i] = 0.0f;
                }
            }
            // validate prefix by EVOGP node types before launching to avoid illegal memory access
            if (!valid_prefix_types(&h_type[base], len)) {
                // keep benchmark output clean; only log first few
                static int bench_skip_logged = 0; if (bench_skip_logged < 8) { std::fprintf(stderr, "[bench] skip invalid prefix tree=%u len=%d\n", n, len); ++bench_skip_logged; }
                continue;
            }
            // H2D once per tree
            gpu_check(cudaMemcpy(d_et_tok, tok.data(), sizeof(int) * len, cudaMemcpyHostToDevice), "H2D et_tok");
            gpu_check(cudaMemcpy(d_et_vals, val.data(), sizeof(float) * len, cudaMemcpyHostToDevice), "H2D et_vals");
            // Timed batched kernel over dataPoints
            cudaEventRecord(te0);
            eval_tree_gpu_batch(d_et_tok, d_et_vals, d_Xdp, len, (int)varLen, (int)dataPoints, d_outdp, (int)et_blocks, (int)et_threads);
            cudaEventRecord(te1);
            cudaEventSynchronize(te1);
            {
                auto err = cudaGetLastError();
                if (err != cudaSuccess) {
                    std::fprintf(stderr, "eval_tree_gpu_batch kernel error (tree %u): %s\n", n, cudaGetErrorString(err));
                }
            }
            float ms = 0.0f; cudaEventElapsedTime(&ms, te0, te1);
            total_ms += ms;
        }
        cudaEventDestroy(te0); cudaEventDestroy(te1);
        return total_ms / (float)dataPoints; // average ms per datapoint for whole population
    };

    // initial population timings (SR_fitness)
    times.push_back(sr_avg(d_val, d_type, d_size));
    std::vector<float> et_gpu_times; et_gpu_times.reserve(gens + 1);
    et_gpu_times.push_back(bench_eval_tree_pop_avg(d_val, d_type, d_size));
    check_correctness("init", d_val, d_type, d_size);

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
        // Use current population as donors for mutation (no new random generation)
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

        // Build mutate indices per individual bounded by its tree size
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

        // Build crossover parent indices and positions bounded by their tree sizes
        {
            // current sizes after mutation are in d_size_mut
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

        times.push_back(sr_avg(d_val_cx, d_type_cx, d_size_cx));
        et_gpu_times.push_back(bench_eval_tree_pop_avg(d_val_cx, d_type_cx, d_size_cx));
        check_correctness("gen", d_val_cx, d_type_cx, d_size_cx);

        cur_val = d_val_cx; cur_type = d_type_cx; cur_size = d_size_cx;
    }

    std::vector<float> out(popSize);
    gpu_check(cudaMemcpy(out.data(), d_res, sizeof(float) * popSize, cudaMemcpyDeviceToHost), "D2H res");
    std::printf("EVOGP SR_fitness avg over %u pts: ", dataPoints);
    for (size_t i = 0; i < times.size(); ++i) {
        std::printf("[%zu]=%.6f ms%s", i, times[i], (i + 1 == times.size()) ? "\n" : ", ");
    }
    float avg = 0.0f; for (float v : times) avg += v; avg /= (float)times.size();
    // Also show total time for clarity with large N
    float total_ms = avg * (float)dataPoints;
    std::printf("Overall average: %.6f ms (%.3f us/tree), total per stage: ~%.6f ms\n", avg, 1000.0f * avg / popSize, total_ms);
    // print eval_tree_gpu timing per stage (avg ms per datapoint)
    std::printf("eval_tree_gpu (batch per tree) avg over %u pts [blocks=%u, threads=%u]: ", dataPoints, et_blocks, et_threads);
    for (size_t i = 0; i < et_gpu_times.size(); ++i) {
        std::printf("[%zu]=%.6f ms%s", i, et_gpu_times[i], (i + 1 == et_gpu_times.size()) ? "\n" : ", ");
    }
    // Combined per-stage summary
    std::printf("Per-stage (ms/pt): EVOGP (SR_fitness) | eval_tree_gpu\n");
    for (size_t i = 0; i < times.size() && i < et_gpu_times.size(); ++i) {
        std::printf("  [%zu] %.6f | %.6f\n", i, times[i], et_gpu_times[i]);
    }

    // Cleanup
    cudaFree(d_val); cudaFree(d_type); cudaFree(d_size);
    cudaFree(d_keys); cudaFree(d_depth2leaf); cudaFree(d_roulette); cudaFree(d_consts);
    cudaFree(d_vars); cudaFree(d_res);
    cudaFree(d_val_mut); cudaFree(d_type_mut); cudaFree(d_size_mut);
    cudaFree(d_val_cx); cudaFree(d_type_cx); cudaFree(d_size_cx);
    cudaFree(d_val_new); cudaFree(d_type_new); cudaFree(d_size_new);
    cudaFree(d_left); cudaFree(d_right); cudaFree(d_left_pos); cudaFree(d_right_pos);
    cudaFree(d_mut_idx);
    cudaFree(d_Xdp); cudaFree(d_outdp); cudaFree(d_et_tok); cudaFree(d_et_vals);
    cudaFree(d_labels); cudaFree(d_fitness);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return 0;
}

// Build & Run (assuming current dir = symbolic_regression_gpu and evogp is outside)
// nvcc -O3 -std=c++17 -arch=sm_80 run_evogp.cu ../evogp/src/evogp/cuda/forward.cu ../evogp/src/evogp/cuda/generate.cu ../evogp/src/evogp/cuda/mutation.cu eval_tree.cu -I ../evogp/src/evogp/cuda -o run_evogp

// # Default sizes (popSize=32768, dataPoints=256, gens=2, eval_tree blocks=auto, threads=256)
// ./run_evogp

// # Smaller quick test with explicit eval_tree config (blocks auto, 512 threads)
// blocks 0 is auto select blocks in eval_tree_gpu_batch 
// ./run_evogp --pop 512 --points 512 --gens 1 --blocks 0 --threads 256 --check 1

// benchmark on actual dataset 