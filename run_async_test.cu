#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <cstdint>

// extern from eval_tree.cu
extern "C" void eval_tree_gpu_pop_dp_async(
    const int* tokens_all,
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

extern "C" float eval_tree_cpu(const int* tokens,
                                const float* values,
                                const float* features,
                                int len,
                                int num_features);

static void check(cudaError_t e, const char* where) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", where, cudaGetErrorString(e));
        std::exit(1);
    }
}

static bool run_population(const char* name,
                           const std::vector<std::vector<int>>& toks,
                           const std::vector<std::vector<float>>& vals,
                           int num_features,
                           int dataPoints) {
    const int popSize = (int)toks.size();
    std::vector<int> lengths(popSize), offsets(popSize);
    size_t total_len = 0;
    for (int i = 0; i < popSize; ++i) { lengths[i] = (int)toks[i].size(); offsets[i] = (int)total_len; total_len += toks[i].size(); }
    std::vector<int> tokens_all(total_len);
    std::vector<float> values_all(total_len);
    {
        size_t p = 0;
        for (int i = 0; i < popSize; ++i) {
            for (size_t j = 0; j < toks[i].size(); ++j, ++p) {
                tokens_all[p] = toks[i][j];
                values_all[p] = vals[i][j];
            }
        }
    }

    // Build dataset X
    std::vector<float> X(dataPoints * num_features);
    for (int k = 0; k < dataPoints; ++k) {
        for (int f = 0; f < num_features; ++f) {
            X[(size_t)k * num_features + f] = 0.01f * (float)((k + 1) * (f + 1));
        }
    }

    // Device buffers
    int *d_tokens_all = nullptr, *d_offsets = nullptr, *d_lengths = nullptr;
    float *d_values_all = nullptr, *d_X = nullptr, *d_out = nullptr;
    check(cudaMalloc(&d_tokens_all, sizeof(int) * tokens_all.size()), "cudaMalloc tokens_all");
    check(cudaMalloc(&d_values_all, sizeof(float) * values_all.size()), "cudaMalloc values_all");
    check(cudaMalloc(&d_offsets, sizeof(int) * offsets.size()), "cudaMalloc offsets");
    check(cudaMalloc(&d_lengths, sizeof(int) * lengths.size()), "cudaMalloc lengths");
    check(cudaMalloc(&d_X, sizeof(float) * X.size()), "cudaMalloc X");
    check(cudaMalloc(&d_out, sizeof(float) * (size_t)popSize * (size_t)dataPoints), "cudaMalloc out");

    check(cudaMemcpy(d_tokens_all, tokens_all.data(), sizeof(int) * tokens_all.size(), cudaMemcpyHostToDevice), "H2D tokens_all");
    check(cudaMemcpy(d_values_all, values_all.data(), sizeof(float) * values_all.size(), cudaMemcpyHostToDevice), "H2D values_all");
    check(cudaMemcpy(d_offsets, offsets.data(), sizeof(int) * offsets.size(), cudaMemcpyHostToDevice), "H2D offsets");
    check(cudaMemcpy(d_lengths, lengths.data(), sizeof(int) * lengths.size(), cudaMemcpyHostToDevice), "H2D lengths");
    check(cudaMemcpy(d_X, X.data(), sizeof(float) * X.size(), cudaMemcpyHostToDevice), "H2D X");

    // Launch
    int threads = 256; // threads per block
    int blocks_y = (dataPoints + threads - 1) / threads; // tiles over datapoints
    eval_tree_gpu_pop_dp_async(d_tokens_all, d_values_all, d_offsets, d_lengths,
                               popSize, d_X, num_features, dataPoints,
                               d_out, blocks_y, threads);
    check(cudaDeviceSynchronize(), "kernel sync");
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
        cudaFree(d_tokens_all); cudaFree(d_values_all); cudaFree(d_offsets); cudaFree(d_lengths); cudaFree(d_X); cudaFree(d_out);
        return false;
    }

    // Copy results back
    std::vector<float> out((size_t)popSize * (size_t)dataPoints);
    check(cudaMemcpy(out.data(), d_out, sizeof(float) * out.size(), cudaMemcpyDeviceToHost), "D2H out");

    // Spot-check a few outputs vs CPU reference
    bool ok = true; const float tol = 1e-4f;
    for (int t = 0; t < popSize; ++t) {
        int off = offsets[t]; int len = lengths[t];
        for (int k = 0; k < std::min(8, dataPoints); ++k) {
            float cpu = eval_tree_cpu(&tokens_all[off], &values_all[off], &X[k * num_features], len, num_features);
            float gpu = out[t * dataPoints + k];
            float diff = std::fabs(cpu - gpu);
            if (!(diff <= tol || (!std::isfinite(cpu) && !std::isfinite(gpu)))) {
                std::fprintf(stderr, "[%s] Mismatch t=%d k=%d cpu=%.7f gpu=%.7f diff=%.3e\n", name, t, k, cpu, gpu, diff);
                ok = false;
                break;
            }
        }
    }

    std::printf("[%s] %s\n", name, ok ? "OK" : "FAIL");

    cudaFree(d_tokens_all); cudaFree(d_values_all); cudaFree(d_offsets); cudaFree(d_lengths);
    cudaFree(d_X); cudaFree(d_out);
    return ok;
}

int main() {
    bool all_ok = true;
    // Case 1: Basic functions: sin(x0)+cos(x1)+3, x0*x1, IF(x0, x1, 2)
    {
        std::vector<std::vector<int>> toks = {
            {1, 1, 5, -1, 6, -1, 0},      // add(add(sin(x0), cos(x1)), 3)
            {3, -1, -1},                  // mul(x0, x1)
            {29, -1, -1, 0}               // if(x0>0, x1, 2)
        };
        std::vector<std::vector<float>> vals = {
            {0, 0, 0, 0, 0, 1, 3},
            {0, 0, 1},
            {0, 0, 1, 2}
        };
        all_ok = run_population("basic", toks, vals, /*num_features=*/2, /*dataPoints=*/256) && all_ok;
    }
    // Case 2: Strict ops & NaN: DIV by zero, LOG negative, POW negative base^0.5, saturation mul(1e6,1e6)
    {
        // DIV(1, SUB(x0, x0)) => division by zero; LOG(NEG x0); POW(NEG x0, 0.5); MUL(1e6, 1e6)
        std::vector<std::vector<int>> toks = {
            {4, 0, 2, -1, -1},            // 1 / (x0 - x0)
            {8, 25, -1},                  // log(-x0)
            {10, 25, -1, 0},              // pow(-x0, 0.5)
            {3, 0, 0}                     // 1e6 * 1e6 -> clamp to 1e9
        };
        std::vector<std::vector<float>> vals = {
            {0, 1.0f, 0, 0},              // const 1; vars’ vals unused here
            {0, 0, 0},
            {0, 0, 0, 0.5f},
            {0, 1.0e6f, 1.0e6f}
        };
        all_ok = run_population("strict_ops_and_saturation", toks, vals, /*num_features=*/1, /*dataPoints=*/128) && all_ok;
    }

    std::printf("Async pop+dp tests: %s\n", all_ok ? "ALL OK" : "FAILURES");
    return all_ok ? 0 : 2;
}

// for testing make run_async_test