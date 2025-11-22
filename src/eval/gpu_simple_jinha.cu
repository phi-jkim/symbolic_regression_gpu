#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include "../utils/utils.h"

// Import eval_tree_gpu_batch from utils.cu
extern "C" void eval_tree_gpu_batch(
    const int* tokens,
    const float* values,
    const float* X,           // [dataPoints, num_features]
    int len,
    int num_features,
    int dataPoints,
    float* out_dev,
    int blocks,
    int threads
);

#ifndef MAX_EVAL_STACK
#define MAX_EVAL_STACK 52
#endif

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

void eval_jinha_batch(InputInfo &input_info, double ***all_vars, double **all_predictions)
{
    double alloc_ms = 0.0;
    double memcpy_h2d_ms_total = 0.0;
    double memcpy_d2h_ms_total = 0.0;
    float total_gpu_ms = 0.0f; // sum of kernel times over expressions (excluding expr 0)
    float total_gpu_ms_incl0 = 0.0f; // sum including expr 0
    int   counted_exprs = 0;   // number of expressions included in totals/averages
    std::vector<float> gpu_ms_list; // per-expression kernel times (excluding expr 0)
    // allocate device buffers once using precomputed maxima in InputInfo
    int *d_tokens = nullptr;
    float *d_values = nullptr;
    float *d_X = nullptr;
    float *d_out = nullptr;
    
    TimePoint t_alloc = measure_clock();
    CUDA_CHECK(cudaMalloc(&d_tokens, (size_t)input_info.max_tokens * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values, (size_t)input_info.max_tokens * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_X, (size_t)input_info.max_num_dps * (size_t)input_info.max_num_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)input_info.max_num_dps * sizeof(float)));
    CUDA_CHECK(cudaDeviceSynchronize());
    alloc_ms = clock_to_ms(t_alloc, measure_clock());

    // reusable timing events
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    for (int expr_id = 0; expr_id < input_info.num_exprs; expr_id++)
    {
        int num_vars = input_info.num_vars[expr_id];
        int num_dps = input_info.num_dps[expr_id];
        int num_tokens = input_info.num_tokens[expr_id];
        int num_features = num_vars + 1;  // Include output variable
        
        // ============================================
        // Step 1: Copy tokens (no conversion needed)
        // ============================================
        const int* tokens_host_ptr = nullptr;
        std::vector<int> tokens_temp;
        if (input_info.tokens_packed && input_info.tokens_packed[expr_id]) {
            tokens_host_ptr = input_info.tokens_packed[expr_id];
        } else {
            // redudant copying
            tokens_temp.resize(num_tokens);
            for (int i = 0; i < num_tokens; i++) 
                tokens_temp[i] = input_info.tokens[expr_id][i];
            tokens_host_ptr = tokens_temp.data();
        }
        
        // ============================================
        // Step 2: Convert values double -> float
        // ============================================
        const float* values_host_ptr = nullptr;
        std::vector<float> values_temp;
        if (input_info.values_packed_f32 && input_info.values_packed_f32[expr_id]) {
            values_host_ptr = input_info.values_packed_f32[expr_id];
        } else {
            // redudant copying
            values_temp.resize(num_tokens);
            for (int i = 0; i < num_tokens; i++) 
                values_temp[i] = (float)input_info.values[expr_id][i];
            values_host_ptr = values_temp.data();
        }
        
        const float* X_host_ptr = nullptr;
        std::vector<float> X_temp;
        if (input_info.X_packed_f32 && input_info.X_packed_f32[expr_id]) {
            X_host_ptr = input_info.X_packed_f32[expr_id];
        } else {
            X_temp.resize((size_t)num_dps * (size_t)num_features);
            for (int dp = 0; dp < num_dps; dp++) {
                for (int var = 0; var < num_features; var++) {
                    X_temp[(size_t)dp * (size_t)num_features + var] = (float) all_vars[expr_id][var][dp];
                }
            }
            X_host_ptr = X_temp.data();
        }
        
        // ============================================
        // Step 4: Device memory already allocated (reused)
        // ============================================
        
        // ============================================
        // Step 5: Copy to device
        // ============================================
        TimePoint t_h2d = measure_clock();
        CUDA_CHECK(cudaMemcpy(d_tokens, tokens_host_ptr, 
                             num_tokens * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_values, values_host_ptr, 
                             num_tokens * sizeof(float), cudaMemcpyHostToDevice));
        
        // Host-side early-exit check (no device symbol):
        const bool host_early = (num_tokens > MAX_EVAL_STACK);
        fprintf(stderr,
                "[expr %d] host_early_exit(len>%d): %s (len=%d)\n",
                expr_id, MAX_EVAL_STACK, host_early ? "YES" : "NO", num_tokens);
        
        CUDA_CHECK(cudaMemcpy(d_X, X_host_ptr,
                              (size_t)num_dps * (size_t)num_features * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
        memcpy_h2d_ms_total += clock_to_ms(t_h2d, measure_clock());
        
        // ============================================
        // Step 6: Call eval_tree_gpu_batch kernel
        // ============================================
        int threads = 128;
        // int threads=64; 
        int blocks = (num_dps + threads - 1) / threads;
        
        // Time only the GPU kernel execution
        CUDA_CHECK(cudaEventRecord(ev_start));

        eval_tree_gpu_batch(d_tokens, d_values, d_X, 
                           num_tokens, num_features, num_dps,
                           d_out, blocks, threads);

        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        total_gpu_ms_incl0 += ms;
        if (expr_id == 0) {
            // Print first expression's time but do not add it to totals
            std::cout << "[expr 0] kernel time: " << ms << " ms (excluded from totals)" << std::endl;
        } else {
            total_gpu_ms += ms;
            counted_exprs += 1;
            gpu_ms_list.push_back(ms);
        }

        // ============================================
        // Step 7: Copy results back and convert float -> double
        // ============================================
        std::vector<float> out_float(num_dps);
        TimePoint t_d2h = measure_clock();
        CUDA_CHECK(cudaMemcpy(out_float.data(), d_out,
                              num_dps * sizeof(float), cudaMemcpyDeviceToHost));
        memcpy_d2h_ms_total += clock_to_ms(t_d2h, measure_clock());
        for (int dp = 0; dp < num_dps; dp++) {
            all_predictions[expr_id][dp] = (double)out_float[dp];
        }
    }
    // cleanup reusable resources
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaFree(d_tokens));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_out));
    float median_gpu_ms = 0.0f;
    if (counted_exprs > 0) {
        std::sort(gpu_ms_list.begin(), gpu_ms_list.end());
        if (counted_exprs % 2 == 1) {
            median_gpu_ms = gpu_ms_list[counted_exprs / 2];
        } else {
            median_gpu_ms = 0.5f * (gpu_ms_list[counted_exprs / 2 - 1] + gpu_ms_list[counted_exprs / 2]);
        }
    }
    std::cout << "GPU computation time (kernels only): total_excl_expr0=" << total_gpu_ms
              << " ms, median_per_expr_excl_expr0=" << median_gpu_ms
              << " ms over " << counted_exprs << " exprs" << std::endl;
    std::cout << "Total kernel time (incl expr0): " << total_gpu_ms_incl0 << " ms" << std::endl;
    std::cout << "Alloc time (host wall): " << alloc_ms << " ms" << std::endl;
    std::cout << "Total H2D memcpy time (host wall): " << memcpy_h2d_ms_total << " ms" << std::endl;
    std::cout << "Total D2H memcpy time (host wall): " << memcpy_d2h_ms_total << " ms" << std::endl;
}
