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

void eval_jinha_batch(InputInfo &input_info, double ***all_vars, double **all_predictions, EvalMetrics* metrics)
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

    // Optional shared-data optimization for X (features) buffer:
    // if all expressions share the same dataset, build the host X buffer once
    // from expression 0 and reuse it for all expressions.
    const bool use_shared_data = (input_info.has_shared_data && input_info.num_exprs > 1);
    const float* X_shared_host_ptr = nullptr;
    std::vector<float> X_shared_temp;
    if (use_shared_data)
    {
        int shared_num_vars = input_info.num_vars[0];
        int shared_num_dps = input_info.num_dps[0];
        int shared_num_features = shared_num_vars + 1;

        if (input_info.X_packed_f32 && input_info.X_packed_f32[0])
        {
            // Reuse pre-packed shared X buffer from expression 0
            X_shared_host_ptr = input_info.X_packed_f32[0];
        }
        else
        {
            // Build a single shared X buffer from all_vars[0]
            X_shared_temp.resize((size_t)shared_num_dps * (size_t)shared_num_features);
            for (int dp = 0; dp < shared_num_dps; dp++)
            {
                for (int var = 0; var < shared_num_features; var++)
                {
                    X_shared_temp[(size_t)dp * (size_t)shared_num_features + var] =
                        (float)all_vars[0][var][dp];
                }
            }
            X_shared_host_ptr = X_shared_temp.data();
        }
        // One-time H2D copy of shared X into d_X
        TimePoint t_h2d_shared_X = measure_clock();
        CUDA_CHECK(cudaMemcpy(d_X, X_shared_host_ptr,
                              (size_t)shared_num_dps * (size_t)shared_num_features * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
        memcpy_h2d_ms_total += clock_to_ms(t_h2d_shared_X, measure_clock());
    }

    float *out_float = nullptr;
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
        if (!use_shared_data)
        {
            // Non-shared path: build/capture per-expression X and copy each time
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
        // const bool host_early = (num_tokens > MAX_EVAL_STACK);
        // fprintf(stderr,
        //         "[expr %d] host_early_exit(len>%d): %s (len=%d)\n",
        //         expr_id, MAX_EVAL_STACK, host_early ? "YES" : "NO", num_tokens);
        
        if (!use_shared_data)
        {
            CUDA_CHECK(cudaMemcpy(d_X, X_host_ptr,
                                  (size_t)num_dps * (size_t)num_features * sizeof(float), cudaMemcpyHostToDevice));
        }
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
        // std::vector<float> out_float(num_dps);
        CUDA_CHECK(cudaHostAlloc(&out_float,
                                 num_dps * sizeof(float),
                                 cudaHostAllocDefault));
        TimePoint t_d2h = measure_clock();
        CUDA_CHECK(cudaMemcpy(out_float, d_out,
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
    CUDA_CHECK(cudaFreeHost(out_float));
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

// Helper: Flatten 2D host array to 1D for GPU transfer
float* flatten_vars_jinha(double **vars, int num_vars, int num_dps)
{
    float *flat = new float[(num_vars + 1) * num_dps];
    
    for (int i = 0; i <= num_vars; i++)
    {
        for (int dp = 0; dp < num_dps; dp++)
        {
            flat[i * num_dps + dp] = (float)vars[i][dp];
        }
    }
    
    return flat;
}

// Import the kernel launch wrapper from utils.cu
extern "C" void launch_eval_prefix_kernel_multi_expression_batch(
    int *d_tokens_batch, float *d_values_batch, int *d_token_offsets, int *d_num_tokens,
    float *d_vars_flat, float *d_pred_batch, int num_vars, int num_dps, int num_exprs, int exprs_per_block,
    int blocks_x, int blocks_y, int threads_per_block);

// Batch evaluation function for GPU (matches MultiEvalFunc signature 
// assumes data is shared 
void eval_batch_multi_expression_single_kernel(InputInfo &input_info, double ***all_vars, double **all_predictions, EvalMetrics* metrics)
{
    float total_h2d_time = 0.0f;
    float total_kernel_time = 0.0f;
    float total_d2h_time = 0.0f;
    int num_kernel_launches = 0;
    
    // Use first expression's dimensions (assuming shared data)
    int num_vars = input_info.num_vars[0];
    int num_dps = input_info.num_dps[0];
    int num_exprs = input_info.num_exprs;
    
    // Calculate total tokens needed and offsets for each expression
    std::vector<int> token_offsets(num_exprs);
    std::vector<int> num_tokens_array(num_exprs);
    int total_tokens = 0;
    
    for (int expr_id = 0; expr_id < num_exprs; expr_id++) {
        token_offsets[expr_id] = total_tokens;
        num_tokens_array[expr_id] = input_info.num_tokens[expr_id];
        total_tokens += input_info.num_tokens[expr_id];
    }
    
    // Allocate and transfer shared data once
    float *d_vars_flat = nullptr;
    float *h_vars_flat = nullptr;
    double **vars = all_vars[0];
    
    CUDA_CHECK(cudaMalloc(&d_vars_flat, (num_vars + 1) * num_dps * sizeof(float)));
    h_vars_flat = flatten_vars_jinha(vars, num_vars, num_dps);
    
    TimePoint t_h2d = measure_clock();
    CUDA_CHECK(cudaMemcpy(d_vars_flat, h_vars_flat, 
                          (num_vars + 1) * num_dps * sizeof(float), 
                          cudaMemcpyHostToDevice));
    float h2d_time = clock_to_ms(t_h2d, measure_clock());
    total_h2d_time += h2d_time;
    
    // Allocate and pack all tokens and values into contiguous arrays
    int *d_tokens_batch = nullptr;
    float *d_values_batch = nullptr;
    int *d_token_offsets = nullptr;
    int *d_num_tokens = nullptr;
    float *d_pred_batch = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_tokens_batch, total_tokens * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values_batch, total_tokens * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_token_offsets, num_exprs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_num_tokens, num_exprs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pred_batch, num_exprs * num_dps * sizeof(float)));
    
    // Pack tokens and values from all expressions
    int *h_tokens_batch = new int[total_tokens];
    float *h_values_batch = new float[total_tokens];
    
    for (int expr_id = 0; expr_id < num_exprs; expr_id++) {
        int offset = token_offsets[expr_id];
        int num_tokens = input_info.num_tokens[expr_id];
        int *tokens = input_info.tokens[expr_id];
        double *values = input_info.values[expr_id];
        
        for (int i = 0; i < num_tokens; i++) {
            h_tokens_batch[offset + i] = tokens[i];
            h_values_batch[offset + i] = (float)values[i];
        }
    }
    
    // Transfer packed data to GPU
    TimePoint t_h2d2 = measure_clock();
    CUDA_CHECK(cudaMemcpy(d_tokens_batch, h_tokens_batch, total_tokens * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values_batch, h_values_batch, total_tokens * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_token_offsets, token_offsets.data(), num_exprs * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_num_tokens, num_tokens_array.data(), num_exprs * sizeof(int), cudaMemcpyHostToDevice));
    float h2d_time2 = clock_to_ms(t_h2d2, measure_clock());
    total_h2d_time += h2d_time2;
    
    // Launch kernel: each threadblock handles multiple expressions and subset of datapoints
    int threads_per_block = 128; // Fixed thread count per block
    int exprs_per_block = 4; // Each threadblock processes 4 expressions
    int blocks_x = (num_exprs + exprs_per_block - 1) / exprs_per_block; // Expression dimension
    int blocks_y = (num_dps + threads_per_block - 1) / threads_per_block; // Datapoint dimension
    dim3 grid(blocks_x, blocks_y);
    
    // Time only the GPU kernel execution using CUDA events
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    
    CUDA_CHECK(cudaEventRecord(ev_start));
    launch_eval_prefix_kernel_multi_expression_batch(
        d_tokens_batch, d_values_batch, d_token_offsets, d_num_tokens,
        d_vars_flat, d_pred_batch, num_vars, num_dps, num_exprs, exprs_per_block,
        blocks_x, blocks_y, threads_per_block);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    float kernel_time = ms;
    total_kernel_time += kernel_time;
    num_kernel_launches++;
    
    // Cleanup events
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    
    // Check for kernel errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Transfer results back (D2H)
    // float *h_pred_batch = new float[num_exprs * num_dps];
    float *h_pred_batch = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h_pred_batch,
                             num_exprs * num_dps * sizeof(float),
                             cudaHostAllocDefault));
    TimePoint t_d2h = measure_clock();
    CUDA_CHECK(cudaMemcpy(h_pred_batch, d_pred_batch, num_exprs * num_dps * sizeof(float), cudaMemcpyDeviceToHost));
    float d2h_time = clock_to_ms(t_d2h, measure_clock());
    total_d2h_time += d2h_time;
    
    // Unpack results back to individual prediction arrays
    for (int expr_id = 0; expr_id < num_exprs; expr_id++) {
        double *pred = all_predictions[expr_id];
        for (int dp = 0; dp < num_dps; dp++) {
            pred[dp] = (double)h_pred_batch[expr_id * num_dps + dp];
        }
    }
    
    // Free GPU memory
    CUDA_CHECK(cudaFree(d_tokens_batch));
    CUDA_CHECK(cudaFree(d_values_batch));
    CUDA_CHECK(cudaFree(d_token_offsets));
    CUDA_CHECK(cudaFree(d_num_tokens));
    CUDA_CHECK(cudaFree(d_pred_batch));
    CUDA_CHECK(cudaFree(d_vars_flat));
    
    // Free host memory
    delete[] h_vars_flat;
    delete[] h_tokens_batch;
    delete[] h_values_batch;
    CUDA_CHECK(cudaFreeHost(h_pred_batch));
    
    // Fill metrics if provided
    if (metrics != nullptr) {
        metrics->h2d_transfer_ms = total_h2d_time;
        metrics->kernel_exec_ms = total_kernel_time;
        metrics->d2h_transfer_ms = total_d2h_time;
        metrics->total_gpu_ms = total_h2d_time + total_kernel_time + total_d2h_time;
        metrics->num_kernel_launches = num_kernel_launches;
    }
    
    // Print timing breakdown
    std::cout << "\nGPU Multi-Expression Batch Timing (Jinha):" << std::endl;
    std::cout << "  H→D Transfer:   " << total_h2d_time << " ms" << std::endl;
    std::cout << "  Kernel Exec:    " << total_kernel_time << " ms" << std::endl;
    std::cout << "  D→H Transfer:   " << total_d2h_time << " ms" << std::endl;
    std::cout << "  GPU Subtotal:   " << (total_h2d_time + total_kernel_time + total_d2h_time) << " ms" << std::endl;
    std::cout << "  Expressions:    " << num_exprs << std::endl;
    std::cout << "  Total Tokens:   " << total_tokens << std::endl;
    std::cout << "  Threads/Block:  " << threads_per_block << " (fixed)" << std::endl;
    std::cout << "  Grid Size:       " << blocks_x << "x" << blocks_y << " (expressions x datapoints)" << std::endl;
    std::cout << "  Exprs/Block:    " << exprs_per_block << std::endl;
    std::cout << "  Method:         Single kernel for all expressions" << std::endl;
}
