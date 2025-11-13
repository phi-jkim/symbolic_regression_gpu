#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "../utils/utils.h"

// Import eval_tree_gpu_async from utils.cu
extern "C" void eval_tree_gpu_async(
    const int* tokens,
    const float* values,
    const float* X,           // [dataPoints, num_features]
    int len,
    int num_features,
    int dataPoints,
    float* out_dev,
    int blocks_y,
    int threads
);

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

static bool g_warned_about_sm = false;

void eval_async_jinha_batch(InputInfo &input_info, double ***all_vars, double **all_predictions)
{
    // Check GPU compute capability and warn if not optimal
    if (!g_warned_about_sm) {
        int device;
        cudaGetDevice(&device);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        
        int compute_capability = prop.major * 10 + prop.minor;
        
        if (compute_capability < 80) {
            std::cout << "\n⚠️  WARNING: GPU Async Double-Buffer Performance Notice" << std::endl;
            std::cout << "   Your GPU: " << prop.name << " (SM " << prop.major << "." << prop.minor << ")" << std::endl;
            std::cout << "   This GPU does not support cp.async (requires SM 8.0+/Ampere+)" << std::endl;
            std::cout << "   Using fallback synchronous copy - performance similar to batch version" << std::endl;
            std::cout << "   For optimal async performance, use GPU with SM 8.0+ (A100, RTX 30xx, etc.)" << std::endl;
            std::cout << "   Consider using 'gpu_jinha_eval' instead for similar performance with less overhead\n" << std::endl;
        } else {
            std::cout << "\n✓ GPU supports async copy (SM " << prop.major << "." << prop.minor << ")" << std::endl;
            std::cout << "  Using optimized async double-buffer evaluation\n" << std::endl;
        }
        g_warned_about_sm = true;
    }
    for (int expr_id = 0; expr_id < input_info.num_exprs; expr_id++)
    {
        int num_vars = input_info.num_vars[expr_id];
        int num_dps = input_info.num_dps[expr_id];
        int num_tokens = input_info.num_tokens[expr_id];
        
        // ============================================
        // Step 1: Copy tokens (no conversion needed)
        // ============================================
        std::vector<int> tokens_host(num_tokens);
        for (int i = 0; i < num_tokens; i++) {
            tokens_host[i] = input_info.tokens[expr_id][i];
        }
        
        // ============================================
        // Step 2: Convert values double -> float
        // ============================================
        std::vector<float> values_host(num_tokens);
        for (int i = 0; i < num_tokens; i++) {
            values_host[i] = (float)input_info.values[expr_id][i];
        }
        
        // ============================================
        // Step 3: Transpose and convert data layout
        // From: all_vars[expr_id][var_id][dp] (column-major, double)
        // To:   X_flat[dp * num_features + var_id] (row-major, float)
        // ============================================
        int num_features = num_vars + 1;  // Include output variable
        std::vector<float> X_flat(num_dps * num_features);
        
        for (int dp = 0; dp < num_dps; dp++) {
            for (int var = 0; var < num_features; var++) {
                // Transpose: [var][dp] -> [dp][var]
                // Cast: double -> float
                X_flat[dp * num_features + var] = (float)all_vars[expr_id][var][dp];
            }
        }
        
        // ============================================
        // Step 4: Allocate device memory
        // ============================================
        int *d_tokens = nullptr;
        float *d_values = nullptr;
        float *d_X = nullptr;
        float *d_out = nullptr;
        
        CUDA_CHECK(cudaMalloc(&d_tokens, num_tokens * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_values, num_tokens * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_X, X_flat.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, num_dps * sizeof(float)));
        
        // ============================================
        // Step 5: Copy to device
        // ============================================
        CUDA_CHECK(cudaMemcpy(d_tokens, tokens_host.data(), 
                             num_tokens * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_values, values_host.data(), 
                             num_tokens * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_X, X_flat.data(), 
                             X_flat.size() * sizeof(float), cudaMemcpyHostToDevice));
        
        // ============================================
        // Step 6: Call async double-buffer kernel
        // ============================================
        int threads = 256;
        int blocks_y = (num_dps + threads - 1) / threads;
        
        eval_tree_gpu_async(d_tokens, d_values, d_X, 
                           num_tokens, num_features, num_dps,
                           d_out, blocks_y, threads);
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // ============================================
        // Step 7: Copy results back and convert float -> double
        // ============================================
        std::vector<float> out_float(num_dps);
        CUDA_CHECK(cudaMemcpy(out_float.data(), d_out, 
                             num_dps * sizeof(float), cudaMemcpyDeviceToHost));
        
        for (int dp = 0; dp < num_dps; dp++) {
            all_predictions[expr_id][dp] = (double)out_float[dp];
        }
        
        // ============================================
        // Step 8: Cleanup
        // ============================================
        CUDA_CHECK(cudaFree(d_tokens));
        CUDA_CHECK(cudaFree(d_values));
        CUDA_CHECK(cudaFree(d_X));
        CUDA_CHECK(cudaFree(d_out));
    }
}
