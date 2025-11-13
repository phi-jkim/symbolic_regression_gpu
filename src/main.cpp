#include "utils/utils.h"
#include "eval/evaluator.h"
#include <iostream>
#include <string>

#ifdef USE_GPU_SIMPLE
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do                                                                        \
    {                                                                         \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess)                                               \
        {                                                                     \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__      \
                      << ": " << cudaGetErrorString(err) << std::endl;        \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)
#endif

int main(int argc, char **argv)
{
    TimePoint main_start = measure_clock();

    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <digest_file>" << std::endl;
        return 1;
    }

    std::string digest_file = argv[1];

#ifdef USE_GPU_SIMPLE
    // GPU-specific initialization
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    
    if (device_count == 0)
    {
        std::cerr << "Error: No CUDA-capable GPU found!" << std::endl;
        return 1;
    }
    
    // Print GPU info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Using GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
#endif

    // Parse input file (handles both single and multiple expressions)
    InputInfo input_info = parse_input_info(digest_file);

    if (input_info.num_exprs == 0)
    {
        std::cerr << "Error: Failed to parse input file" << std::endl;
        return 1;
    }

    // Evaluate and save results (batch evaluation)
    // eval_batch is defined in evaluator.h based on compile-time flags
    evaluate_and_save_results(digest_file, input_info, eval_batch, main_start);

    // Clean up
    free_input_info(input_info);

    return 0;
}
