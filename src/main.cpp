
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

    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <digest_file> [num_runs] [warmup_runs]" << std::endl;
        std::cerr << "  num_runs: number of measured eval runs (default: 5)" << std::endl;
        std::cerr << "  warmup_runs: number of warmup runs before measurement (default: 2)" << std::endl;
        return 1;
    }

    // Check for evolution mode
    if (std::string(argv[1]) == "-evolution")
    {
        if (argc < 4)
        {
            std::cerr << "Usage: " << argv[0] << " -evolution <start_gen> <end_gen> [data_dir]" << std::endl;
            return 1;
        }
        
        int start_gen = atoi(argv[2]);
        int end_gen = atoi(argv[3]);
        std::string data_dir = (argc >= 5) ? argv[4] : "data/evolution";
        
#ifdef USE_CPU_SUBTREE
        return run_evolution_benchmark(start_gen, end_gen, data_dir);
#else
        return run_evolution_benchmark_stateless(start_gen, end_gen, data_dir);
#endif
    }

    std::string digest_file = argv[1];
    int num_runs = (argc >= 3) ? atoi(argv[2]) : 5;
    int warmup_runs = (argc >= 4) ? atoi(argv[3]) : 2;
    
    // Validate args
    if (num_runs < 1) num_runs = 1;
    if (warmup_runs < 0) warmup_runs = 0;

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
    evaluate_and_save_results(digest_file, input_info, eval_batch, main_start, num_runs, warmup_runs);

    // Clean up
    free_input_info(input_info);

    return 0;
}
