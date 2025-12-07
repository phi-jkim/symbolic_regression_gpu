#include "../utils/utils.h"
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <algorithm>

// Forward declarations
void evaluate_cpu_mse(InputInfo& input_info, double*** all_vars, std::vector<double>& mses);

// GPU context forward declaration
struct GPUSubtreeStateContext;
// We need to define the context type here or include the header if we want to instantiate it.
// Since it's defined in .cu, we can't easily include it in .cpp without separating the struct definition.
// For this minimal version, let's just declare the function and use void* for context in main, 
// or better, declare the struct in a common header? 
// No, let's just keep it simple and assume the GPU function manages the context if we pass a void* or 
// we can just make the GPU function take a void* and cast it internally.
// But the plan said "Keep the persistent subtree cache".
// Let's define a wrapper in cl_gpu_eval.cu that manages the context.

// Wrapper for GPU eval that manages static/persistent context internally or via a handle
// Let's use a void* handle for the context
void* create_gpu_context();
void destroy_gpu_context(void* ctx);
void evaluate_gpu_mse_wrapper(InputInfo& input_info, double*** all_vars, std::vector<double>& mses, void* ctx);

// Simple GPU eval wrappers
void* create_gpu_simple_context();
void destroy_gpu_simple_context(void* ctx);
void evaluate_gpu_simple_wrapper(InputInfo& input_info, double*** all_vars, std::vector<double>& mses, void* ctx);


int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <gens> <dps> [data_dir] [verify_cpu]" << std::endl;
        return 1;
    }

    int gens = std::atoi(argv[1]);
    int dps = std::atoi(argv[2]);
    std::string data_dir = (argc >= 4) ? argv[3] : "data/evolution_test20";

    bool verify_cpu = false;
    if (argc >= 5) {
        std::string arg4 = argv[4];
        if (arg4 == "verify" || arg4 == "1" || arg4 == "true") {
            verify_cpu = true;
        }
    }

    std::cout << "Running Minimal Benchmark" << std::endl;
    std::cout << "Generations: " << gens << std::endl;
    std::cout << "Data Points: " << dps << std::endl;
    std::cout << "Data Directory: " << data_dir << std::endl;
    std::cout << "Verify CPU: " << (verify_cpu ? "Yes" : "No") << std::endl;

    // Load shared data
    // We need to load the shared data file from the directory
    // The utils/utils.cpp has load_data_file but it expects a specific format or filename?
    // Let's look at how run_evolution_benchmark works.
    // It usually loads "shared_data.txt" if present.
    
    std::string shared_data_path = data_dir + "/shared_data.txt";
    // We need to know num_vars to load data.
    // Let's peek at the first generation file to get num_vars.
    std::string first_gen_file = data_dir + "/gen_0.txt";
    InputInfo info_0 = parse_input_info(first_gen_file);
    int num_vars = info_0.num_vars[0];
    free_input_info(info_0);

    std::cout << "Loading shared data (" << num_vars << " vars)..." << std::endl;
    double** shared_data = load_data_file(shared_data_path, num_vars, dps);
    
    // We need to wrap this in double*** for the evaluators (which expect [expr][var][dp])
    // But since it's shared data, we can just point all expressions to this same data.
    // However, the evaluators take double*** all_vars.
    // We'll handle this per generation.

    void* gpu_ctx = create_gpu_context();
    void* gpu_simple_ctx = create_gpu_simple_context();

    for (int gen = 0; gen < gens; gen++) {
        std::string gen_file = data_dir + "/gen_" + std::to_string(gen) + ".txt";
        if (std::ifstream(gen_file).fail()) {
            std::cerr << "Generation file not found: " << gen_file << std::endl;
            break;
        }

        InputInfo info = parse_input_info(gen_file);
        // Override num_dps in info with our requested dps
        for(int i=0; i<info.num_exprs; i++) {
            info.num_dps[i] = dps;
        }
        info.max_num_dps = dps;

        // Prepare all_vars
        double*** all_vars = new double**[info.num_exprs];
        for(int i=0; i<info.num_exprs; i++) {
            all_vars[i] = shared_data;
        }

        std::cout << "\n--- Generation " << gen << " (" << info.num_exprs << " exprs) ---" << std::endl;

        // CPU Eval
        std::vector<double> cpu_mses;
        if (verify_cpu) {
            TimePoint t1 = measure_clock();
            evaluate_cpu_mse(info, all_vars, cpu_mses);
            double cpu_ms = clock_to_ms(t1, measure_clock());
            std::cout << "CPU Time: " << cpu_ms << " ms" << std::endl;
        } else {
            std::cout << "CPU Time: Skipped" << std::endl;
        }

        // GPU Simple Eval
        std::vector<double> gpu_simple_mses;
        TimePoint t1 = measure_clock();
        evaluate_gpu_simple_wrapper(info, all_vars, gpu_simple_mses, gpu_simple_ctx);
        double gpu_simple_ms = clock_to_ms(t1, measure_clock());
        std::cout << "GPU Simple Time: " << gpu_simple_ms << " ms" << std::endl;

        // GPU Optimized Eval
        std::vector<double> gpu_mses;
        t1 = measure_clock();
        evaluate_gpu_mse_wrapper(info, all_vars, gpu_mses, gpu_ctx);
        double gpu_ms = clock_to_ms(t1, measure_clock());
        std::cout << "GPU Opt Time: " << gpu_ms << " ms" << std::endl;

        // Verify
        if (verify_cpu) {
            double max_diff = 0.0;
            for(int i=0; i<info.num_exprs; i++) {
                double diff = std::abs(cpu_mses[i] - gpu_mses[i]);
                if (std::isnan(cpu_mses[i]) && std::isnan(gpu_mses[i])) diff = 0.0;
                if (diff > max_diff) max_diff = diff;
            }
            std::cout << "Max MSE Diff (CPU vs Opt): " << max_diff << std::endl;
            
            // Print first 5 MSEs
            std::cout << "Sample MSEs (CPU vs Simple vs Opt):" << std::endl;
            for(int i=0; i<std::min(5, info.num_exprs); i++) {
                std::cout << "  Expr " << i << ": " << cpu_mses[i] << " vs " << gpu_simple_mses[i] << " vs " << gpu_mses[i] << std::endl;
            }
        } else {
            // Verify Simple vs Opt
            double max_diff = 0.0;
            for(int i=0; i<info.num_exprs; i++) {
                double diff = std::abs(gpu_simple_mses[i] - gpu_mses[i]);
                if (std::isnan(gpu_simple_mses[i]) && std::isnan(gpu_mses[i])) diff = 0.0;
                if (diff > max_diff) max_diff = diff;
            }
            std::cout << "Max MSE Diff (Simple vs Opt): " << max_diff << std::endl;

            // Print first 5 GPU MSEs
            std::cout << "Sample MSEs (Simple vs Opt):" << std::endl;
            for(int i=0; i<std::min(5, info.num_exprs); i++) {
                std::cout << "  Expr " << i << ": " << gpu_simple_mses[i] << " vs " << gpu_mses[i] << std::endl;
            }
        }

        delete[] all_vars;
        free_input_info(info);
    }

    destroy_gpu_context(gpu_ctx);
    destroy_gpu_simple_context(gpu_simple_ctx);
    free_data(shared_data, num_vars); // Actually num_vars+1 rows
    
    return 0;
}
