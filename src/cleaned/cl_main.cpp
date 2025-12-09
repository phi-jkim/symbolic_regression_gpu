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
void evaluate_cpu_mse(InputInfo& input_info, float*** all_vars, std::vector<float>& mses);

// Wrapper for GPU eval that manages static/persistent context internally or via a handle
void* create_gpu_context();
void destroy_gpu_context(void* ctx);
void evaluate_gpu_mse_wrapper(InputInfo& input_info, float*** all_vars, std::vector<float>& mses, void* ctx, bool upload_X, bool clear_cache);

// Simple GPU eval wrappers
void* create_gpu_simple_context();
void destroy_gpu_simple_context(void* ctx);
void evaluate_gpu_simple_wrapper(InputInfo& input_info, float*** all_vars, std::vector<float>& mses, void* ctx, bool upload_X);


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

    std::string csv_file = "";
    if (argc >= 6) {
        csv_file = argv[5];
    }

    std::cout << "Running Minimal Benchmark (FLOAT PRECISION)" << std::endl;
    std::cout << "Generations: " << gens << std::endl;
    std::cout << "Data Points: " << dps << std::endl;
    std::cout << "Data Directory: " << data_dir << std::endl;
    std::cout << "Verify CPU: " << (verify_cpu ? "Yes" : "No") << std::endl;
    if (!csv_file.empty()) {
        std::cout << "CSV Output: " << csv_file << std::endl;
    }

    // Load shared data
    std::string shared_data_path = data_dir + "/shared_data.txt";
    std::string first_gen_file = data_dir + "/gen_0.txt";
    InputInfo info_0 = parse_input_info(first_gen_file);
    int num_vars = info_0.num_vars[0];
    free_input_info(info_0);

    std::cout << "Loading shared data (" << num_vars << " vars)..." << std::endl;
    double** shared_data_double = load_data_file(shared_data_path, num_vars, dps);
    
    // Convert shared data to float
    // shared_data_double is [num_vars+1][dps]
    float** shared_data_float = new float*[num_vars + 1];
    for (int v = 0; v <= num_vars; v++) {
        shared_data_float[v] = new float[dps];
        for (int dp = 0; dp < dps; dp++) {
            shared_data_float[v][dp] = (float)shared_data_double[v][dp];
        }
    }
    // We can free the double data now since we use floats
    free_data(shared_data_double, num_vars);

    void* gpu_ctx = create_gpu_context();
    void* gpu_simple_ctx = create_gpu_simple_context();

    // Stats Helper
    struct GenStats {
        double time_ms;
        double avg_mse;
        float min_mse;
        float max_mse;
    };

    auto compute_stats = [](const std::vector<float>& mses, double time_ms) -> GenStats {
        GenStats s;
        s.time_ms = time_ms;
        double sum = 0;
        s.min_mse = 1e30f;
        s.max_mse = -1e30f;
        int count = 0;
        for (float v : mses) {
            if (!std::isnan(v)) {
                sum += v;
                if (v < s.min_mse) s.min_mse = v;
                if (v > s.max_mse) s.max_mse = v;
                count++;
            }
        }
        s.avg_mse = (count > 0) ? sum / count : NAN;
        return s;
    };

    std::vector<GenStats> simple_results(gens);
    
    std::ofstream csv_out;
    if (!csv_file.empty()) {
        csv_out.open(csv_file);
        csv_out << "Generation,Simple_Time_ms,Simple_AvgMSE,Simple_MinMSE,Simple_MaxMSE,"
                   "Opt_Time_ms,Opt_AvgMSE,Opt_MinMSE,Opt_MaxMSE,CPU_Time_ms\n";
    }

    // --- PHASE 1: GPU Simple ---
    // Note: upload_X = true for Gen 0
    std::cout << "\n=== Phase 1: GPU Simple Benchmark ===" << std::endl;
    for (int gen = 0; gen < gens; gen++) {
        std::string gen_file = data_dir + "/gen_" + std::to_string(gen) + ".txt";
        if (std::ifstream(gen_file).fail()) break;

        InputInfo info = parse_input_info(gen_file);
        for(int i=0; i<info.num_exprs; i++) info.num_dps[i] = dps;
        info.max_num_dps = dps;

        float*** all_vars = new float**[info.num_exprs];
        for(int i=0; i<info.num_exprs; i++) all_vars[i] = shared_data_float;

        bool upload_X = (gen == 0);
        std::vector<float> gpu_simple_mses;
        TimePoint t1 = measure_clock();
        evaluate_gpu_simple_wrapper(info, all_vars, gpu_simple_mses, gpu_simple_ctx, upload_X);
        double ms = clock_to_ms(t1, measure_clock());
        
        simple_results[gen] = compute_stats(gpu_simple_mses, ms);
        std::cout << "Gen " << gen << ": " << ms << " ms" << std::endl;

        delete[] all_vars;
        free_input_info(info);
    }

    // --- PHASE 2: GPU Optimized ---
    std::cout << "\n=== Phase 2: GPU Optimized Benchmark ===" << std::endl;
    // Reuse existing context or create new if needed. We already created gpu_ctx above.
    // If we want to ensure clean state, we can recreate it, but it handles persistent state.
    // Let's just use the existing one.
    
    for (int gen = 0; gen < gens; gen++) {
        std::string gen_file = data_dir + "/gen_" + std::to_string(gen) + ".txt";
        if (std::ifstream(gen_file).fail()) break;

        InputInfo info = parse_input_info(gen_file);
        for(int i=0; i<info.num_exprs; i++) info.num_dps[i] = dps;
        info.max_num_dps = dps;

        float*** all_vars = new float**[info.num_exprs];
        for(int i=0; i<info.num_exprs; i++) all_vars[i] = shared_data_float;

        bool upload_X = (gen == 0);
        std::vector<float> gpu_mses;
        TimePoint t1 = measure_clock();
        // clear_cache = false (default) unless experimenting
        evaluate_gpu_mse_wrapper(info, all_vars, gpu_mses, gpu_ctx, upload_X, true); 
        double ms = clock_to_ms(t1, measure_clock());
        
        GenStats opt_stats = compute_stats(gpu_mses, ms);
        std::cout << "Gen " << gen << ": " << ms << " ms" << std::endl;

        // CPU Verify (Optional, just for first few gens to save time?)
        double cpu_ms = 0;
        if (verify_cpu) {
             std::vector<float> cpu_mses;
             TimePoint t2 = measure_clock();
             evaluate_cpu_mse(info, all_vars, cpu_mses);
             cpu_ms = clock_to_ms(t2, measure_clock());
        }

        if (csv_out.is_open()) {
            GenStats& s = simple_results[gen];
            csv_out << gen << ","
                    << s.time_ms << "," << s.avg_mse << "," << s.min_mse << "," << s.max_mse << ","
                    << opt_stats.time_ms << "," << opt_stats.avg_mse << "," << opt_stats.min_mse << "," << opt_stats.max_mse << ","
                    << cpu_ms << "\n";
        }

        delete[] all_vars;
        free_input_info(info);
    }

    if (csv_out.is_open()) {
        csv_out.close();
        std::cout << "Results saved to: " << csv_file << std::endl;
    }

    destroy_gpu_context(gpu_ctx);
    destroy_gpu_simple_context(gpu_simple_ctx);
    
    // Free float data
    for (int v = 0; v <= num_vars; v++) {
        delete[] shared_data_float[v];
    }
    delete[] shared_data_float;
    
    return 0;
}
