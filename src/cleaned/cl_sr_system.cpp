#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <fstream>
#include <chrono>
#include <ctime>
#include <iomanip>
#include "../utils/utils.h"

// Evaluator signatures
// function prototypes
void evaluate_cpu_mse(InputInfo& input_info, float*** all_vars, std::vector<float>& mses, RunStats& stats);
void evaluate_gpu_mse_wrapper(InputInfo& input_info, float*** all_vars, std::vector<float>& mses, void* ctx, bool upload_X, bool clear_cache, RunStats& stats);
void evaluate_gpu_simple_wrapper(InputInfo& input_info, float*** all_vars, std::vector<float>& mses, void* ctx, bool upload_X, RunStats& stats);
void evaluate_gpu_ptx_wrapper(InputInfo& input_info, float*** all_vars, std::vector<float>& mses, void* ctx, bool upload_X, RunStats& stats);

void* create_gpu_context();
void destroy_gpu_context(void* ctx);
void* create_gpu_simple_context();
void destroy_gpu_simple_context(void* ctx);
void* create_gpu_ptx_context();
void destroy_gpu_ptx_context(void* ctx);

struct Individual {
    std::vector<int> tokens;
    std::vector<float> values;
    float fitness; // MSE
};

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <pop_size> <gens> [data_dir]" << std::endl;
        std::cerr << "Usage: " << argv[0] << " <pop_size> <gens> [data_dir] [limit_dps]" << std::endl;
        return 1;
    }

    int pop_size = std::atoi(argv[1]);
    int gens = std::atoi(argv[2]);
    std::string data_dir = (argc >= 4) ? argv[3] : "../../data/evolution_test20";
    int limit_dps = (argc >= 5) ? std::atoi(argv[4]) : 500000;
    
    std::string eval_name = "Unknown";
    void* eval_ctx = nullptr; // Initialize to nullptr
    (void)eval_ctx; // Silence unused warning for CPU build
    #ifdef USE_GPU_SUBTREE
    eval_name = "GPU Optimized (Subtree)";
    eval_ctx = create_gpu_context();
    #elif defined(USE_GPU_SIMPLE)
    eval_name = "GPU Simple";
    eval_ctx = create_gpu_simple_context();
    #elif defined(USE_GPU_PTX)
    eval_name = "GPU PTX";
    eval_ctx = create_gpu_ptx_context();
    #elif defined(USE_CPU)
    eval_name = "CPU";
    #else
    #error "Must define USE_GPU_SUBTREE, USE_GPU_SIMPLE, USE_GPU_PTX, or USE_CPU"
    #endif

    std::cout << "Starting Minimal SR System (" << eval_name << ")" << std::endl;
    std::cout << "Population: " << pop_size << ", Gens: " << gens << std::endl;
    std::cout << "Data Dir: " << data_dir << std::endl;
    if (limit_dps > 0) std::cout << "Limit DPS: " << limit_dps << std::endl;

    // Load Data
    std::string shared_data_path = data_dir + "/shared_data.txt";
    std::string header_file = data_dir + "/gen_0.txt"; // To get num_vars
    InputInfo info_0 = parse_input_info(header_file);
    if (info_0.num_exprs == 0) {
        std::cerr << "Failed to load header: " << header_file << std::endl;
        return 1;
    }
    int num_vars = info_0.num_vars[0]; // Input vars
    int file_dps = info_0.num_dps[0];
    free_input_info(info_0);

    // Determine actual DPS to use
    int num_dps = file_dps;
    if (limit_dps > 0 && limit_dps < file_dps) {
        num_dps = limit_dps;
    }

    // Load shared data (double) and convert to float
    // Note: load_data_file currently loads ALL data. We will just copy the first num_dps.
    double** shared_data_double = load_data_file(shared_data_path, num_vars, file_dps); 
    float** shared_data_float = new float*[num_vars + 1];
    for(int i=0; i<=num_vars; i++) {
        shared_data_float[i] = new float[num_dps];
        for(int j=0; j<num_dps; j++) { // Only copy up to num_dps
            shared_data_float[i][j] = (float)shared_data_double[i][j];
        }
    }
    // Free double data
    for(int i=0; i<=num_vars; i++) free(shared_data_double[i]);
    free(shared_data_double);

    // Prepare CSV Output
    // Sanitize eval name for filename
    std::string safe_eval_name = eval_name;
    for (char &c : safe_eval_name) {
        if (c == ' ' || c == '(' || c == ')') c = '_';
    }
    // Remove consecutive underscores
    std::string::iterator new_end = std::unique(safe_eval_name.begin(), safe_eval_name.end(), 
        [](char a, char b){ return a == '_' && b == '_'; });
    safe_eval_name.erase(new_end, safe_eval_name.end());
    
    // Trim leading/trailing underscores
    if (!safe_eval_name.empty() && safe_eval_name.front() == '_') safe_eval_name.erase(0, 1);
    if (!safe_eval_name.empty() && safe_eval_name.back() == '_') safe_eval_name.pop_back();

    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm* now_tm = std::localtime(&now_c);
    
    char timestamp[32];
    std::strftime(timestamp, sizeof(timestamp), "%m%d-%H%M%S", now_tm);

    std::string csv_filename = "../../data/output/sr_results/sr_results_" + safe_eval_name + 
                               "_gen" + std::to_string(gens) + 
                               "_pop" + std::to_string(pop_size) + 
                               "_dps" + std::to_string(num_dps) + 
                               "_" + std::string(timestamp) + ".csv";
    
    std::ofstream csv_file(csv_filename);
    if (!csv_file.is_open()) {
        std::cerr << "Failed to open CSV file: " << csv_filename << std::endl;
    } else {
        // CSV Header
        csv_file << "Gen,BestMSE,MedianMSE,TotalTimeMs,DetectTimeMs,H2D_D2H_TimeMs,JITTimeMs,KernelTimeMs,NumSubtrees,AvgSubtreeSize,Coverage\n";
    }

    // Initialize Population
    std::vector<Individual> population(pop_size);
    EvolutionParams params;
    params.num_vars = num_vars;
    params.max_tokens = MAX_TOKEN_NUM;

    std::cout << "Initializing population..." << std::endl;
    for(int i=0; i<pop_size; i++) {
        generate_random_expression(params, population[i].tokens, population[i].values);
    }

    // Init Evaluator
    // eval_ctx is already initialized at the beginning of main


    // Main Loop
    std::vector<float> mses(pop_size);
    std::vector<float*> var_ptrs(pop_size);
    for(int i=0; i<pop_size; i++) var_ptrs[i] = (float*)shared_data_float;

    for(int gen=0; gen<gens; gen++) {
        std::cout << "Gen " << gen << ": ";

        // 1. Evaluate
        float*** all_vars_ptr = new float**[pop_size];
        for(int i=0; i<pop_size; i++) all_vars_ptr[i] = shared_data_float;

        InputInfo batch_info;
        batch_info.num_exprs = pop_size;
        batch_info.num_vars = new int[pop_size];
        batch_info.num_dps = new int[pop_size];
        batch_info.num_tokens = new int[pop_size];
        batch_info.tokens = new int*[pop_size];
        batch_info.values = new double*[pop_size];
        
        double** temp_values_double = new double*[pop_size];
        int max_tokens_in_batch = 0;

        for(int i=0; i<pop_size; i++) {
            batch_info.num_vars[i] = num_vars;
            batch_info.num_dps[i] = num_dps;
            batch_info.num_tokens[i] = population[i].tokens.size();
            batch_info.tokens[i] = population[i].tokens.data();
            
            if (batch_info.num_tokens[i] > max_tokens_in_batch) max_tokens_in_batch = batch_info.num_tokens[i];
            
            temp_values_double[i] = new double[batch_info.num_tokens[i]];
            for(int k=0; k<batch_info.num_tokens[i]; k++) {
                temp_values_double[i][k] = (double)population[i].values[k];
            }
            batch_info.values[i] = temp_values_double[i];
        }
        batch_info.max_num_dps = num_dps;
        batch_info.max_tokens = max_tokens_in_batch;
        batch_info.max_num_features = num_vars + 1;

        RunStats stats;

        #ifdef USE_GPU_SUBTREE
        evaluate_gpu_mse_wrapper(batch_info, all_vars_ptr, mses, eval_ctx, (gen==0), false, stats);
        #elif defined(USE_GPU_SIMPLE)
        evaluate_gpu_simple_wrapper(batch_info, all_vars_ptr, mses, eval_ctx, (gen==0), stats);
        #elif defined(USE_GPU_PTX)
        evaluate_gpu_ptx_wrapper(batch_info, all_vars_ptr, mses, eval_ctx, (gen==0), stats);
        #elif defined(USE_CPU)
        evaluate_cpu_mse(batch_info, all_vars_ptr, mses, stats);
        #endif

        // Cleanup Info immediately after eval
        delete[] batch_info.num_vars;
        delete[] batch_info.num_dps;
        delete[] batch_info.num_tokens;
        delete[] batch_info.tokens;
        delete[] batch_info.values;
        for(int i=0; i<pop_size; i++) delete[] temp_values_double[i];
        delete[] temp_values_double;
        delete[] all_vars_ptr;

        // 2. Statistics & Fitness Assignment
        float best_mse = 1e9;
        std::vector<float> sorted_mses;
        sorted_mses.reserve(pop_size);
        int best_ind_idx = -1;

        for(int i=0; i<pop_size; i++) {
            // Assign fitness to individual
            population[i].fitness = mses[i];
            
            if (!std::isnan(mses[i]) && !std::isinf(mses[i])) {
                if(mses[i] < best_mse) {
                    best_mse = mses[i];
                    best_ind_idx = i;
                }
                sorted_mses.push_back(mses[i]);
            }
        }
        
        float median_mse = 0;
        if (!sorted_mses.empty()) {
            std::sort(sorted_mses.begin(), sorted_mses.end());
            median_mse = sorted_mses[sorted_mses.size()/2];
        }

        std::cout << "Best MSE: " << best_mse << " Time: " << stats.total_eval_time_ms << "ms" << std::endl;

        if (best_ind_idx != -1) {
             std::vector<double> d_vals(population[best_ind_idx].values.size());
             for(size_t k=0; k<d_vals.size(); k++) d_vals[k] = (double)population[best_ind_idx].values[k];
             std::string formula = format_formula(population[best_ind_idx].tokens.data(), 
                                                 d_vals.data(), 
                                                 population[best_ind_idx].tokens.size());
             std::cout << "  Best Expr: " << formula << std::endl;
        }

        if (csv_file.is_open()) {
            csv_file << gen << "," 
                     << best_mse << "," 
                     << median_mse << "," 
                     << stats.total_eval_time_ms << ","
                     << stats.drift_detect_time_ms << ","
                     << stats.data_transfer_time_ms << ","
                     << stats.jit_compile_time_ms << "," // New field
                     << stats.gpu_kernel_time_ms << ","
                     << stats.num_subtrees << ","
                     << stats.avg_subtree_size << ","
                     << stats.coverage << "\n";
            if (gen % 10 == 0) csv_file.flush(); 
        }

        // 3. Selection & Breeding
        // Sort population by fitness
        std::sort(population.begin(), population.end(), [](const Individual& a, const Individual& b) {
            if (std::isnan(a.fitness)) return false;
            if (std::isnan(b.fitness)) return true;
            return a.fitness < b.fitness;
        });
        // if (min_mse < 1e-10) {
        //     std::cout << "Solution Found!" << std::endl;
        //     break;
        // }

        std::vector<Individual> next_gen;
        next_gen.reserve(pop_size);
        
        // Elitism (Top 5%)
        int elites = pop_size * 0.05;
        if (elites < 1) elites = 1;
        for(int i=0; i<elites; i++) next_gen.push_back(population[i]);
        
        while((int)next_gen.size() < pop_size) {
            // Tournament
            int t_size = 3;
            int best_idx = rand() % pop_size; // dumb tournament for speed (just random pick vs random pick)
            for(int k=0; k<t_size-1; k++) {
                int cand = rand() % pop_size;
                if (population[cand].fitness < population[best_idx].fitness) best_idx = cand;
            }
            
            Individual child;
            if ((rand() / (float)RAND_MAX) < 0.2f) { // 20% Mutation
                 mutate_expression(params, population[best_idx].tokens, population[best_idx].values, child.tokens, child.values);
            } else { // 80% Crossover
                 // Pick second parent
                 int best_idx2 = rand() % pop_size;
                 for(int k=0; k<t_size-1; k++) {
                    int cand = rand() % pop_size;
                    if (population[cand].fitness < population[best_idx2].fitness) best_idx2 = cand;
                 }
                 crossover_expressions(params, population[best_idx].tokens, population[best_idx].values,
                                       population[best_idx2].tokens, population[best_idx2].values,
                                       child.tokens, child.values);
            }
            next_gen.push_back(child);
        }
        population = next_gen;


    }

    #ifdef USE_GPU_SUBTREE
    destroy_gpu_context(eval_ctx);
    #elif defined(USE_GPU_SIMPLE)
    destroy_gpu_simple_context(eval_ctx);
    #elif defined(USE_GPU_PTX)
    destroy_gpu_ptx_context(eval_ctx);
    #endif
    
    // Free shared data
    for(int i=0; i<=num_vars; i++) delete[] shared_data_float[i];
    delete[] shared_data_float;
    
    return 0;
}
