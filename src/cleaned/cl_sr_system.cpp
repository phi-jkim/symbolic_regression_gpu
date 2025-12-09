#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <fstream>
#include "../utils/utils.h"

// Evaluator signatures
#ifdef USE_GPU_SUBTREE
void* create_gpu_context();
void destroy_gpu_context(void* ctx);
void evaluate_gpu_mse_wrapper(InputInfo& input_info, float*** all_vars, std::vector<float>& mses, void* ctx, bool upload_X, bool clear_cache);
#endif

#ifdef USE_GPU_SIMPLE
void* create_gpu_simple_context();
void destroy_gpu_simple_context(void* ctx);
void evaluate_gpu_simple_wrapper(InputInfo& input_info, float*** all_vars, std::vector<float>& mses, void* ctx, bool upload_X);
#endif

#ifdef USE_CPU
void evaluate_cpu_mse(InputInfo& input_info, float*** all_vars, std::vector<float>& mses);
#endif

struct Individual {
    std::vector<int> tokens;
    std::vector<float> values;
    float fitness; // MSE
};

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <pop_size> <gens> [data_dir]" << std::endl;
        return 1;
    }

    int pop_size = std::atoi(argv[1]);
    int gens = std::atoi(argv[2]);
    std::string data_dir = (argc >= 4) ? argv[3] : "../../data/evolution_test20";
    
    std::string eval_name = "Unknown";
    #ifdef USE_GPU_SUBTREE
    eval_name = "GPU Optimized (Subtree)";
    #elif defined(USE_GPU_SIMPLE)
    eval_name = "GPU Simple";
    #elif defined(USE_CPU)
    eval_name = "CPU";
    #else
    #error "Must define USE_GPU_SUBTREE, USE_GPU_SIMPLE, or USE_CPU"
    #endif

    std::cout << "Starting Minimal SR System (" << eval_name << ")" << std::endl;
    std::cout << "Population: " << pop_size << ", Gens: " << gens << std::endl;
    std::cout << "Data Dir: " << data_dir << std::endl;

    // Load Data
    std::string shared_data_path = data_dir + "/shared_data.txt";
    std::string header_file = data_dir + "/gen_0.txt"; // To get num_vars
    InputInfo info_0 = parse_input_info(header_file);
    if (info_0.num_exprs == 0) {
        std::cerr << "Failed to load header: " << header_file << std::endl;
        return 1;
    }
    int num_vars = info_0.num_vars[0]; // Input vars
    int num_dps = info_0.num_dps[0];
    free_input_info(info_0);

    // Load shared data (double) and convert to float
    double** shared_data_double = load_data_file(shared_data_path, num_vars, num_dps);
    float** shared_data_float = new float*[num_vars + 1];
    for(int i=0; i<=num_vars; i++) {
        shared_data_float[i] = new float[num_dps];
        for(int j=0; j<num_dps; j++) {
            shared_data_float[i][j] = (float)shared_data_double[i][j];
        }
    }
    // Free double data
    for(int i=0; i<=num_vars; i++) free(shared_data_double[i]);
    free(shared_data_double);

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
    void* eval_ctx = nullptr;
    (void)eval_ctx; // Suppress unused warning for CPU build
    #ifdef USE_GPU_SUBTREE
    eval_ctx = create_gpu_context();
    #elif defined(USE_GPU_SIMPLE)
    eval_ctx = create_gpu_simple_context();
    #endif

    // Stats
    std::ofstream csv("evolution_trace.csv");
    csv << "Gen,BestMSE,AvgMSE,P25,P50,P75,P90,TimeMS\n";

    // Main Loop
    for(int gen=0; gen<gens; gen++) {
        // 1. Pack InputInfo
        InputInfo info;
        info.num_exprs = pop_size;
        info.num_vars = new int[pop_size];
        info.num_dps = new int[pop_size];
        info.num_tokens = new int[pop_size];
        info.tokens = new int*[pop_size];
        info.values = new double*[pop_size];
        
        double** temp_values_double = new double*[pop_size];
        
        int max_tokens = 0;
        for(int i=0; i<pop_size; i++) {
            info.num_vars[i] = num_vars;
            info.num_dps[i] = num_dps;
            info.num_tokens[i] = population[i].tokens.size();
            info.tokens[i] = population[i].tokens.data();
            
            if (info.num_tokens[i] > max_tokens) max_tokens = info.num_tokens[i];
            
            temp_values_double[i] = new double[info.num_tokens[i]];
            for(int k=0; k<info.num_tokens[i]; k++) {
                temp_values_double[i][k] = (double)population[i].values[k];
            }
            info.values[i] = temp_values_double[i];
        }
        info.max_num_dps = num_dps; // Important
        info.max_tokens = max_tokens;
        info.max_num_features = num_vars + 1;

        // Prepare all_vars (Input Data)
        float*** all_vars = new float**[pop_size];
        for(int i=0; i<pop_size; i++) all_vars[i] = shared_data_float;

        // 2. Evaluate
        std::vector<float> mses;
        bool upload_X = (gen == 0);
        (void)upload_X; // Suppress unused warning for CPU
        TimePoint t0 = measure_clock();
        
        #ifdef USE_GPU_SUBTREE
        evaluate_gpu_mse_wrapper(info, all_vars, mses, eval_ctx, upload_X, true);
        #elif defined(USE_GPU_SIMPLE)
        evaluate_gpu_simple_wrapper(info, all_vars, mses, eval_ctx, upload_X);
        #elif defined(USE_CPU)
        evaluate_cpu_mse(info, all_vars, mses);
        #endif
        
        double dt = clock_to_ms(t0, measure_clock());

        // 3. Update Fitness & Stats
        std::vector<float> valid_mses;
        valid_mses.reserve(pop_size);

        float min_mse = 1e30f;
        double sum_mse = 0.0;
        int best_ind_idx = -1;
        
        for(int i=0; i<pop_size; i++) {
            population[i].fitness = mses[i];
            if (!std::isnan(mses[i]) && !std::isinf(mses[i])) {
                valid_mses.push_back(mses[i]);
                if (mses[i] < min_mse) {
                    min_mse = mses[i];
                    best_ind_idx = i;
                }
                sum_mse += mses[i];
            }
        }
        
        float avg_mse = valid_mses.empty() ? NAN : (float)(sum_mse / valid_mses.size());
        
        // Percentiles
        float p25 = NAN, p50 = NAN, p75 = NAN, p90 = NAN;
        if (!valid_mses.empty()) {
            std::sort(valid_mses.begin(), valid_mses.end());
            auto get_p = [&](float p) {
                int idx = (int)(p * valid_mses.size());
                if (idx >= (int)valid_mses.size()) idx = (int)valid_mses.size() - 1;
                return valid_mses[idx];
            };
            p25 = get_p(0.25f);
            p50 = get_p(0.50f);
            p75 = get_p(0.75f);
            p90 = get_p(0.90f);
        }
        
        std::cout << "Gen " << gen << ": " << dt << " ms | Best: " << min_mse 
                  << " | P50: " << p50 << " | P90: " << p90 << std::endl;
        
        if (best_ind_idx != -1) {
             std::vector<double> d_vals(population[best_ind_idx].values.size());
             for(size_t k=0; k<d_vals.size(); k++) d_vals[k] = (double)population[best_ind_idx].values[k];
             std::string formula = format_formula(population[best_ind_idx].tokens.data(), 
                                                 d_vals.data(), 
                                                 population[best_ind_idx].tokens.size());
             std::cout << "  Best Expr: " << formula << std::endl;
        }

        csv << gen << "," << min_mse << "," << avg_mse << "," 
            << p25 << "," << p50 << "," << p75 << "," << p90 << "," 
            << dt << "\n";

        // 4. Selection & Breeding
        // Sort
        std::sort(population.begin(), population.end(), [](const Individual& a, const Individual& b) {
            // handle NaN
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

        // Cleanup Info
        delete[] all_vars;
        delete[] info.num_vars;
        delete[] info.num_dps;
        delete[] info.num_tokens;
        delete[] info.tokens;
        // Free temp doubles
        for(int i=0; i<pop_size; i++) delete[] temp_values_double[i];
        delete[] temp_values_double;
        delete[] info.values;
    }

    #ifdef USE_GPU_SUBTREE
    destroy_gpu_context(eval_ctx);
    #elif defined(USE_GPU_SIMPLE)
    destroy_gpu_simple_context(eval_ctx);
    #endif
    
    // Free shared data
    for(int i=0; i<=num_vars; i++) delete[] shared_data_float[i];
    delete[] shared_data_float;
    
    return 0;
}
