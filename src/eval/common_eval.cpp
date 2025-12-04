#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include "../utils/utils.h"
#include "evaluator.h"

// Stateless evolution benchmark runner
// Iterates through generations and calls the configured eval_batch function
int run_evolution_benchmark_stateless(int start_gen, int end_gen, const std::string& data_dir)
{
    std::cout << "Running Stateless Evolution Benchmark: Gen " << start_gen << " to " << end_gen << std::endl;
    
    double total_time_ms = 0.0;
    
    // Pre-load shared data if possible
    double** shared_data_ptr = nullptr;
    int shared_num_vars = 0;
    int shared_num_dps = 0;
    
    // Check Gen 0 for shared data
    {
        std::string filename = data_dir + "/gen_" + std::to_string(start_gen) + ".txt";
        InputInfo info = parse_input_info(filename);
        if (info.num_exprs > 0 && info.has_shared_data) {
            // std::cout << "Pre-loading shared data from " << info.data_filenames[0] << "..." << std::endl;
            shared_data_ptr = load_data_file(info.data_filenames[0], info.num_vars[0], info.num_dps[0]);
            shared_num_vars = info.num_vars[0];
            shared_num_dps = info.num_dps[0];
        }
        free_input_info(info);
    }
    
    for (int gen = start_gen; gen <= end_gen; gen++)
    {
        std::string filename = data_dir + "/gen_" + std::to_string(gen) + ".txt";
        InputInfo input_info = parse_input_info(filename);
        
        if (input_info.num_exprs == 0) {
            std::cerr << "Skipping generation " << gen << " (failed to load)" << std::endl;
            continue;
        }
        
        // Load data (reuse shared if available)
        double ***all_vars = new double **[input_info.num_exprs];
        
        if (input_info.has_shared_data && shared_data_ptr != nullptr) {
            // Point to pre-loaded data
            for (int i = 0; i < input_info.num_exprs; i++) {
                all_vars[i] = shared_data_ptr;
            }
        } else {
            // Fallback to standard loading
            load_all_data_parallel(input_info, all_vars, 8);
        }
        
        // Allocate predictions
        double **all_predictions = new double *[input_info.num_exprs];
        for (int i = 0; i < input_info.num_exprs; i++)
            all_predictions[i] = new double[input_info.num_dps[i]];
            
        // Evaluate (Stateless)
        TimePoint t0 = measure_clock();
        eval_batch(input_info, all_vars, all_predictions, nullptr);
        double dt = clock_to_ms(t0, measure_clock());
        total_time_ms += dt;
        
        std::cout << "Gen " << gen << ": " << dt << " ms" << std::endl;
        
        // Cleanup
        for (int i = 0; i < input_info.num_exprs; i++) delete[] all_predictions[i];
        delete[] all_predictions;
        
        if (input_info.has_shared_data && shared_data_ptr != nullptr) {
            // Do NOT free shared data here, just the array of pointers
            delete[] all_vars;
        } else {
            // Free individually
             for (int i = 0; i < input_info.num_exprs; i++) {
                for(int j=0; j<=input_info.num_vars[i]; j++) free(all_vars[i][j]);
                free(all_vars[i]);
             }
             delete[] all_vars;
        }
        free_input_info(input_info);
    }
    
    // Free shared data at the end
    if (shared_data_ptr != nullptr) {
        // std::cout << "Freeing shared data..." << std::endl;
        for(int i=0; i<=shared_num_vars; i++) free(shared_data_ptr[i]);
        free(shared_data_ptr);
    }
    
    std::cout << "### Total Evolution Time: " << total_time_ms << " ms ###" << std::endl;
    return 0;
}
