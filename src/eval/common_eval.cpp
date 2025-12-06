#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include "../utils/utils.h"
#include "evaluator.h"

// Stateless evolution benchmark runner
// Iterates through generations and calls the configured eval_batch function
// Stateless evolution benchmark runner
// Iterates through generations and calls the configured eval_batch function
int run_evolution_benchmark_stateless(int start_gen, int end_gen, const std::string& data_dir)
{
    // Lambda adapter for stateless eval
    auto eval_cb = [](InputInfo& info, double*** vars, double** preds, void* state) {
        eval_batch(info, vars, preds, nullptr);
    };
    
    return evaluate_evolution_benchmark(start_gen, end_gen, data_dir, eval_cb, nullptr);
}
