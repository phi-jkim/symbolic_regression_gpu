#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include <thread>
#include <atomic>
#include <cstdlib>
#include "../utils/utils.h"
#include "../utils/detect.h"

// ============================================================================
// Standard CPU Evaluator (Helper for evaluating subtrees)
// ============================================================================

static double eval_op(int op, double val1, double val2)
{
    const double DELTA = 1e-9;
    const double MAX_VAL = 1e9;

    switch (op)
    {
    // Binary operators (1-9)
    case 1: return val1 + val2; // ADD
    case 2: return val1 - val2; // SUB
    case 3: return val1 * val2; // MUL
    case 4: return (val2 == 0.0) ? NAN : val1 / val2; // DIV
    case 5: return pow(val1, val2); // POW
    case 6: return (val1 <= val2) ? val1 : val2; // MIN
    case 7: return (val1 >= val2) ? val1 : val2; // MAX
    case 8: // LOOSE_DIV
    {
        double denom = fabs(val2) <= DELTA ? (val2 < 0 ? -DELTA : DELTA) : val2;
        return val1 / denom;
    }
    case 9: // LOOSE_POW
        return (val1 == 0.0 && val2 == 0.0) ? 0.0 : pow(fabs(val1), val2);

    // Unary operators (10-27)
    case 10: return sin(val1); // SIN
    case 11: return cos(val1); // COS
    case 12: return tan(val1); // TAN
    case 13: return sinh(val1); // SINH
    case 14: return cosh(val1); // COSH
    case 15: return tanh(val1); // TANH
    case 16: return exp(val1); // EXP
    case 17: return log(val1); // LOG
    case 18: return 1.0 / val1; // INV
    case 19: return asin(val1); // ASIN
    case 20: return acos(val1); // ACOS
    case 21: return atan(val1); // ATAN
    case 22: // LOOSE_LOG
        return (val1 == 0.0) ? -MAX_VAL : log(fabs(val1));
    case 23: // LOOSE_INV
    {
        double denom = fabs(val1) <= DELTA ? (val1 < 0 ? -DELTA : DELTA) : val1;
        return 1.0 / denom;
    }
    case 24: return fabs(val1); // ABS
    case 25: return -val1; // NEG
    case 26: return sqrt(val1); // SQRT
    case 27: return sqrt(fabs(val1)); // LOOSE_SQRT
    default: return 0;
    }
}

// Thread-local stack to avoid reallocation
static thread_local double stk[MAX_STACK_SIZE];
static thread_local int sp = 0;

static inline void stack_push(double val)
{
    stk[sp] = val;
    sp++;
}

static inline double stack_pop()
{
    sp--;
    return stk[sp];
}

static double eval_tree_cpu_standard(int *tokens, double *values, double *x, int num_tokens)
{
    sp = 0;
    double tmp, val1, val2 = 0.0;
    for (int i = num_tokens - 1; i >= 0; i--)
    {
        int tok = tokens[i];
        if (tok > 0) // operation
        {
            val1 = stack_pop();
            if (tok < 10) // binary operation (1-9)
                val2 = stack_pop();

            tmp = eval_op(tok, val1, val2);
            stack_push(tmp);
        }
        else if (tok == 0) // constant
        {
            stack_push(values[i]);
        }
        else if (tok == -1) // variable
        {
            stack_push(x[(int)values[i]]);
        }
    }
    return stk[0];
}

// ============================================================================
// Optimized CPU Evaluator with Subtree Reuse
// ============================================================================

static double eval_tree_cpu_optimized(
    int *tokens, double *values, int *hints, 
    double *x, double **cached_subtrees, int dp_idx,
    int num_tokens)
{
    sp = 0;
    double tmp, val1, val2 = 0.0;
    
    for (int i = num_tokens - 1; i >= 0; i--)
    {
        int hint = hints[i];
        
        // Case 1: Inside a common subtree (SKIP)
        if (hint == -2) 
        {
            continue;
        }
        
        // Case 2: Root of a common subtree (REUSE)
        if (hint >= 0)
        {
            stack_push(cached_subtrees[hint][dp_idx]);
            continue;
        }
        
        // Case 3: Standard node (EVALUATE)
        int tok = tokens[i];
        if (tok > 0) // operation
        {
            val1 = stack_pop();
            if (tok < 10) // binary operation (1-9)
                val2 = stack_pop();

            tmp = eval_op(tok, val1, val2);
            stack_push(tmp);
        }
        else if (tok == 0) // constant
        {
            stack_push(values[i]);
        }
        else if (tok == -1) // variable
        {
            stack_push(x[(int)values[i]]);
        }
    }
    return stk[0];
}

// Unified batch evaluation function
void eval_cpu_common_subtree_batch(InputInfo &input_info, double ***all_vars, double **all_predictions, EvalMetrics* metrics)
{
    // Determine thread count from environment variable
    int num_threads = 1;
    const char* env_threads = std::getenv("CPU_EVAL_THREADS");
    if (env_threads) {
        num_threads = std::atoi(env_threads);
        if (num_threads < 1) num_threads = 1;
    }
    
    if (num_threads > 1) {
        std::cout << "Using multi-threaded evaluation with " << num_threads << " threads." << std::endl;
    } else {
        // std::cout << "Using single-threaded evaluation." << std::endl;
    }

    // Step 1: Detect common subtrees (Serial)
    int min_freq = 5; 
    if (input_info.num_exprs < 10) min_freq = 2;
    
    SubtreeDetectionResult detect_res = detect_common_subtrees(
        input_info.num_exprs,
        input_info.max_num_features - 1,
        input_info.num_tokens,
        (const int**)input_info.tokens,
        (const double**)input_info.values,
        3, // min_size
        min_freq
    );
    
    if (detect_res.num_subs > 0)
    {
        std::cout << "Detected " << detect_res.num_subs << " common subtrees. Reusing results..." << std::endl;
    }
    
    // Step 2: Evaluate all common subtrees
    double **cached_results = nullptr;
    double total_eval_ms = 0.0;
    
    if (detect_res.num_subs > 0)
    {
        cached_results = new double*[detect_res.num_subs];
        
        int ref_expr_id = detect_res.sub_occ_expr[0][0];
        int total_dps = input_info.num_dps[ref_expr_id];
        int num_vars = input_info.num_vars[ref_expr_id];
        double **vars = all_vars[ref_expr_id];
        
        for (int sub_id = 0; sub_id < detect_res.num_subs; sub_id++)
        {
            cached_results[sub_id] = new double[total_dps];
        }
        
        TimePoint t0 = measure_clock();
        
        if (num_threads > 1)
        {
            // Parallel Cache Filling
            std::vector<std::thread> sub_threads;
            auto sub_worker = [&](int worker_id)
            {
                double *x = new double[MAX_VAR_NUM];
                for (int dp = worker_id; dp < total_dps; dp += num_threads)
                {
                    for (int i = 0; i <= num_vars; i++) x[i] = vars[i][dp];
                    for (int sub_id = 0; sub_id < detect_res.num_subs; sub_id++)
                    {
                        cached_results[sub_id][dp] = eval_tree_cpu_standard(
                            detect_res.sub_tokens[sub_id], 
                            detect_res.sub_values[sub_id], 
                            x, 
                            detect_res.num_sub_tokens[sub_id]);
                    }
                }
                delete[] x;
            };
            for (int i = 0; i < num_threads; i++) sub_threads.emplace_back(sub_worker, i);
            for (auto &t : sub_threads) t.join();
        }
        else
        {
            // Serial Cache Filling
            double *x = new double[MAX_VAR_NUM];
            for (int sub_id = 0; sub_id < detect_res.num_subs; sub_id++)
            {
                int sub_len = detect_res.num_sub_tokens[sub_id];
                int *sub_toks = detect_res.sub_tokens[sub_id];
                double *sub_vals = detect_res.sub_values[sub_id];
                
                for (int dp = 0; dp < total_dps; dp++)
                {
                    for (int i = 0; i <= num_vars; i++) x[i] = vars[i][dp];
                    cached_results[sub_id][dp] = eval_tree_cpu_standard(sub_toks, sub_vals, x, sub_len);
                }
            }
            delete[] x;
        }
        
        total_eval_ms += clock_to_ms(t0, measure_clock());
    }
    
    // Step 3: Evaluate all expressions
    TimePoint t0 = measure_clock();
    
    if (num_threads > 1)
    {
        // Parallel Expression Evaluation
        std::vector<std::thread> expr_threads;
        auto expr_worker = [&](int worker_id)
        {
            double *x = new double[MAX_VAR_NUM];
            for (int expr_id = worker_id; expr_id < input_info.num_exprs; expr_id += num_threads)
            {
                int num_vars = input_info.num_vars[expr_id];
                int num_dps = input_info.num_dps[expr_id];
                int num_tokens = input_info.num_tokens[expr_id];
                int *tokens = input_info.tokens[expr_id];
                double *values = input_info.values[expr_id];
                int *hints = detect_res.expr_sub_hints[expr_id];
                double **vars = all_vars[expr_id];
                double *pred = all_predictions[expr_id];

                for (int dp = 0; dp < num_dps; dp++)
                {
                    for (int i = 0; i <= num_vars; i++) x[i] = vars[i][dp];
                    
                    if (detect_res.num_subs > 0)
                        pred[dp] = eval_tree_cpu_optimized(tokens, values, hints, x, cached_results, dp, num_tokens);
                    else
                        pred[dp] = eval_tree_cpu_standard(tokens, values, x, num_tokens);
                }
            }
            delete[] x;
        };
        for (int i = 0; i < num_threads; i++) expr_threads.emplace_back(expr_worker, i);
        for (auto &t : expr_threads) t.join();
    }
    else
    {
        // Serial Expression Evaluation
        double *x = new double[MAX_VAR_NUM];
        for (int expr_id = 0; expr_id < input_info.num_exprs; expr_id++)
        {
            int num_vars = input_info.num_vars[expr_id];
            int num_dps = input_info.num_dps[expr_id];
            int num_tokens = input_info.num_tokens[expr_id];
            int *tokens = input_info.tokens[expr_id];
            double *values = input_info.values[expr_id];
            int *hints = detect_res.expr_sub_hints[expr_id];
            double **vars = all_vars[expr_id];
            double *pred = all_predictions[expr_id];

            for (int dp = 0; dp < num_dps; dp++)
            {
                for (int i = 0; i <= num_vars; i++) x[i] = vars[i][dp];
                
                if (detect_res.num_subs > 0)
                    pred[dp] = eval_tree_cpu_optimized(tokens, values, hints, x, cached_results, dp, num_tokens);
                else
                    pred[dp] = eval_tree_cpu_standard(tokens, values, x, num_tokens);
            }
        }
        delete[] x;
    }
    
    total_eval_ms += clock_to_ms(t0, measure_clock());
    
    std::cout << "CPU computation time (eval + reuse): " << total_eval_ms << " ms" << std::endl;
    
    // Cleanup
    if (cached_results != nullptr)
    {
        for (int i = 0; i < detect_res.num_subs; i++)
        {
            delete[] cached_results[i];
        }
        delete[] cached_results;
    }
}

// Helper: Fill new subtrees in cache
static void fill_new_subtrees_in_cache(
    SubtreeCache& cache, 
    const std::vector<int>& new_sub_ids, 
    double** vars, 
    int num_vars, 
    int total_dps, 
    int num_threads)
{
    if (new_sub_ids.empty()) return;
    
    if (num_threads > 1) {
         // Parallel Cache Filling for NEW subtrees
        std::vector<std::thread> sub_threads;
        auto sub_worker = [&](int worker_id)
        {
            double *x = new double[MAX_VAR_NUM];
            for (int dp = worker_id; dp < total_dps; dp += num_threads)
            {
                for (int i = 0; i <= num_vars; i++) x[i] = vars[i][dp];
                for (int sub_id : new_sub_ids)
                {
                    cache.results[sub_id][dp] = eval_tree_cpu_standard(
                        cache.sub_tokens[sub_id], 
                        cache.sub_values[sub_id], 
                        x, 
                        cache.sub_sizes[sub_id]);
                }
            }
            delete[] x;
        };
        for (int i = 0; i < num_threads; i++) sub_threads.emplace_back(sub_worker, i);
        for (auto &t : sub_threads) t.join();
    } else {
        // Serial Cache Filling
        double *x = new double[MAX_VAR_NUM];
        for (int sub_id : new_sub_ids)
        {
            int sub_len = cache.sub_sizes[sub_id];
            int *sub_toks = cache.sub_tokens[sub_id];
            double *sub_vals = cache.sub_values[sub_id];
            
            for (int dp = 0; dp < total_dps; dp++)
            {
                for (int i = 0; i <= num_vars; i++) x[i] = vars[i][dp];
                cache.results[sub_id][dp] = eval_tree_cpu_standard(sub_toks, sub_vals, x, sub_len);
            }
        }
        delete[] x;
    }
}

// Helper: Evaluate batch using cache
static void evaluate_batch_with_cache(
    InputInfo &input_info, 
    double ***all_vars, 
    double **all_predictions, 
    SubtreeCache& cache, 
    SubtreeDetectionResult& detect_res, 
    int num_threads)
{
    // We need to pass the cache.results vector as a double**
    // Since std::vector stores pointers contiguously, we can pass &cache.results[0]
    double** cached_results_ptr = cache.results.empty() ? nullptr : &cache.results[0];

    if (num_threads > 1)
    {
        // Parallel Expression Evaluation
        std::vector<std::thread> expr_threads;
        auto expr_worker = [&](int worker_id)
        {
            double *x = new double[MAX_VAR_NUM];
            for (int expr_id = worker_id; expr_id < input_info.num_exprs; expr_id += num_threads)
            {
                int num_vars = input_info.num_vars[expr_id];
                int num_dps = input_info.num_dps[expr_id];
                int num_tokens = input_info.num_tokens[expr_id];
                int *tokens = input_info.tokens[expr_id];
                double *values = input_info.values[expr_id];
                int *hints = detect_res.expr_sub_hints[expr_id];
                double **vars = all_vars[expr_id];
                double *pred = all_predictions[expr_id];

                for (int dp = 0; dp < num_dps; dp++)
                {
                    for (int i = 0; i <= num_vars; i++) x[i] = vars[i][dp];
                    
                    if (cached_results_ptr)
                        pred[dp] = eval_tree_cpu_optimized(tokens, values, hints, x, cached_results_ptr, dp, num_tokens);
                    else
                        pred[dp] = eval_tree_cpu_standard(tokens, values, x, num_tokens);
                }
            }
            delete[] x;
        };
        for (int i = 0; i < num_threads; i++) expr_threads.emplace_back(expr_worker, i);
        for (auto &t : expr_threads) t.join();
    }
    else
    {
        // Serial Expression Evaluation
        double *x = new double[MAX_VAR_NUM];
        for (int expr_id = 0; expr_id < input_info.num_exprs; expr_id++)
        {
            int num_vars = input_info.num_vars[expr_id];
            int num_dps = input_info.num_dps[expr_id];
            int num_tokens = input_info.num_tokens[expr_id];
            int *tokens = input_info.tokens[expr_id];
            double *values = input_info.values[expr_id];
            int *hints = detect_res.expr_sub_hints[expr_id];
            double **vars = all_vars[expr_id];
            double *pred = all_predictions[expr_id];

            for (int dp = 0; dp < num_dps; dp++)
            {
                for (int i = 0; i <= num_vars; i++) x[i] = vars[i][dp];
                
                if (cached_results_ptr)
                    pred[dp] = eval_tree_cpu_optimized(tokens, values, hints, x, cached_results_ptr, dp, num_tokens);
                else
                    pred[dp] = eval_tree_cpu_standard(tokens, values, x, num_tokens);
            }
        }
        delete[] x;
    }
}

// Stateful batch evaluation function
void eval_cpu_stateful_batch(
    InputInfo &input_info, 
    double ***all_vars, 
    double **all_predictions, 
    SubtreeCache& cache,
    EvalMetrics* metrics)
{
    // Determine thread count
    int num_threads = 1;
    const char* env_threads = std::getenv("CPU_EVAL_THREADS");
    if (env_threads) {
        num_threads = std::atoi(env_threads);
        if (num_threads < 1) num_threads = 1;
    }

    // Step 1: Detect and update cache (Serial)
    int min_freq = 5; 
    if (input_info.num_exprs < 10) min_freq = 2;
    
    SubtreeDetectionResult detect_res = detect_and_update_cache(
        cache,
        input_info.num_exprs,
        input_info.max_num_features - 1,
        input_info.num_tokens,
        (const int**)input_info.tokens,
        (const double**)input_info.values,
        3, // min_size
        min_freq
    );
    
    // Step 2: Evaluate NEW subtrees in cache
    int total_dps = input_info.num_dps[0]; // Assume shared data
    int num_vars = input_info.num_vars[0];
    double **vars = all_vars[0];
    
    std::vector<int> new_sub_ids;
    for (int i = 0; i < (int)cache.results.size(); i++) {
        if (cache.results[i] == nullptr) {
            cache.results[i] = new double[total_dps];
            new_sub_ids.push_back(i);
        }
    }
    
    fill_new_subtrees_in_cache(cache, new_sub_ids, vars, num_vars, total_dps, num_threads);
    
    // Step 3: Evaluate all expressions using cache
    evaluate_batch_with_cache(input_info, all_vars, all_predictions, cache, detect_res, num_threads);
    
    // Cleanup hints (but NOT the cache)
    free_subtree_detection_result(detect_res);
}

// Evolution benchmark runner
int run_evolution_benchmark(int start_gen, int end_gen, const std::string& data_dir)
{
    std::cout << "Running Evolution Benchmark: Gen " << start_gen << " to " << end_gen << std::endl;
    
    SubtreeCache cache;
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
            std::cout << "Pre-loading shared data from " << info.data_filenames[0] << "..." << std::endl;
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
            
        // Evaluate statefully
        TimePoint t0 = measure_clock();
        eval_cpu_stateful_batch(input_info, all_vars, all_predictions, cache, nullptr);
        double dt = clock_to_ms(t0, measure_clock());
        total_time_ms += dt;
        
        std::cout << "Gen " << gen << ": " << dt << " ms (Cache size: " << cache.results.size() << ")" << std::endl;
        
        // Cleanup
        for (int i = 0; i < input_info.num_exprs; i++) delete[] all_predictions[i];
        delete[] all_predictions;
        
        if (input_info.has_shared_data && shared_data_ptr != nullptr) {
            // Do NOT free shared data here, just the array of pointers
            delete[] all_vars; // Free the array of pointers itself
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
        std::cout << "Freeing shared data..." << std::endl;
        for(int i=0; i<=shared_num_vars; i++) free(shared_data_ptr[i]);
        free(shared_data_ptr);
    }
    
    std::cout << "Total Evolution Time: " << total_time_ms << " ms" << std::endl;
    return 0;
}
