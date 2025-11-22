/*
 * Token Mapping Reference
 * =======================
 *
 * Special Tokens:
 *   0  = CONST (constant value)
 *   -1 = VAR (variable reference)
 *
 * Binary Operators (1-9): op(val1, val2)
 *   1 = ADD         (val1 + val2)
 *   2 = SUB         (val1 - val2)
 *   3 = MUL         (val1 * val2)
 *   4 = DIV         (val1 / val2, returns NAN if val2 == 0)
 *   5 = POW         (val1 ** val2)
 *   6 = MIN         (min(val1, val2))
 *   7 = MAX         (max(val1, val2))
 *   8 = LOOSE_DIV   (safe division with epsilon protection)
 *   9 = LOOSE_POW   (safe power, handles 0^0 case)
 *
 * Unary Operators (10-27): op(val1)
 *   10 = SIN        (sine)
 *   11 = COS        (cosine)
 *   12 = TAN        (tangent)
 *   13 = SINH       (hyperbolic sine)
 *   14 = COSH       (hyperbolic cosine)
 *   15 = TANH       (hyperbolic tangent)
 *   16 = EXP        (e^val1)
 *   17 = LOG        (natural logarithm)
 *   18 = INV        (1/val1)
 *   19 = ASIN       (arcsine)
 *   20 = ACOS       (arccosine)
 *   21 = ATAN       (arctangent)
 *   22 = LOOSE_LOG  (safe log: log(abs(val1)), returns -MAX_VAL if val1 == 0)
 *   23 = LOOSE_INV  (safe inverse with epsilon protection)
 *   24 = ABS        (absolute value)
 *   25 = NEG        (negation: -val1)
 *   26 = SQRT       (square root)
 *   27 = LOOSE_SQRT (safe sqrt: sqrt(abs(val1)))
 *
 * Notation: Prefix notation (operator before operands)
 * Evaluation: Stack-based, processes tokens right to left
 */

#include "utils.h"
#include "opcodes.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <sys/stat.h>
#include <sys/types.h>
#include <thread>
#include <atomic>

std::string format_formula(int *tokens, double *values, int num_tokens)
{
    // Operator name mappings
    static const char *binary_ops[] = {// 1 ~ 9
                                       nullptr, "+", "-", "*", "/", "**", "min", "max", "loose_div", "loose_pow"};
    static const char *unary_ops[] = {
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        "sin", "cos", "tan", "sinh", "cosh", "tanh", "exp", "log", "inv",
        "asin", "acos", "atan", "loose_log", "loose_inv", "abs", "-", "sqrt", "loose_sqrt"};

    std::vector<std::string> stack;

    for (int i = num_tokens - 1; i >= 0; i--)
    {
        int token = tokens[i];

        if (token == 0) // Constant
        {
            stack.push_back(std::to_string(values[i]));
        }
        else if (token == -1) // Variable
        {
            stack.push_back("x" + std::to_string(static_cast<int>(values[i])));
        }
        else if (token >= 1 && token <= 9) // Binary operators
        {
            if (stack.size() < 2)
                return "INVALID";
            std::string a = stack.back();
            stack.pop_back();
            std::string b = stack.back();
            stack.pop_back();
            stack.push_back("(" + a + " " + binary_ops[token] + " " + b + ")");
        }
        else if (token >= 10 && token <= 27) // Unary operators
        {
            if (stack.empty())
                return "INVALID";
            std::string a = stack.back();
            stack.pop_back();
            if (token == 25) // NEG is prefix
                stack.push_back("(-" + a + ")");
            else
                stack.push_back(std::string(unary_ops[token]) + "(" + a + ")");
        }
        else
        {
            return "UNKNOWN_TOKEN";
        }
    }

    if (stack.size() != 1)
    {
        std::cerr << "ERROR: Invalid stack state: ";
        for (const auto &item : stack)
        {
            std::cerr << item << ", ";
        }
        std::cerr << std::endl;
        return "INVALID";
    }
    return stack[0];
}

InputInfo parse_input_info(const std::string &input_file)
{
    InputInfo info;
    std::ifstream file(input_file);

    if (!file.is_open())
    {
        std::cerr << "Error: Could not open input file: " << input_file << std::endl;
        info.num_exprs = 0;
        return info;
    }

    // Read num_exprs
    file >> info.num_exprs;

    if (info.num_exprs <= 0)
    {
        std::cerr << "Error: Invalid num_exprs: " << info.num_exprs << std::endl;
        info.num_exprs = 0;
        return info;
    }

    // Allocate arrays for each expression's metadata
    info.num_vars = new int[info.num_exprs];
    info.num_dps = new int[info.num_exprs];
    info.num_tokens = new int[info.num_exprs];
    info.tokens = new int *[info.num_exprs];
    info.values = new double *[info.num_exprs];
    info.data_filenames = new std::string[info.num_exprs];
    // Initialize optional packed buffers as null (allocated later on demand)
    info.tokens_packed = nullptr;
    info.values_packed_f32 = nullptr;
    info.X_packed_f32 = nullptr;
    // Initialize maxima
    info.max_tokens = 0;
    info.max_num_dps = 0;
    info.max_num_features = 0;

    // Read each expression
    for (int expr_id = 0; expr_id < info.num_exprs; expr_id++)
    {
        // Read metadata for this expression
        file >> info.num_vars[expr_id];
        file >> info.num_dps[expr_id];
        file >> info.num_tokens[expr_id];

        // Update maxima as we parse
        if (info.num_tokens[expr_id] > info.max_tokens) info.max_tokens = info.num_tokens[expr_id];
        if (info.num_dps[expr_id] > info.max_num_dps) info.max_num_dps = info.num_dps[expr_id];
        int nf = info.num_vars[expr_id] + 1;
        if (nf > info.max_num_features) info.max_num_features = nf;

        // std::cout << "  Expression " << (expr_id + 1) << ": " << info.num_vars[expr_id]
        //           << " vars, " << info.num_dps[expr_id] << " dps, "
        //           << info.num_tokens[expr_id] << " tokens" << std::endl;

        // Allocate and read tokens
        info.tokens[expr_id] = new int[info.num_tokens[expr_id]];
        for (int i = 0; i < info.num_tokens[expr_id]; i++)
        {
            file >> info.tokens[expr_id][i];
        }

        // Allocate and read values
        info.values[expr_id] = new double[info.num_tokens[expr_id]];
        for (int i = 0; i < info.num_tokens[expr_id]; i++)
        {
            file >> info.values[expr_id][i];
        }

        // Read data filename
        file.ignore(); // Skip newline
        std::getline(file, info.data_filenames[expr_id]);
    }

    file.close();

    // Detect if all expressions share the same data file
    info.has_shared_data = false;
    if (info.num_exprs > 1)
    {
        bool all_same = true;
        std::string first_file = info.data_filenames[0];
        int first_num_vars = info.num_vars[0];
        int first_num_dps = info.num_dps[0];
        
        for (int i = 1; i < info.num_exprs; i++)
        {
            if (info.data_filenames[i] != first_file ||
                info.num_vars[i] != first_num_vars ||
                info.num_dps[i] != first_num_dps)
            {
                all_same = false;
                break;
            }
        }
        
        info.has_shared_data = all_same;
        
        if (info.has_shared_data)
        {
            std::cout << "Detected shared data mode: all expressions use " << first_file << std::endl;
        }
    }

    std::cout << format_formula(info.tokens[0], info.values[0], info.num_tokens[0]) << std::endl;

    std::cout << "Loaded digest file: " << input_file << std::endl;
    std::cout << "Number of expressions: " << info.num_exprs << std::endl;

    return info;
}

double **load_data_file(const std::string &filename, int num_vars, int num_dps)
{
    // Allocate memory for variables (including output variable)
    double **vars = (double **)malloc((num_vars + 1) * sizeof(double *));
    for (int i = 0; i <= num_vars; i++)
    {
        vars[i] = (double *)malloc(num_dps * sizeof(double));
    }

    // Open file and get size
    FILE *file = fopen(filename.c_str(), "rb");
    if (!file)
    {
        std::cerr << "Error: Could not open data file: " << filename << std::endl;
        return vars;
    }

    // Get file size
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    // Allocate buffer for entire file
    char *buffer = (char *)malloc(file_size + 1);
    if (!buffer)
    {
        std::cerr << "Error: Could not allocate buffer for file: " << filename << std::endl;
        fclose(file);
        return vars;
    }

    // Read entire file into buffer
    size_t bytes_read = fread(buffer, 1, file_size, file);
    fclose(file);

    if (bytes_read != (size_t)file_size)
    {
        std::cerr << "Error: Could not read entire file: " << filename << std::endl;
        free(buffer);
        return vars;
    }

    buffer[file_size] = '\0'; // Null-terminate for safety

    // Fast parsing with strtod
    char *ptr = buffer;
    char *end = buffer + file_size;

    for (int j = 0; j < num_dps; j++)
    {
        for (int i = 0; i <= num_vars; i++)
        {
            vars[i][j] = strtod(ptr, &ptr);
            // Skip whitespace (space, newline, tab, carriage return)
            while (ptr < end && (*ptr == ' ' || *ptr == '\n' || *ptr == '\t' || *ptr == '\r'))
            {
                ptr++;
            }
        }
    }

    // Free the buffer
    free(buffer);

    return vars;
}

void load_all_data_parallel(InputInfo &input_info, double ***all_vars, int num_threads)
{
    // Optimization: If all expressions share the same data file, load once
    if (input_info.has_shared_data)
    {
        std::cout << "Loading shared data file once..." << std::endl;
        
        // Load the shared data file once
        int num_vars = input_info.num_vars[0];
        int num_dps = input_info.num_dps[0];
        std::string data_filename = input_info.data_filenames[0];
        
        double **shared_vars = load_data_file(data_filename, num_vars, num_dps);
        
        // All expressions point to the same data
        for (int expr_id = 0; expr_id < input_info.num_exprs; expr_id++)
        {
            all_vars[expr_id] = shared_vars;
        }
        
        return;
    }
    
    // Original parallel loading for non-shared data
    std::atomic<int> next_expr(0);
    std::vector<std::thread> threads;

    // Worker function - each thread processes expressions until none remain
    auto worker = [&]()
    {
        while (true)
        {
            int expr_id = next_expr.fetch_add(1);
            if (expr_id >= input_info.num_exprs)
                break;

            int num_vars = input_info.num_vars[expr_id];
            int num_dps = input_info.num_dps[expr_id];
            std::string data_filename = input_info.data_filenames[expr_id];

            all_vars[expr_id] = load_data_file(data_filename, num_vars, num_dps);
        }
    };

    // Launch worker threads
    for (int i = 0; i < num_threads; i++)
    {
        threads.emplace_back(worker);
    }

    // Wait for all threads to complete
    for (auto &t : threads)
    {
        t.join();
    }
}

ResultInfo make_result_info(double *pred, double **vars, int num_vars, int num_dps,
                            double init_time_ms, double eval_time_ms, double output_time_ms)
{
    ResultInfo result;
    result.pred = pred;
    result.init_time_ms = init_time_ms;
    result.eval_time_ms = eval_time_ms;
    result.output_time_ms = output_time_ms;

    // Calculate statistics
    std::vector<double> diffs(num_dps);
    double sum_squared_error = 0.0;

    for (int i = 0; i < num_dps; i++)
    {
        double diff = pred[i] - vars[num_vars][i];
        diffs[i] = std::abs(diff);
        sum_squared_error += diff * diff;
    }

    result.mse = sum_squared_error / num_dps;

    // Calculate median
    std::vector<double> sorted_diffs = diffs;
    std::sort(sorted_diffs.begin(), sorted_diffs.end());
    result.median = (num_dps % 2 == 0)
                        ? (sorted_diffs[num_dps / 2 - 1] + sorted_diffs[num_dps / 2]) / 2.0
                        : sorted_diffs[num_dps / 2];

    // Calculate standard deviation
    double mean_diff = 0.0;
    for (int i = 0; i < num_dps; i++)
    {
        mean_diff += diffs[i];
    }
    mean_diff /= num_dps;

    double variance = 0.0;
    for (int i = 0; i < num_dps; i++)
    {
        double dev = diffs[i] - mean_diff;
        variance += dev * dev;
    }
    result.stdev = std::sqrt(variance / num_dps);

    return result;
}

void save_results(const std::string &digest_file, const ExpressionInfo &expr_info, const ResultInfo &result_info, double **vars)
{
    int num_dps = expr_info.num_dps;
    int num_vars = expr_info.num_vars;
    double *pred = result_info.pred;

    // Create output directory if it doesn't exist
    std::string output_dir = "data/output/ai_feyn/raw";
    mkdir("data", 0755);
    mkdir("data/output", 0755);
    mkdir("data/output/ai_feyn", 0755);
    mkdir(output_dir.c_str(), 0755);

    // Extract filename from digest_file path
    size_t last_slash = digest_file.find_last_of('/');
    std::string filename = (last_slash != std::string::npos) ? digest_file.substr(last_slash + 1) : digest_file;
    size_t dot_pos = filename.find_last_of('.');
    std::string base_name = (dot_pos != std::string::npos) ? filename.substr(0, dot_pos) : filename;

    // Write the output info to output file
    std::string output_file = output_dir + "/" + base_name + "_output.txt";
    std::ofstream result_file(output_file, std::ios::out);

    // Write statistics at the top
    result_file << "MSE =       \t" << result_info.mse << "\n";
    result_file << "Median =    \t" << result_info.median << "\n";
    result_file << "StdDev =    \t" << result_info.stdev << "\n";
    result_file << "InitTime =  \t" << result_info.init_time_ms << "\n";
    result_file << "EvalTime =  \t" << result_info.eval_time_ms << "\n";
    result_file << "OutputTime =\t" << result_info.output_time_ms << "\n";
    result_file << "##\n";

    result_file << num_dps << "\n";

    for (int i = 0; i < num_dps; i++)
    {
        double diff = std::abs(pred[i] - vars[num_vars][i]);
        if (diff > 1e-10)
            std::cerr << "Warning!\n";
        result_file << diff << " ; " << pred[i] << " " << vars[num_vars][i] << "\n";
    }
    result_file.close();

    std::cout << "\nResults written to: " << output_file << std::endl;
    std::cout << "MSE: " << result_info.mse << ", Median: " << result_info.median
              << ", StdDev: " << result_info.stdev << std::endl;
    std::cout << "Timing - Init: " << result_info.init_time_ms << "ms, Eval: " << result_info.eval_time_ms
              << "ms, Output: " << result_info.output_time_ms << "ms" << std::endl;
}

void free_result_info(ResultInfo &info)
{
    if (info.pred != nullptr)
    {
        free(info.pred);
        info.pred = nullptr;
    }
}

void save_aggregated_results(const std::string &digest_file, const AggregatedResults &results)
{
    // Create output directory if it doesn't exist
    std::string output_dir = "data/output/ai_feyn/raw";
    mkdir("data", 0755);
    mkdir("data/output", 0755);
    mkdir("data/output/ai_feyn", 0755);
    mkdir(output_dir.c_str(), 0755);

    // Extract filename from digest_file path
    size_t last_slash = digest_file.find_last_of('/');
    std::string filename = (last_slash != std::string::npos) ? digest_file.substr(last_slash + 1) : digest_file;
    size_t dot_pos = filename.find_last_of('.');
    std::string base_name = (dot_pos != std::string::npos) ? filename.substr(0, dot_pos) : filename;

    // Write aggregated results
    std::string output_file = output_dir + "/" + base_name + "_output.txt";
    std::ofstream result_file(output_file, std::ios::out);

    // Write summary statistics
    result_file << "NumExpressions " << results.num_exprs << "\n";
    result_file << "TotalTime " << results.total_time_ms << "\n";
    result_file << "TotalInitTime " << results.total_init_ms << "\n";
    result_file << "TotalEvalTime " << results.total_eval_ms << "\n";
    result_file << "AvgMSE " << results.avg_mse << "\n";
    result_file << "AvgMedian " << results.avg_median << "\n";
    result_file << "AvgStdDev " << results.avg_stdev << "\n";
    result_file << "##\n";

    // Write per-expression metrics
    for (int i = 0; i < results.num_exprs; i++)
    {
        result_file << "Expr " << (i + 1)
                    << " MSE " << results.mse_values[i]
                    << " Median " << results.median_values[i]
                    << " StdDev " << results.stdev_values[i]
                    << "\n";
    }

    result_file.close();

    // Print summary to console
    std::cout << "\n"
              << std::string(60, '=') << std::endl;
    std::cout << "Aggregated Results (" << results.num_exprs << " expressions)" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    std::cout << "Timing Breakdown:" << std::endl;
    std::cout << "  Total Time:      " << results.total_time_ms << " ms" << std::endl;
    std::cout << "  Total Init Time: " << results.total_init_ms << " ms" << std::endl;
    std::cout << "  Total Eval Time: " << results.total_eval_ms << " ms" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    std::cout << "Accuracy Metrics (Average):" << std::endl;
    std::cout << "  MSE:    " << results.avg_mse << std::endl;
    std::cout << "  Median: " << results.avg_median << std::endl;
    std::cout << "  StdDev: " << results.avg_stdev << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    std::cout << "Results written to: " << output_file << std::endl;
    std::cout << std::string(60, '=') << std::endl;
}

void free_aggregated_results(AggregatedResults &results)
{
    if (results.num_exprs > 0)
    {
        delete[] results.mse_values;
        delete[] results.median_values;
        delete[] results.stdev_values;
        results.num_exprs = 0;
    }
}

void evaluate_and_save_results(const std::string &digest_file, InputInfo &input_info,
                               MultiEvalFunc multi_eval_func,
                               TimePoint start_time)
{
    // ============================================
    // PHASE 1: Load all data (Total Init Time)
    // ============================================
    TimePoint init_start = measure_clock();

    // Allocate arrays for all expression data
    double ***all_vars = new double **[input_info.num_exprs];

    // Load all expression data files in parallel (8 worker threads)
    load_all_data_parallel(input_info, all_vars, 8);

    double total_init_time = clock_to_ms(init_start, measure_clock());

    // ============================================
    // PHASE 2: Allocate prediction arrays
    // ============================================
    double **all_predictions = new double *[input_info.num_exprs];
    for (int expr_id = 0; expr_id < input_info.num_exprs; expr_id++)
    {
        int num_dps = input_info.num_dps[expr_id];
        all_predictions[expr_id] = new double[num_dps];
    }

    // ============================================
    // PHASE 3: Compute maxima and optional packing for GPU evaluator
    // ============================================
    // compute maxima across expressions for downstream alloc sizing

    // Allocate arrays of pointers for packed buffers
    input_info.tokens_packed = new int*[input_info.num_exprs];
    input_info.values_packed_f32 = new float*[input_info.num_exprs];
    input_info.X_packed_f32 = new float*[input_info.num_exprs];
    for (int expr_id = 0; expr_id < input_info.num_exprs; expr_id++) {
        input_info.tokens_packed[expr_id] = nullptr;
        input_info.values_packed_f32[expr_id] = nullptr;
        input_info.X_packed_f32[expr_id] = nullptr;
    }

    // TODO: could do this step in parallel or omit and modify how expression data is stored 
    // Build contiguous buffers
    for (int expr_id = 0; expr_id < input_info.num_exprs; expr_id++) {
        const int num_tokens = input_info.num_tokens[expr_id];
        const int num_vars = input_info.num_vars[expr_id];
        const int num_dps = input_info.num_dps[expr_id];
        const int num_features = num_vars + 1;

        // tokens
        int* tok_buf = new int[num_tokens];
        for (int i = 0; i < num_tokens; ++i) tok_buf[i] = input_info.tokens[expr_id][i];
        input_info.tokens_packed[expr_id] = tok_buf;

        // values (double -> float)
        float* val_buf = new float[num_tokens];
        for (int i = 0; i < num_tokens; ++i) val_buf[i] = static_cast<float>(input_info.values[expr_id][i]);
        input_info.values_packed_f32[expr_id] = val_buf;

        // X row-major [dp, feature], convert double -> float
        float* x_buf = new float[static_cast<size_t>(num_dps) * static_cast<size_t>(num_features)];
        for (int dp = 0; dp < num_dps; ++dp) {
            for (int f = 0; f < num_features; ++f) {
                x_buf[static_cast<size_t>(dp) * static_cast<size_t>(num_features) + f] =
                    static_cast<float>(all_vars[expr_id][f][dp]);
            }
        }
        input_info.X_packed_f32[expr_id] = x_buf;
    }

    // ============================================
    // PHASE 4: Evaluate (Total Eval Time)
    // BLACK BOX: CPU or GPU implementation
    // ============================================
    TimePoint eval_start = measure_clock();

    multi_eval_func(input_info, all_vars, all_predictions);

    double total_eval_time = clock_to_ms(eval_start, measure_clock());

    // ============================================
    // PHASE 4: Calculate statistics per expression
    // ============================================
    AggregatedResults agg_results;
    agg_results.num_exprs = input_info.num_exprs;
    agg_results.mse_values = new double[agg_results.num_exprs];
    agg_results.median_values = new double[agg_results.num_exprs];
    agg_results.stdev_values = new double[agg_results.num_exprs];

    for (int expr_id = 0; expr_id < input_info.num_exprs; expr_id++)
    {
        int num_vars = input_info.num_vars[expr_id];
        int num_dps = input_info.num_dps[expr_id];

        ResultInfo result_info = make_result_info(
            all_predictions[expr_id],
            all_vars[expr_id],
            num_vars,
            num_dps,
            0.0, // No per-expr init time
            0.0, // No per-expr eval time
            0.0);

        agg_results.mse_values[expr_id] = result_info.mse;
        agg_results.median_values[expr_id] = result_info.median;
        agg_results.stdev_values[expr_id] = result_info.stdev;

        // Don't free pred pointer - it's owned by all_predictions array
        result_info.pred = nullptr;
        free_result_info(result_info);
    }

    // ============================================
    // PHASE 5: Calculate totals and averages
    // ============================================
    agg_results.total_time_ms = clock_to_ms(start_time, measure_clock());
    agg_results.total_init_ms = total_init_time;
    agg_results.total_eval_ms = total_eval_time;

    agg_results.avg_mse = 0.0;
    agg_results.avg_median = 0.0;
    agg_results.avg_stdev = 0.0;
    int valid_count = 0;

    // Calculate averages (skip NaN/inf)
    for (int i = 0; i < agg_results.num_exprs; i++)
    {
        if (std::isfinite(agg_results.mse_values[i]) &&
            std::isfinite(agg_results.median_values[i]) &&
            std::isfinite(agg_results.stdev_values[i]))
        {
            agg_results.avg_mse += agg_results.mse_values[i];
            agg_results.avg_median += agg_results.median_values[i];
            agg_results.avg_stdev += agg_results.stdev_values[i];
            valid_count++;
        }
    }

    if (valid_count > 0)
    {
        agg_results.avg_mse /= valid_count;
        agg_results.avg_median /= valid_count;
        agg_results.avg_stdev /= valid_count;
    }

    // ============================================
    // PHASE 6: Save results and cleanup
    // ============================================
    save_aggregated_results(digest_file, agg_results);

    // Cleanup
    if (input_info.has_shared_data)
    {
        // Shared data: only free once (all pointers are the same)
        free_data(all_vars[0], input_info.num_vars[0]);
    }
    else
    {
        // Non-shared data: free each separately
        for (int expr_id = 0; expr_id < input_info.num_exprs; expr_id++)
        {
            free_data(all_vars[expr_id], input_info.num_vars[expr_id]);
        }
    }
    
    // Free prediction arrays (always separate)
    for (int expr_id = 0; expr_id < input_info.num_exprs; expr_id++)
    {
        delete[] all_predictions[expr_id];
    }
    
    delete[] all_vars;
    delete[] all_predictions;
    free_aggregated_results(agg_results);
}

void free_input_info(InputInfo &info)
{
    if (info.num_exprs > 0)
    {
        // Free each expression's data
        for (int i = 0; i < info.num_exprs; i++)
        {
            if (info.tokens[i] != nullptr)
            {
                delete[] info.tokens[i];
            }
            if (info.values[i] != nullptr)
            {
                delete[] info.values[i];
            }
            if (info.tokens_packed && info.tokens_packed[i] != nullptr) {
                delete[] info.tokens_packed[i];
            }
            if (info.values_packed_f32 && info.values_packed_f32[i] != nullptr) {
                delete[] info.values_packed_f32[i];
            }
            if (info.X_packed_f32 && info.X_packed_f32[i] != nullptr) {
                delete[] info.X_packed_f32[i];
            }
        }

        // Free arrays
        delete[] info.num_vars;
        delete[] info.num_dps;
        delete[] info.num_tokens;
        delete[] info.tokens;
        delete[] info.values;
        delete[] info.data_filenames;
        if (info.tokens_packed) delete[] info.tokens_packed;
        if (info.values_packed_f32) delete[] info.values_packed_f32;
        if (info.X_packed_f32) delete[] info.X_packed_f32;

        info.num_exprs = 0;
    }
}

void free_data(double **vars, int num_vars)
{
    if (vars != nullptr)
    {
        for (int i = 0; i <= num_vars; i++)
        {
            if (vars[i] != nullptr)
            {
                free(vars[i]);
            }
        }
        free(vars);
    }
}

// ----------------------------------------------------------------------------
// Evolution helpers
// ----------------------------------------------------------------------------
namespace
{
    using TokenVec = std::vector<int>;
    using ValueVec = std::vector<float>;

    constexpr int kUnaryOps[] = {
        OP_SIN, OP_COS, OP_TAN, OP_SINH, OP_COSH, OP_TANH,
        OP_EXP, OP_LOG, OP_INV, OP_ASIN, OP_ACOS, OP_ATAN,
        OP_LOOSE_LOG, OP_LOOSE_INV, OP_ABS, OP_NEG, OP_SQRT, OP_LOOSE_SQRT
    };

    constexpr int kBinaryOps[] = {
        OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_POW,
        OP_MIN, OP_MAX, OP_LOOSE_DIV, OP_LOOSE_POW
    };

    std::mt19937& global_rng()
    {
        static thread_local std::mt19937 rng(std::random_device{}());
        return rng;
    }

    float rand_uniform(float lo, float hi)
    {
        std::uniform_real_distribution<float> dist(lo, hi);
        return dist(global_rng());
    }

    int rand_int(int inclusive_lo, int inclusive_hi)
    {
        if (inclusive_hi <= inclusive_lo)
            return inclusive_lo;
        std::uniform_int_distribution<int> dist(inclusive_lo, inclusive_hi);
        return dist(global_rng());
    }

    template <typename T, size_t N>
    T random_from(const T (&arr)[N])
    {
        return arr[rand_int(0, static_cast<int>(N) - 1)];
    }

    struct BuildContext
    {
        const EvolutionParams& params;
        int tokens_left;
        TokenVec* tokens;
        ValueVec* values;
        int num_vars;
    };

    void emit_leaf(BuildContext& ctx)
    {
        if (ctx.tokens_left <= 0)
            return;

        bool choose_const = rand_uniform(0.0f, 1.0f) < ctx.params.prob_const_leaf;
        if (ctx.num_vars <= 0)
            choose_const = true;

        if (choose_const)
        {
            ctx.tokens->push_back(TOK_CONST);
            ctx.values->push_back(rand_uniform(ctx.params.const_min, ctx.params.const_max));
        }
        else
        {
            int var_idx = rand_int(0, ctx.num_vars - 1);
            ctx.tokens->push_back(TOK_VAR);
            ctx.values->push_back(static_cast<float>(var_idx));
        }
        ctx.tokens_left -= 1;
    }

    void build_node(BuildContext& ctx, int depth)
    {
        if (ctx.tokens_left <= 0)
            return;

        bool max_depth_reached = depth >= ctx.params.max_depth;
        bool must_leaf = max_depth_reached || ctx.tokens_left == 1;

        if (!must_leaf)
        {
            bool force_leaf = rand_uniform(0.0f, 1.0f) < ctx.params.prob_leaf;
            if (force_leaf)
                must_leaf = true;
        }

        if (must_leaf)
        {
            emit_leaf(ctx);
            return;
        }

        bool allow_unary = ctx.tokens_left >= 2;
        bool allow_binary = ctx.tokens_left >= 3;

        int arity = 0;
        int op = TOK_CONST;

        if (!allow_unary && !allow_binary)
        {
            emit_leaf(ctx);
            return;
        }

        float unary_thresh = ctx.params.prob_unary;
        if (!allow_unary)
            unary_thresh = 0.0f;
        if (!allow_binary)
            unary_thresh = 1.0f;

        bool pick_unary = rand_uniform(0.0f, 1.0f) < unary_thresh;
        if (pick_unary)
        {
            op = random_from(kUnaryOps);
            arity = 1;
        }
        else
        {
            op = random_from(kBinaryOps);
            arity = 2;
        }

        ctx.tokens->push_back(op);
        ctx.values->push_back(0.0f);
        ctx.tokens_left -= 1;

        for (int i = 0; i < arity; ++i)
        {
            if (ctx.tokens_left <= 0)
            {
                // fallback: ensure missing children become leaves if budget exhausted
                ctx.tokens->push_back(TOK_VAR);
                ctx.values->push_back(0.0f);
                continue;
            }
            build_node(ctx, depth + 1);
        }
    }

    void generate_with_limit(const EvolutionParams& params, int max_tokens,
                              TokenVec& tokens_out, ValueVec& values_out)
    {
        tokens_out.clear();
        values_out.clear();
        if (max_tokens <= 0)
            return;

        tokens_out.reserve(max_tokens);
        values_out.reserve(max_tokens);

        EvolutionParams tuned = params;
        tuned.max_tokens = std::min(params.max_tokens, max_tokens);

        BuildContext ctx{tuned, tuned.max_tokens, &tokens_out, &values_out, std::max(1, tuned.num_vars)};
        build_node(ctx, 0);

        if (tokens_out.empty())
        {
            tokens_out.push_back(TOK_CONST);
            values_out.push_back(0.0f);
        }
    }

    int subtree_size(const TokenVec& tokens, int start)
    {
        if (start < 0 || start >= static_cast<int>(tokens.size()))
            return 0;
        int token = tokens[start];
        int arity = (token == TOK_CONST || token == TOK_VAR) ? 0 : op_arity(token);
        int consumed = 1;
        int cursor = start + 1;
        for (int i = 0; i < arity; ++i)
        {
            if (cursor >= static_cast<int>(tokens.size()))
                break;
            int child = subtree_size(tokens, cursor);
            if (child <= 0)
                break;
            consumed += child;
            cursor += child;
        }
        return consumed;
    }

    int random_node_index(const TokenVec& tokens)
    {
        if (tokens.empty())
            return -1;
        return rand_int(0, static_cast<int>(tokens.size()) - 1);
    }

    void splice_subtree(const TokenVec& src_tokens, const ValueVec& src_values,
                        int begin, int count,
                        TokenVec& dst_tokens, ValueVec& dst_values,
                        const TokenVec& insert_tokens, const ValueVec& insert_values)
    {
        dst_tokens.insert(dst_tokens.end(), src_tokens.begin(), src_tokens.end());
        dst_values.insert(dst_values.end(), src_values.begin(), src_values.end());

        if (count <= 0)
            return;

        dst_tokens.erase(dst_tokens.begin() + begin, dst_tokens.begin() + begin + count);
        dst_values.erase(dst_values.begin() + begin, dst_values.begin() + begin + count);
        dst_tokens.insert(dst_tokens.begin() + begin, insert_tokens.begin(), insert_tokens.end());
        dst_values.insert(dst_values.begin() + begin, insert_values.begin(), insert_values.end());
    }
}

void generate_random_expression(const EvolutionParams& params,
                                std::vector<int>& tokens_out,
                                std::vector<float>& values_out)
{
    generate_with_limit(params, params.max_tokens, tokens_out, values_out);
}

void mutate_expression(const EvolutionParams& params,
                       const std::vector<int>& parent_tokens,
                       const std::vector<float>& parent_values,
                       std::vector<int>& child_tokens,
                       std::vector<float>& child_values)
{
    if (parent_tokens.empty())
    {
        generate_random_expression(params, child_tokens, child_values);
        return;
    }

    int replace_idx = random_node_index(parent_tokens);
    if (replace_idx < 0)
    {
        child_tokens = parent_tokens;
        child_values = parent_values;
        return;
    }

    int old_subtree = subtree_size(parent_tokens, replace_idx);
    if (old_subtree <= 0)
        old_subtree = 1;

    int base_tokens = static_cast<int>(parent_tokens.size()) - old_subtree;
    int max_new_tokens = params.max_tokens - base_tokens;
    if (max_new_tokens <= 0)
    {
        child_tokens = parent_tokens;
        child_values = parent_values;
        return;
    }

    std::vector<int> new_tokens;
    std::vector<float> new_values;
    generate_with_limit(params, max_new_tokens, new_tokens, new_values);

    child_tokens.clear();
    child_values.clear();
    splice_subtree(parent_tokens, parent_values, replace_idx, old_subtree,
                   child_tokens, child_values, new_tokens, new_values);
}

void crossover_expressions(const EvolutionParams& params,
                           const std::vector<int>& left_tokens,
                           const std::vector<float>& left_values,
                           const std::vector<int>& right_tokens,
                           const std::vector<float>& right_values,
                           std::vector<int>& child_tokens,
                           std::vector<float>& child_values)
{
    if (left_tokens.empty() || right_tokens.empty())
    {
        child_tokens = left_tokens;
        child_values = left_values;
        return;
    }

    int left_idx = random_node_index(left_tokens);
    if (left_idx < 0)
    {
        child_tokens = left_tokens;
        child_values = left_values;
        return;
    }

    int left_subtree = subtree_size(left_tokens, left_idx);
    if (left_subtree <= 0)
        left_subtree = 1;

    bool replaced = false;
    TokenVec candidate_tokens;
    ValueVec candidate_values;
    const int attempts = std::min<int>(static_cast<int>(right_tokens.size()), 32);
    for (int attempt = 0; attempt < attempts; ++attempt)
    {
        int right_idx = random_node_index(right_tokens);
        if (right_idx < 0)
            continue;
        int right_subtree = subtree_size(right_tokens, right_idx);
        if (right_subtree <= 0)
            continue;

        int total = static_cast<int>(left_tokens.size()) - left_subtree + right_subtree;
        if (total > params.max_tokens)
            continue;

        candidate_tokens.clear();
        candidate_values.clear();
        candidate_tokens.insert(candidate_tokens.end(),
                                right_tokens.begin() + right_idx,
                                right_tokens.begin() + right_idx + right_subtree);
        candidate_values.insert(candidate_values.end(),
                                right_values.begin() + right_idx,
                                right_values.begin() + right_idx + right_subtree);

        child_tokens.clear();
        child_values.clear();
        splice_subtree(left_tokens, left_values, left_idx, left_subtree,
                       child_tokens, child_values, candidate_tokens, candidate_values);
        replaced = true;
        break;
    }

    if (!replaced)
    {
        child_tokens = left_tokens;
        child_values = left_values;
    }
}
