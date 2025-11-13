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
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <sys/stat.h>
#include <sys/types.h>

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

    // Read each expression
    for (int expr_id = 0; expr_id < info.num_exprs; expr_id++)
    {
        // Read metadata for this expression
        file >> info.num_vars[expr_id];
        file >> info.num_dps[expr_id];
        file >> info.num_tokens[expr_id];

        std::cout << "  Expression " << (expr_id + 1) << ": " << info.num_vars[expr_id]
                  << " vars, " << info.num_dps[expr_id] << " dps, "
                  << info.num_tokens[expr_id] << " tokens" << std::endl;

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

    // Read data from file
    std::ifstream data_file(filename);
    if (!data_file.is_open())
    {
        std::cerr << "Error: Could not open data file: " << filename << std::endl;
        return vars;
    }

    for (int j = 0; j < num_dps; j++)
    {
        for (int i = 0; i <= num_vars; i++)
            data_file >> vars[i][j];
    }

    data_file.close();
    return vars;
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

    // Load all expression data files
    for (int expr_id = 0; expr_id < input_info.num_exprs; expr_id++)
    {
        int num_vars = input_info.num_vars[expr_id];
        int num_dps = input_info.num_dps[expr_id];
        std::string data_filename = input_info.data_filenames[expr_id];

        all_vars[expr_id] = load_data_file(data_filename, num_vars, num_dps);
    }

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
    // PHASE 3: Evaluate (Total Eval Time)
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
    for (int expr_id = 0; expr_id < input_info.num_exprs; expr_id++)
    {
        free_data(all_vars[expr_id], input_info.num_vars[expr_id]);
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
        }

        // Free arrays
        delete[] info.num_vars;
        delete[] info.num_dps;
        delete[] info.num_tokens;
        delete[] info.tokens;
        delete[] info.values;
        delete[] info.data_filenames;

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
