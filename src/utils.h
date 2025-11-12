#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include <chrono>

// Input information structure (supports 1 or more expressions)
typedef struct {
    int num_exprs;
    int *num_vars;         // Array of size num_exprs
    int *num_dps;          // Array of size num_exprs
    int *num_tokens;       // Array of size num_exprs
    int **tokens;          // 2D array [expr_id][token_id]
    double **values;       // 2D array [expr_id][value_id]
    std::string *data_filenames; // Array of size num_exprs
} InputInfo;

// Single expression view (for compatibility with save_results)
typedef struct {
    int num_vars;
    int num_dps;
    int num_tokens;
    int *tokens;
    double *values;
    std::string data_filename;
} ExpressionInfo;

// Timing utilities
typedef std::chrono::high_resolution_clock::time_point TimePoint;
inline TimePoint measure_clock() { return std::chrono::high_resolution_clock::now(); }
inline double clock_to_ms(TimePoint start, TimePoint end) { 
    return std::chrono::duration<double, std::milli>(end - start).count(); 
}

// Result information structure (per expression)
typedef struct {
    double *pred;         // Predictions array
    double mse;
    double median;
    double stdev;
    double init_time_ms;   // Initialization time (parse + load + malloc)
    double eval_time_ms;   // Evaluation time (main loop)
    double output_time_ms; // Output processing time (format + stats)
} ResultInfo;

// Aggregated results for multi-expression evaluation
typedef struct {
    int num_exprs;
    double *mse_values;     // MSE for each expression
    double *median_values;  // Median for each expression
    double *stdev_values;   // StdDev for each expression
    double *init_times;     // Init time for each expression
    double *eval_times;     // Eval time for each expression
    double total_time_ms;   // Total wall-clock time
    double total_init_ms;   // Sum of all init times
    double total_eval_ms;   // Sum of all eval times
    double avg_mse;         // Average MSE across all expressions
    double avg_median;      // Average median across all expressions
    double avg_stdev;       // Average stdev across all expressions
} AggregatedResults;

// Parse input digest file (handles 1 or more expressions)
InputInfo parse_input_info(const std::string& input_file);

// Load data from Feynman data file
double** load_data_file(const std::string& filename, int num_vars, int num_dps);

// Format formula from tokens for display
std::string format_formula(int *tokens, double *values, int num_tokens);

// Create result info with statistics
ResultInfo make_result_info(double *pred, double **vars, int num_vars, int num_dps, 
                             double init_time_ms, double eval_time_ms, double output_time_ms);

// Save evaluation results to file (single expression)
void save_results(const std::string& digest_file, const ExpressionInfo& expr_info, const ResultInfo& result_info, double **vars);

// Save aggregated results (multi-expression)
void save_aggregated_results(const std::string& digest_file, const AggregatedResults& results);

// High-level evaluation function (handles both single and multi expressions)
void evaluate_and_save_results(const std::string& digest_file, InputInfo& input_info,
                                double (*eval_func)(int*, double*, double*, int, int),
                                TimePoint start_time);

// Free result info
void free_result_info(ResultInfo& info);
void free_aggregated_results(AggregatedResults& results);

// Free allocated memory
void free_input_info(InputInfo& info);
void free_data(double** vars, int num_vars);

#endif
