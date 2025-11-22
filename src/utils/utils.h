#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include <chrono>

// ============================================================================
// Common Constants
// ============================================================================

// Maximum number of variables (input features) per expression
#define MAX_VAR_NUM 32

// Stack size for expression evaluation (all current expressions use < 16)
#define MAX_STACK_SIZE 16

// Maximum number of tokens per expression
#define MAX_TOKEN_NUM 64

// ============================================================================

// Input information structure (supports 1 or more expressions)
typedef struct {
    int num_exprs;
    int *num_vars;         // Array of size num_exprs
    int *num_dps;          // Array of size num_exprs
    int *num_tokens;       // Array of size num_exprs
    int **tokens;          // 2D array [expr_id][token_id]
    double **values;       // 2D array [expr_id][value_id]
    std::string *data_filenames; // Array of size num_exprs
    // Precomputed maxima across expressions (computed outside of eval)
    int max_tokens;         // max(num_tokens)
    int max_num_dps;        // max(num_dps)
    int max_num_features;   // max(num_vars + 1)
    // Optional prepacked host buffers (if provided, evaluator will memcpy directly)
    // tokens_packed[expr_id]: int[num_tokens[expr_id]]
    // values_packed_f32[expr_id]: float[num_tokens[expr_id]]
    // X_packed_f32[expr_id]: float[num_dps[expr_id] * (num_vars[expr_id] + 1)] in row-major [dp, feature]
    int   **tokens_packed;        // nullable
    float **values_packed_f32;    // nullable
    float **X_packed_f32;         // nullable
    
    // Shared data optimization (for mutation benchmarks)
    bool has_shared_data;  // True if all expressions use the same data file
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

// Evaluation metrics from eval function (optional, can be nullptr)
typedef struct {
    double h2d_transfer_ms;   // Host to device transfer time
    double kernel_exec_ms;    // Kernel execution time
    double d2h_transfer_ms;   // Device to host transfer time
    double total_gpu_ms;      // Total GPU time
    int num_kernel_launches;  // Number of kernel launches
} EvalMetrics;

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
    double total_time_ms;   // Total wall-clock time
    double total_init_ms;   // Total data loading time
    double total_eval_ms;   // Total evaluation time (black box)
    double avg_mse;         // Average MSE across all expressions
    double avg_median;      // Average median across all expressions
    double avg_stdev;       // Average stdev across all expressions
    // Optional eval metrics (nullptr if not provided by eval function)
    EvalMetrics *eval_metrics;
} AggregatedResults;

// Parse input digest file (handles 1 or more expressions)
InputInfo parse_input_info(const std::string& input_file);

// Load data from Feynman data file
double** load_data_file(const std::string& filename, int num_vars, int num_dps);

// Load all data files in parallel (for multi-expression workloads)
void load_all_data_parallel(InputInfo &input_info, double ***all_vars, int num_threads = 8);

// Format formula from tokens for display
std::string format_formula(int *tokens, double *values, int num_tokens);

// Create result info with statistics
ResultInfo make_result_info(double *pred, double **vars, int num_vars, int num_dps, 
                             double init_time_ms, double eval_time_ms, double output_time_ms);

// Save evaluation results to file (single expression)
void save_results(const std::string& digest_file, const ExpressionInfo& expr_info, const ResultInfo& result_info, double **vars);

// Save aggregated results (multi-expression)
void save_aggregated_results(const std::string& digest_file, const AggregatedResults& results);

// Multi-expression batch evaluation function type
// Takes all input data, fills all prediction arrays
// Optional metrics pointer: if non-null, eval function should fill it with timing info
typedef void (*MultiEvalFunc)(InputInfo& input_info, double ***all_vars, double **all_predictions, EvalMetrics* metrics);

// High-level evaluation function (handles both single and multi expressions)
// num_benchmark_runs: number of measured eval runs (default 1 for normal operation)
// warmup_runs: number of warmup runs before measurement (default 0)
void evaluate_and_save_results(const std::string& digest_file, InputInfo& input_info,
                                MultiEvalFunc multi_eval_func,
                                TimePoint start_time,
                                int num_benchmark_runs = 1,
                                int warmup_runs = 0);

// Free result info
void free_result_info(ResultInfo& info);
void free_aggregated_results(AggregatedResults& results);

// Free allocated memory
void free_input_info(InputInfo& info);
void free_data(double** vars, int num_vars);

// ----------------------------------------------------------------------------
// Evolution helpers (single-expression tree ops)
// ----------------------------------------------------------------------------

struct EvolutionParams {
    int num_vars = 1;            // number of input variables (excludes output column)
    int max_tokens = MAX_TOKEN_NUM;
    int max_depth = 5;
    float const_min = -5.0f;
    float const_max = 5.0f;
    float prob_leaf = 0.1f;      // extra probability to terminate early (in addition to depth cap)
    float prob_const_leaf = 0.5f;
    float prob_unary = 0.35f;    // probability of picking a unary op vs binary
};

// Generate a random prefix expression within the provided constraints.
void generate_random_expression(const EvolutionParams& params,
                                std::vector<int>& tokens_out,
                                std::vector<float>& values_out);

// Replace a random subtree with a freshly generated subtree (mutation).
void mutate_expression(const EvolutionParams& params,
                       const std::vector<int>& parent_tokens,
                       const std::vector<float>& parent_values,
                       std::vector<int>& child_tokens,
                       std::vector<float>& child_values);

// Swap a random subtree from the left tree with one sampled from the right tree.
void crossover_expressions(const EvolutionParams& params,
                           const std::vector<int>& left_tokens,
                           const std::vector<float>& left_values,
                           const std::vector<int>& right_tokens,
                           const std::vector<float>& right_values,
                           std::vector<int>& child_tokens,
                           std::vector<float>& child_values);

#endif
