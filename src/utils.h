#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include <chrono>

// Input information structure
typedef struct {
    int num_vars;
    int num_dps;
    int num_tokens;
    int *tokens;
    double *values;
    std::string data_filename;
} InputInfo;

// Result information structure
typedef struct {
    double *pred;         // Predictions array
    double mse;
    double median;
    double stdev;
    double init_time_ms;   // Initialization time (parse + load + malloc)
    double eval_time_ms;   // Evaluation time (main loop)
    double output_time_ms; // Output processing time (format + stats)
} ResultInfo;

// Parse digest file (validates and prints info)
InputInfo parse_input_info(const std::string& input_file);

// Load data from Feynman data file
double** load_data_file(const std::string& filename, int num_vars, int num_dps);

// Format formula from tokens for display
std::string format_formula(int *tokens, double *values, int num_tokens);

// Create result info with statistics
ResultInfo make_result_info(double *pred, double **vars, int num_vars, int num_dps, 
                             double init_time_ms, double eval_time_ms, double output_time_ms);

// Save evaluation results to file
void save_results(const std::string& digest_file, const InputInfo& input_info, const ResultInfo& result_info, double **vars);

// Free result info
void free_result_info(ResultInfo& info);

// Free allocated memory
void free_input_info(InputInfo& info);
void free_data(double** vars, int num_vars);

// Timing utilities
typedef std::chrono::high_resolution_clock::time_point TimePoint;
inline TimePoint measure_clock() { return std::chrono::high_resolution_clock::now(); }
inline double clock_to_ms(TimePoint start, TimePoint end) { 
    return std::chrono::duration<double, std::milli>(end - start).count(); 
}

#endif
