#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>

// Input information structure
typedef struct {
    int num_vars;
    int num_dps;
    int num_tokens;
    int *tokens;
    double *values;
    std::string data_filename;
} InputInfo;

// Parse digest file (validates and prints info)
InputInfo parse_input_info(const std::string& input_file);

// Load data from Feynman data file
double** load_data_file(const std::string& filename, int num_vars, int num_dps);

// Format formula from tokens for display
std::string format_formula(int *tokens, double *values, int num_tokens);

// Save evaluation results to file
void save_results(const std::string& digest_file, double *pred, double **vars, int num_vars, int num_dps);

// Free allocated memory
void free_input_info(InputInfo& info);
void free_data(double** vars, int num_vars);

#endif
