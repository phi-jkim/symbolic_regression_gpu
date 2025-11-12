#include "utils.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <sys/stat.h>
#include <sys/types.h>

std::string format_formula(int *tokens, double *values, int num_tokens)
{
    std::vector<std::string> stack;

    for (int i = num_tokens - 1; i >= 0; i--)
    {
        int token = tokens[i];

        switch (token)
        {
        case 0: // Constant
            stack.push_back(std::to_string(values[i]));
            break;
        case -1: // Variable
            stack.push_back("x" + std::to_string(static_cast<int>(values[i])));
            break;
        case 1: // ADD
        case 2: // SUB
        case 3: // MUL
        case 4: // DIV
        case 5: // POW
            if (stack.size() < 2)
                return "INVALID";
            {
                std::string a = stack.back();
                stack.pop_back();
                std::string b = stack.back();
                stack.pop_back();
                const char *op = (token == 1) ? " + " : (token == 2) ? " - "
                                                    : (token == 3)   ? " * "
                                                    : (token == 4)   ? " / "
                                                                     : " ** ";
                stack.push_back("(" + a + op + b + ")");
            }
            break;
        case 10: // SIN
        case 11: // COS
        case 12: // TAN
        case 13: // SINH
        case 14: // COSH
        case 15: // TANH
        case 16: // EXP
        case 17: // LOG
        case 18: // INV
        case 19: // ASIN
        case 20: // ACOS
        case 21: // ATAN
        case 25: // NEG
            if (stack.empty())
                return "INVALID";
            {
                std::string a = stack.back();
                stack.pop_back();
                const char *func = (token == 10) ? "sin" : (token == 11) ? "cos"
                                                       : (token == 12)   ? "tan"
                                                       : (token == 13)   ? "sinh"
                                                       : (token == 14)   ? "cosh"
                                                       : (token == 15)   ? "tanh"
                                                       : (token == 16)   ? "exp"
                                                       : (token == 17)   ? "log"
                                                       : (token == 18)   ? "inv"
                                                       : (token == 19)   ? "asin"
                                                       : (token == 20)   ? "acos"
                                                       : (token == 21)   ? "atan"
                                                       : (token == 25)   ? "-"
                                                                         : "?";
                stack.push_back(std::string(func) + "(" + a + ")");
            }
            break;
        default:
            return "?";
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

InputInfo parse_input_info(const std::string& input_file)
{
    InputInfo info;
    std::ifstream file(input_file);

    if (!file.is_open())
    {
        std::cerr << "Error: Could not open input file: " << input_file << std::endl;
        info.num_vars = 0;
        info.num_dps = 0;
        info.num_tokens = 0;
        info.tokens = nullptr;
        info.values = nullptr;
        return info;
    }

    // Read num_vars
    file >> info.num_vars;

    // Read num_dps
    file >> info.num_dps;

    // Read num_tokens
    file >> info.num_tokens;

    // Allocate and read tokens
    info.tokens = new int[info.num_tokens];
    for (int i = 0; i < info.num_tokens; i++)
    {
        file >> info.tokens[i];
    }

    // Allocate and read values
    info.values = new double[info.num_tokens];
    for (int i = 0; i < info.num_tokens; i++)
    {
        file >> info.values[i];
    }

    // Read data filename (rest of the line)
    file.ignore(); // Skip newline
    std::getline(file, info.data_filename);

    file.close();
    
    // Validate
    if (info.num_tokens == 0)
    {
        std::cerr << "Error: Failed to parse input file" << std::endl;
        return info;
    }
    
    // Print info
    std::cout << "Loaded digest file: " << input_file << std::endl;
    std::cout << "Data file: " << info.data_filename << std::endl;
    std::cout << "Num vars: " << info.num_vars << std::endl;
    std::cout << "Num datapoints: " << info.num_dps << std::endl;
    std::cout << "Num tokens: " << info.num_tokens << std::endl;

    return info;
}

double** load_data_file(const std::string& filename, int num_vars, int num_dps)
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

void save_results(const std::string& digest_file, double *pred, double **vars, int num_vars, int num_dps)
{
    // Create output directory if it doesn't exist
    std::string output_dir = "data/output/ai_feyn";
    mkdir("data", 0755);
    mkdir("data/output", 0755);
    mkdir(output_dir.c_str(), 0755);

    // Extract filename from digest_file path
    size_t last_slash = digest_file.find_last_of('/');
    std::string filename = (last_slash != std::string::npos) ? digest_file.substr(last_slash + 1) : digest_file;
    size_t dot_pos = filename.find_last_of('.');
    std::string base_name = (dot_pos != std::string::npos) ? filename.substr(0, dot_pos) : filename;

    // Write the output info to output file
    std::string output_file = output_dir + "/" + base_name + "_output.txt";
    std::ofstream result_file(output_file, std::ios::out);
    result_file << num_dps << "\n";
    
    for (int i = 0; i < num_dps; i++)
    {
        double diff = std::abs(pred[i] - vars[num_vars][i]);
        if (diff > 1e-10)
            std::cerr << "Warning!\n";
        result_file << diff << " :: " << pred[i] << " : " << vars[num_vars][i] << "\n";
    }
    result_file.close();

    std::cout << "\nResults written to: " << output_file << std::endl;
}

void free_input_info(InputInfo& info)
{
    if (info.tokens != nullptr)
    {
        delete[] info.tokens;
        info.tokens = nullptr;
    }
    if (info.values != nullptr)
    {
        delete[] info.values;
        info.values = nullptr;
    }
}

void free_data(double** vars, int num_vars)
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
