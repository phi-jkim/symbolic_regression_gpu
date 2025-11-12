
// Data input parsing
//  including toml parsing, token mapping, etc
//

#include <iostream>
#include <string>
#include <random>
#include <chrono>
#include <cmath>
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>
std::uniform_real_distribution<double> unif(0, 10);
unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count();
std::default_random_engine re(seed);

#include <string>
#include <sstream>
#include <iomanip>

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
typedef struct
{
    int num_vars;
    int num_dps;
    int num_tokens;
    int *tokens;
    double *values;
    std::string data_filename;
} InputInfo;

InputInfo parse_input_info(std::string input_file)
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

    return info;
}

double eval_op(int op, double val1, double val2)
{
    switch (op)
    {
    case 1:
        return val1 + val2;
    case 2:
        return val1 - val2;
    case 3:
        return val1 * val2;
    case 4:
        return val1 / val2;
    case 5:
        return pow(val1, val2);
    case 10:
        return sin(val1);
    case 11:
        return cos(val1);
    case 12:
        return tan(val1);
    case 13:
        return sinh(val1);
    case 14:
        return cosh(val1);
    case 15:
        return tanh(val1);
    case 16:
        return exp(val1);
    case 17:
        return log(val1);
    case 18:
        return 1.0 / val1; // INV
    case 19:
        return asin(val1); // ASIN
    case 20:
        return acos(val1); // ACOS
    case 21:
        return atan(val1); // ATAN
    case 25:
        return -val1; // NEG
    default:
        return 0;
    }
}

double stk[1024];

double eval_tree_cpu(int *tokens, double *values, double *x, int num_tokens, int num_vars)
{
    int sp = 0;
    double tmp, val1, val2;
    for (int i = num_tokens - 1; i >= 0; i--)
    {
        int tok = tokens[i];
        if (tok > 0) // operation
        {
            val1 = stk[sp - 1], sp--;
            if (tok < 10) // binary operation
                val2 = stk[sp - 1], sp--;

            tmp = eval_op(tok, val1, val2);
            stk[sp] = tmp, sp++;
        }
        else if (tok == 0) // constant
        {
            stk[sp] = values[i], sp++;
        }
        else if (tok == -1) // variable
        {
            stk[sp] = x[(int)values[i]], sp++;
        }
        else
        {
            // throw std::runtime_error("Invalid token");
            std::cerr << "Invalid token: " << tok << std::endl;
        }
    }
    return stk[0];
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <digest_file>" << std::endl;
        return 1;
    }

    std::string digest_file = argv[1];

    // Parse input digest file
    InputInfo input_info = parse_input_info(digest_file);

    if (input_info.num_tokens == 0)
    {
        std::cerr << "Error: Failed to parse input file" << std::endl;
        return 1;
    }

    std::cout << "Loaded digest file: " << digest_file << std::endl;
    std::cout << "Data file: " << input_info.data_filename << std::endl;
    std::cout << "Num vars: " << input_info.num_vars << std::endl;
    std::cout << "Num datapoints: " << input_info.num_dps << std::endl;
    std::cout << "Num tokens: " << input_info.num_tokens << std::endl;

    int num_vars = input_info.num_vars;
    int num_dps = input_info.num_dps;
    double **vars;
    // vars[i][j] means for variable x_i, its value in jth datapoint is vars[i][j].
    // i.e. i has small dim(=num_var < 10), j has large dim(=num_dps ~= 1M).

    vars = (double **)malloc((num_vars + 1) * sizeof(double *));
    for (int i = 0; i <= num_vars; i++)
    {
        vars[i] = (double *)malloc(num_dps * sizeof(double));
    }

    std::ifstream feyn_data_file(input_info.data_filename);

    for (int j = 0; j < num_dps; j++)
    {
        for (int i = 0; i <= num_vars; i++)
            feyn_data_file >> vars[i][j];
    }
    // Use tokens and values from digest file
    int num_tokens = input_info.num_tokens;
    int *tokens = input_info.tokens;
    double *values = input_info.values;

    double *pred = (double *)malloc(num_dps * sizeof(double));

    // run the single tree evaluation (with timing)
    for (int dp = 0; dp < num_dps; dp++)
    {
        double *x = (double *)malloc((num_vars + 1) * sizeof(double));
        for (int i = 0; i <= num_vars; i++)
            x[i] = vars[i][dp];

        double y = eval_tree_cpu(tokens, values, x, num_tokens, num_vars);
        pred[dp] = y;
    }

    std::cout << "\nFormula: " << format_formula(tokens, values, num_tokens) << std::endl;

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

    // write the output info to output file
    std::string output_file = output_dir + "/" + base_name + "_output.txt";
    std::ofstream result_file(output_file, std::ios::out);
    result_file << num_dps << "\n";
    for (int i = 0; i < num_dps; i++)
    {
        // std::cout << "dp: " << i << ", pred: " << pred[i] << std::endl;
        double diff = abs(pred[i] - vars[num_vars][i]);
        if (diff > 1e-10)
            std::cerr << "Warning!\n";
        result_file << diff << " :: " << pred[i] << " : " << vars[num_vars][i] << "\n";
    }
    result_file.close();

    std::cout << "\nResults written to: " << output_file << std::endl;

    // Clean up
    delete[] input_info.tokens;
    delete[] input_info.values;
    for (int i = 0; i < num_vars; i++)
    {
        free(vars[i]);
    }
    free(vars);
    free(pred);

    return 0;
}