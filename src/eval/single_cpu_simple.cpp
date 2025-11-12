#include <iostream>
#include <string>
#include <chrono>
#include <cmath>
#include <fstream>
#include "../utils.h"

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

    // Parse input digest file (validates and prints info)
    InputInfo input_info = parse_input_info(digest_file);

    if (input_info.num_tokens == 0)
    {
        std::cerr << "Error: Failed to parse input file - (num_tokens == 0)" << std::endl;
        return 1;
    }

    int num_vars = input_info.num_vars;
    int num_dps = input_info.num_dps;

    // Load data from file
    double **vars = load_data_file(input_info.data_filename, num_vars, num_dps);
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

    // Save results to file
    save_results(digest_file, pred, vars, num_vars, num_dps);

    // Clean up
    free_input_info(input_info);
    free_data(vars, num_vars);
    free(pred);

    return 0;
}