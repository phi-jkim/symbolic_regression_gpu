#include <iostream>
#include <string>
#include <cmath>
#include <fstream>
#include "../utils.h"

double eval_op(int op, double val1, double val2)
{
    const double DELTA = 1e-9;
    const double MAX_VAL = 1e9;

    switch (op)
    {
    // Binary operators (1-9)
    case 1:
        return val1 + val2; // ADD
    case 2:
        return val1 - val2; // SUB
    case 3:
        return val1 * val2; // MUL
    case 4:
        return (val2 == 0.0) ? NAN : val1 / val2; // DIV
    case 5:
        return pow(val1, val2); // POW
    case 6:
        return (val1 <= val2) ? val1 : val2; // MIN
    case 7:
        return (val1 >= val2) ? val1 : val2; // MAX
    case 8:                                  // LOOSE_DIV
    {
        double denom = fabs(val2) <= DELTA ? (val2 < 0 ? -DELTA : DELTA) : val2;
        return val1 / denom;
    }
    case 9: // LOOSE_POW
        return (val1 == 0.0 && val2 == 0.0) ? 0.0 : pow(fabs(val1), val2);

    // Unary operators (10-27)
    case 10:
        return sin(val1); // SIN
    case 11:
        return cos(val1); // COS
    case 12:
        return tan(val1); // TAN
    case 13:
        return sinh(val1); // SINH
    case 14:
        return cosh(val1); // COSH
    case 15:
        return tanh(val1); // TANH
    case 16:
        return exp(val1); // EXP
    case 17:
        return log(val1); // LOG
    case 18:
        return 1.0 / val1; // INV
    case 19:
        return asin(val1); // ASIN
    case 20:
        return acos(val1); // ACOS
    case 21:
        return atan(val1); // ATAN
    case 22:               // LOOSE_LOG
        return (val1 == 0.0) ? -MAX_VAL : log(fabs(val1));
    case 23: // LOOSE_INV
    {
        double denom = fabs(val1) <= DELTA ? (val1 < 0 ? -DELTA : DELTA) : val1;
        return 1.0 / denom;
    }
    case 24:
        return fabs(val1); // ABS
    case 25:
        return -val1; // NEG
    case 26:
        return sqrt(val1); // SQRT
    case 27:
        return sqrt(fabs(val1)); // LOOSE_SQRT
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
            if (tok < 10) // binary operation (1-9)
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
    TimePoint main_start = measure_clock();

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

    double init_time_ms = clock_to_ms(main_start, measure_clock());

    // Run the single tree evaluation (with timing)
    TimePoint eval_start = measure_clock();

    for (int dp = 0; dp < num_dps; dp++)
    {
        double *x = (double *)malloc((num_vars + 1) * sizeof(double));
        for (int i = 0; i <= num_vars; i++)
            x[i] = vars[i][dp];

        double y = eval_tree_cpu(tokens, values, x, num_tokens, num_vars);
        pred[dp] = y;
    }

    double eval_time_ms = clock_to_ms(eval_start, measure_clock());

    // Format and calculate statistics (with timing)
    TimePoint output_start = measure_clock();

    std::cout << "\nFormula: " << format_formula(tokens, values, num_tokens) << std::endl;

    // Calculate statistics
    ResultInfo result_info = make_result_info(pred, vars, num_vars, num_dps,
                                              init_time_ms, eval_time_ms, 0.0);

    result_info.output_time_ms = clock_to_ms(output_start, measure_clock());

    // Save results to file
    save_results(digest_file, input_info, result_info, vars);

    // Clean up
    free_result_info(result_info);
    free_input_info(input_info);
    free_data(vars, num_vars);

    return 0;
}