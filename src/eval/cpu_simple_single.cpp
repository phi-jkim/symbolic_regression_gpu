#include <iostream>
#include <string>
#include <cmath>
#include "../utils/utils.h"

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

double stk[MAX_STACK_SIZE];
int sp = 0;

inline void stack_push(double *stk, double val)
{
    stk[sp] = val;
    sp++;
}

inline double stack_pop(double *stk)
{
    sp--;
    return stk[sp];
}

double eval_tree_cpu(int *tokens, double *values, double *x, int num_tokens, int num_vars)
{
    sp = 0;
    double tmp, val1, val2 = 0.0;
    for (int i = num_tokens - 1; i >= 0; i--)
    {
        int tok = tokens[i];
        if (tok > 0) // operation
        {
            val1 = stack_pop(stk);
            if (tok < 10) // binary operation (1-9)
                val2 = stack_pop(stk);

            tmp = eval_op(tok, val1, val2);
            stack_push(stk, tmp);
        }
        else if (tok == 0) // constant
        {
            stack_push(stk, values[i]);
        }
        else if (tok == -1) // variable
        {
            stack_push(stk, x[(int)values[i]]);
        }
        else
        {
            // throw std::runtime_error("Invalid token");
            std::cerr << "Invalid token: " << tok << std::endl;
        }
    }
    return stk[0];
}

// Batch evaluation function for CPU
// Processes all expressions and fills prediction arrays
void eval_cpu_batch(InputInfo &input_info, double ***all_vars, double **all_predictions, EvalMetrics* metrics)
{
    // Sum of CPU evaluation time across expressions (ms)
    double total_cpu_ms = 0.0;

    // Process each expression
    for (int expr_id = 0; expr_id < input_info.num_exprs; expr_id++)
    {
        int num_vars = input_info.num_vars[expr_id];
        int num_dps = input_info.num_dps[expr_id];
        int num_tokens = input_info.num_tokens[expr_id];
        int *tokens = input_info.tokens[expr_id];
        double *values = input_info.values[expr_id];
        double **vars = all_vars[expr_id];
        double *pred = all_predictions[expr_id];

        // Sum only eval_tree_cpu time for this expression (exclude malloc/copies)
        double expr_eval_ms = 0.0;

        // Evaluate all datapoints for this expression
        for (int dp = 0; dp < num_dps; dp++)
        {
            // Prepare input variables for this datapoint
            double *x = (double *)malloc(MAX_VAR_NUM * sizeof(double));
            for (int i = 0; i <= num_vars; i++)
                x[i] = vars[i][dp];
            // Time only the evaluator call
            TimePoint t0 = measure_clock();
            double y = eval_tree_cpu(tokens, values, x, num_tokens, num_vars);
            expr_eval_ms += clock_to_ms(t0, measure_clock());

            // Store prediction (not timed)
            pred[dp] = y;

            free(x);
        }

        total_cpu_ms += expr_eval_ms;
    }

    std::cout << "CPU computation time (eval only): " << total_cpu_ms << " ms" << std::endl;
}