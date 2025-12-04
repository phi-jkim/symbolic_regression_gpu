#include <iostream>
#include <string>
#include <cmath>
#include <thread>
#include <vector>
#include "../utils/utils.h"
#include "evaluator.h"

// Number of CPU worker threads (set via -DCPU_EVAL_THREADS=N at compile time, or defaults in evaluator.h)

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

double eval_tree_cpu(int *tokens, double *values, double *x, int num_tokens, int num_vars)
{
    double stk[MAX_STACK_SIZE]; // Thread-local stack for thread safety
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

// Batch evaluation function for CPU with configurable worker threads
// Processes all expressions and fills prediction arrays
void eval_cpu_multi_batch(InputInfo &input_info, double ***all_vars, double **all_predictions, EvalMetrics* metrics)
{
    const int num_workers = CPU_EVAL_THREADS;
    std::vector<std::thread> threads;

    // Worker function: each thread processes expressions where expr_id % num_workers == worker_id
    auto worker = [&](int worker_id)
    {
        double *x = (double *)malloc(MAX_VAR_NUM * sizeof(double));
        for (int expr_id = worker_id; expr_id < input_info.num_exprs; expr_id += num_workers)
        {
            int num_vars = input_info.num_vars[expr_id];
            int num_dps = input_info.num_dps[expr_id];
            int num_tokens = input_info.num_tokens[expr_id];
            int *tokens = input_info.tokens[expr_id];
            double *values = input_info.values[expr_id];
            double **vars = all_vars[expr_id];
            double *pred = all_predictions[expr_id];

            // Evaluate all datapoints for this expression
            for (int dp = 0; dp < num_dps; dp++)
            {
                // Prepare input variables for this datapoint
                for (int i = 0; i <= num_vars; i++)
                    x[i] = vars[i][dp];

                // Evaluate and store prediction
                pred[dp] = eval_tree_cpu(tokens, values, x, num_tokens, num_vars);
            }
        }
        free(x); // Free the worker's temporary variable array
    };

    // Launch worker threads
    for (int i = 0; i < num_workers; i++)
    {
        threads.emplace_back(worker, i);
    }

    // Wait for all threads to complete
    for (auto &t : threads)
    {
        t.join();
    }
}