#include "../utils/utils.h"
#include "../utils/opcodes.h"
#include <vector>
#include <cmath>
#include <iostream>

#ifndef MAX_NUM_FEATURES
#define MAX_NUM_FEATURES 32
#endif

// Helper for single-point evaluation
static float eval_op_cpu(int op, float val1, float val2) {
    const float DELTA = 1e-9f;
    const float MAX_VAL = 1e9f;

    switch (op) {
    case OP_ADD: return val1 + val2;
    case OP_SUB: return val1 - val2;
    case OP_MUL: return val1 * val2;
    case OP_DIV: return (val2 == 0.0f) ? NAN : val1 / val2;
    case OP_POW: return powf(val1, val2);
    case OP_MIN: return fminf(val1, val2);
    case OP_MAX: return fmaxf(val1, val2);
    case OP_LOOSE_DIV: {
        float denom = fabsf(val2) <= DELTA ? (val2 < 0.0f ? -DELTA : DELTA) : val2;
        return val1 / denom;
    }
    case OP_LOOSE_POW: return (val1 == 0.0f && val2 == 0.0f) ? 0.0f : powf(fabsf(val1), val2);
    case OP_SIN: return sinf(val1);
    case OP_COS: return cosf(val1);
    case OP_TAN: return tanf(val1);
    case OP_SINH: return sinhf(val1);
    case OP_COSH: return coshf(val1);
    case OP_TANH: return tanhf(val1);
    case OP_EXP: return expf(val1);
    case OP_LOG: return logf(val1);
    case OP_INV: return 1.0f / val1;
    case OP_ASIN: return asinf(val1);
    case OP_ACOS: return acosf(val1);
    case OP_ATAN: return atanf(val1);
    case OP_LOOSE_LOG: return (val1 == 0.0f) ? -MAX_VAL : logf(fabsf(val1));
    case OP_LOOSE_INV: {
        float denom = fabsf(val1) <= DELTA ? (val1 < 0.0f ? -DELTA : DELTA) : val1;
        return 1.0f / denom;
    }
    case OP_ABS: return fabsf(val1);
    case OP_NEG: return -val1;
    case OP_SQRT: return sqrtf(val1);
    case OP_LOOSE_SQRT: return sqrtf(fabsf(val1));
    default: return 0.0f;
    }
}

// Simple stack-based evaluator for one expression on one datapoint
static float evaluate_single_expr(const int* tokens, const double* values, int num_tokens, 
                                  const double* vars, int num_vars) {
    float stack[MAX_STACK_SIZE];
    int sp = 0;

    // Prefix notation: process from right to left
    for (int i = num_tokens - 1; i >= 0; i--) {
        int tok = tokens[i];
        if (tok == TOK_CONST) {
            stack[sp++] = (float)values[i];
        } else if (tok == TOK_VAR) {
            int var_idx = (int)values[i];
            if (var_idx < num_vars) {
                stack[sp++] = (float)vars[var_idx];
            } else {
                stack[sp++] = 0.0f; // Should not happen
            }
        } else {
            // Operator
            float v1 = stack[--sp];
            float v2 = 0.0f;
            // Check arity
            // Binary ops are 1-9, unary are 10+
            bool is_binary = (tok >= 1 && tok <= 9);
            if (is_binary) {
                v2 = stack[--sp];
            }
            stack[sp++] = eval_op_cpu(tok, v1, v2);
        }
    }
    return stack[0];
}

// Main entry point for CPU evaluation
// Computes MSE for each expression
void evaluate_cpu_mse(InputInfo& input_info, double*** all_vars, std::vector<double>& mses) {
    int num_exprs = input_info.num_exprs;
    mses.resize(num_exprs);

    // For each expression
    #pragma omp parallel for
    for (int i = 0; i < num_exprs; i++) {
        int num_dps = input_info.num_dps[i];
        int num_vars = input_info.num_vars[i];
        int num_tokens = input_info.num_tokens[i];
        int* tokens = input_info.tokens[i];
        double* values = input_info.values[i];
        
        // Data for this expression (vars[var_idx][dp_idx])
        // Note: all_vars is [expr][var][dp]
        double** vars = all_vars[i];
        
        // Ground truth is usually the last variable (index num_vars)
        // But InputInfo structure implies vars are input features.
        // Wait, load_data_file returns [num_vars+1][num_dps], where last row is y.
        double* y_true = vars[num_vars]; 

        double total_sq_err = 0.0;
        int valid_dps = 0;

        for (int dp = 0; dp < num_dps; dp++) {
            // Extract input features for this datapoint
            // We need to pass a pointer to the variables for this specific datapoint
            // But evaluate_single_expr expects an array of variables.
            // Since vars is column-major [var][dp], we can't just pass a pointer.
            // We need to gather them or change evaluate_single_expr.
            // Let's gather for simplicity (stack allocation is cheap)
            double dp_vars[MAX_NUM_FEATURES];
            for (int v = 0; v < num_vars; v++) {
                dp_vars[v] = vars[v][dp];
            }

            float pred = evaluate_single_expr(tokens, values, num_tokens, dp_vars, num_vars);
            
            if (!std::isnan(pred) && !std::isinf(pred)) {
                double diff = pred - y_true[dp];
                total_sq_err += diff * diff;
                valid_dps++;
            }
        }

        if (valid_dps > 0) {
            mses[i] = total_sq_err / valid_dps;
        } else {
            mses[i] = NAN;
        }
    }
}
