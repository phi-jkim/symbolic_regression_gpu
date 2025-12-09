#include "../utils/utils.h"
#include "../utils/opcodes.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <thread>
#include <mutex>
#include <cstdlib>
#include <algorithm>

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

// Helper for single-point evaluation
static float evaluate_single_expr(const int* tokens, const double* values, int num_tokens, 
                                  const float* vars, int num_vars) {
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
                stack[sp++] = vars[var_idx];
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
void evaluate_cpu_mse(InputInfo& input_info, float*** all_vars, std::vector<float>& mses, RunStats& stats) {
    TimePoint start = measure_clock();
    
    int num_exprs = input_info.num_exprs;
    mses.resize(num_exprs);

    // For each expression
    // Threading setup
    int n_threads = 1;
    if (const char* env_p = std::getenv("THREADS")) {
        n_threads = std::atoi(env_p);
    } else {
        n_threads = std::thread::hardware_concurrency();
    }
    if (n_threads < 1) n_threads = 1;

    std::vector<std::thread> workers;

    for (int t = 0; t < n_threads; t++) {
        workers.emplace_back([&, t]() {
            int start_idx = (long)num_exprs * t / n_threads;
            int end_idx = (long)num_exprs * (t + 1) / n_threads;
            
            for (int i = start_idx; i < end_idx; i++) {
                int len = input_info.num_tokens[i];
                // Evaluate
                const int* tokens = input_info.tokens[i];
                const double* values = input_info.values[i];
                int num_vars = input_info.num_vars[i];
                int num_dps = input_info.num_dps[i];
                float** vars = all_vars[i]; // vars[v][dp]
                
                // Optimization: Pre-extract pointers
                std::vector<float*> var_ptrs(num_vars);
                for(int v=0; v<num_vars; v++) var_ptrs[v] = vars[v];
                
                // Get Y (last var)
                float* y_true = var_ptrs[num_vars]; // num_vars is features? 
                // Wait, all_vars has size num_vars+1 (last is Y)
                
                double total_sq_err = 0.0;
                int valid_dps = 0;

                // Temp buffer for single DP vars
                float dp_vars[MAX_NUM_FEATURES];

                for (int dp = 0; dp < num_dps; dp++) {
                    for(int v=0; v<num_vars; v++) {
                        dp_vars[v] = var_ptrs[v][dp];
                    }
                    
                    float pred = evaluate_single_expr(tokens, values, len, dp_vars, num_vars);
                    
                    if (!std::isnan(pred) && !std::isinf(pred)) {
                        float diff = pred - y_true[dp];
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
        });
    }
    
    for (auto& w : workers) {
        if (w.joinable()) w.join();
    }
    
    stats.total_eval_time_ms = clock_to_ms(start, measure_clock());
}
