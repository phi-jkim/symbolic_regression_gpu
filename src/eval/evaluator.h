#ifndef EVALUATOR_H
#define EVALUATOR_H

#include "../utils/utils.h"

// Unified evaluation interface
// The actual implementation (CPU or GPU) is selected at compile time
// via USE_CPU_SIMPLE, USE_CPU_MULTI, USE_GPU_SIMPLE, USE_GPU_JINHA, or USE_GPU_ASYNC_JINHA preprocessor flags

#if defined(USE_GPU_ASYNC_JINHA)
    // GPU async double-buffer evaluation (defined in gpu_async_jinha.cu)
    void eval_async_jinha_batch(InputInfo &input_info, double ***all_vars, double **all_predictions, EvalMetrics* metrics);
    #define eval_batch eval_async_jinha_batch
#elif defined(USE_GPU_EVOLVE_JINHA)
    // GPU evolution-based evaluation using evolve() (defined in gpu_simple_jinha_with_evolve.cu)
    void eval_evolve_jinha_batch(InputInfo &input_info, double ***all_vars, double **all_predictions, EvalMetrics* metrics);
    #define eval_batch eval_evolve_jinha_batch
#elif defined(USE_GPU_JINHA)
    // GPU evaluation using eval_tree.cu library (defined in gpu_simple_jinha.cu)
    void eval_jinha_batch(InputInfo &input_info, double ***all_vars, double **all_predictions, EvalMetrics* metrics);
    #define eval_batch eval_jinha_batch
#elif defined(USE_GPU_SIMPLE)
    // GPU evaluation function (defined in gpu_simple.cu)
    void eval_gpu_batch(InputInfo &input_info, double ***all_vars, double **all_predictions, EvalMetrics* metrics);
    #define eval_batch eval_gpu_batch
#elif defined(USE_CPU_MULTI)
    // CPU multi-threaded evaluation function with 8 workers (defined in cpu_simple_multi.cpp)
    void eval_cpu_batch(InputInfo &input_info, double ***all_vars, double **all_predictions, EvalMetrics* metrics);
    #define eval_batch eval_cpu_batch
#elif defined(USE_CPU_SIMPLE)
    // CPU single-threaded evaluation function (defined in cpu_simple_single.cpp)
    void eval_cpu_batch(InputInfo &input_info, double ***all_vars, double **all_predictions, EvalMetrics* metrics);
    #define eval_batch eval_cpu_batch
#else
    #error "Must define USE_CPU_SIMPLE, USE_CPU_MULTI, USE_GPU_SIMPLE, USE_GPU_JINHA, or USE_GPU_ASYNC_JINHA"
#endif

#endif // EVALUATOR_H
