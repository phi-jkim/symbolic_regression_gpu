#ifndef EVALUATOR_H
#define EVALUATOR_H

#include "../utils.h"

// Unified evaluation interface
// The actual implementation (CPU or GPU) is selected at compile time
// via USE_CPU_SIMPLE or USE_GPU_SIMPLE preprocessor flags

#if defined(USE_GPU_SIMPLE)
    // GPU evaluation function (defined in gpu_simple.cu)
    void eval_gpu_batch(InputInfo &input_info, double ***all_vars, double **all_predictions);
    #define eval_batch eval_gpu_batch
#elif defined(USE_CPU_SIMPLE)
    // CPU evaluation function (defined in cpu_simple_single.cpp)
    void eval_cpu_batch(InputInfo &input_info, double ***all_vars, double **all_predictions);
    #define eval_batch eval_cpu_batch
#else
    #error "Must define either USE_CPU_SIMPLE or USE_GPU_SIMPLE"
#endif

#endif // EVALUATOR_H
