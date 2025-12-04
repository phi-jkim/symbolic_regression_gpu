#ifndef EVALUATOR_H
#define EVALUATOR_H

#include "../utils/utils.h"

// Unified evaluation interface
// The actual implementation (CPU or GPU) is selected at compile time
// via USE_CPU_SIMPLE, USE_CPU_MULTI, USE_GPU_SIMPLE, USE_GPU_JINHA, or USE_GPU_ASYNC_JINHA preprocessor flags

// Default number of threads for CPU evaluation
#ifndef CPU_EVAL_THREADS
#define CPU_EVAL_THREADS 8
#endif

#if defined(USE_GPU_ASYNC_JINHA)
    // GPU async double-buffer evaluation (defined in gpu_async_jinha.cu)
    void eval_async_jinha_batch(InputInfo &input_info, double ***all_vars, double **all_predictions, EvalMetrics* metrics);
    #define eval_batch eval_async_jinha_batch
#elif defined(USE_GPU_CUSTOM_PEREXPR_MULTI_NVRTC)
    // GPU custom per-expression evaluation using NVRTC C++->PTX multi-expression batch path
    // (defined in gpu_custom_kernel_per_expression.cu)
    void eval_multi_expr_nvrtc_batch(InputInfo &input_info, double ***all_vars, double **all_predictions, EvalMetrics* metrics);
    #define eval_batch eval_multi_expr_nvrtc_batch
#elif defined(USE_GPU_CUSTOM_PEREXPR_MULTI)
    // GPU custom per-expression evaluation using PTX multi-expression batch path
    // (defined in gpu_custom_kernel_per_expression.cu)
    void eval_multi_expr_ptx_batch(InputInfo &input_info, double ***all_vars, double **all_predictions, EvalMetrics* metrics);
    #define eval_batch eval_multi_expr_ptx_batch
#elif defined(USE_GPU_EVOLVE_JINHA_MULTI_EXPRS)
    // GPU evolution-based evaluation using multi-expression batch kernel
    // (defined in gpu_simple_jinha_with_evolve.cu)
    void eval_evolve_jinha_multi_expressions_batch(InputInfo &input_info,
                                                   double ***all_vars,
                                                   double **all_predictions,
                                                   EvalMetrics* metrics);
    #define eval_batch eval_evolve_jinha_multi_expressions_batch
#elif defined(USE_GPU_EVOLVE_JINHA) || defined(USE_GPU_CUSTOM_PEREXPR_EVOLVE)
    // GPU evolution-based evaluation using evolve() (defined in gpu_simple_jinha_with_evolve.cu
    // or gpu_custom_kernel_per_expression.cu for the custom per-expression kernel path)
    void eval_evolve_jinha_batch(InputInfo &input_info,
                                 double ***all_vars,
                                 double **all_predictions,
                                 EvalMetrics* metrics);
    #define eval_batch eval_evolve_jinha_batch
#elif defined(USE_GPU_JINHA_MULTI_EXPRESSION_BATCH)
    // GPU evaluation using multi-expression batch kernel (defined in gpu_simple_jinha.cu)
    void eval_batch_multi_expression_single_kernel(InputInfo &input_info, double ***all_vars, double **all_predictions, EvalMetrics* metrics);
    #define eval_batch eval_batch_multi_expression_single_kernel
    
    // Stateless evolution benchmark (defined in common_eval.cpp)
    int run_evolution_benchmark_stateless(int start_gen, int end_gen, const std::string& data_dir);
#elif defined(USE_GPU_JINHA)
    // GPU evaluation using eval_tree.cu library (defined in gpu_simple_jinha.cu)
    void eval_jinha_batch(InputInfo &input_info, double ***all_vars, double **all_predictions, EvalMetrics* metrics);
    #define eval_batch eval_jinha_batch
#elif defined(USE_GPU_EVOLVE_SIMPLE)
    // GPU evaluation using gpu_simple.cu evolve entry (reuses eval_kernel)
    void eval_evolve_simple_batch(InputInfo &input_info, double ***all_vars, double **all_predictions);
    #define eval_batch eval_evolve_simple_batch
#elif defined(USE_GPU_SIMPLE)
    // GPU evaluation function (defined in gpu_simple.cu)
    void eval_gpu_batch(InputInfo &input_info, double ***all_vars, double **all_predictions, EvalMetrics* metrics);
    #define eval_batch eval_gpu_batch
#elif defined(USE_CPU_MULTI)
    // CPU multi-threaded evaluation function (defined in cpu_simple_multi.cpp)
    void eval_cpu_multi_batch(InputInfo &input_info, double ***all_vars, double **all_predictions, EvalMetrics* metrics);
    #define eval_batch eval_cpu_multi_batch
    
    // Stateless evolution benchmark (defined in common_eval.cpp)
    int run_evolution_benchmark_stateless(int start_gen, int end_gen, const std::string& data_dir);
#elif defined(USE_CPU_SUBTREE)
    // CPU evaluation with common subtree reuse (defined in cpu_subtree.cpp)
    // Supports both single and multi-threaded execution via CPU_EVAL_THREADS env var
    void eval_cpu_common_subtree_batch(InputInfo &input_info, double ***all_vars, double **all_predictions, EvalMetrics* metrics);
    #define eval_batch eval_cpu_common_subtree_batch
    
    // Stateful evaluation for evolution
    #include "../utils/detect.h" // For SubtreeCache
    void eval_cpu_stateful_batch(InputInfo &input_info, double ***all_vars, double **all_predictions, SubtreeCache& cache, EvalMetrics* metrics);
    
    // Evolution benchmark runner
    int run_evolution_benchmark(int start_gen, int end_gen, const std::string& data_dir);
#elif defined(USE_CPU_SIMPLE)
    // CPU single-threaded evaluation function (defined in cpu_simple_single.cpp)
    void eval_cpu_batch(InputInfo &input_info, double ***all_vars, double **all_predictions, EvalMetrics* metrics);
    #define eval_batch eval_cpu_batch
    
    // Stateless evolution benchmark (defined in common_eval.cpp)
    int run_evolution_benchmark_stateless(int start_gen, int end_gen, const std::string& data_dir);
#elif defined(USE_GPU_SUBTREE)
    // GPU Subtree Evaluator (defined in gpu_subtree.cu)
    void eval_gpu_subtree_batch_stateless(InputInfo &input_info, double ***all_vars, double **all_predictions, EvalMetrics* metrics);
    #define eval_batch eval_gpu_subtree_batch_stateless
    
    // Stateless evolution benchmark (defined in common_eval.cpp)
    int run_evolution_benchmark_stateless(int start_gen, int end_gen, const std::string& data_dir);
#else
    #error "Must define USE_CPU_SIMPLE, USE_CPU_MULTI, USE_CPU_SUBTREE, USE_GPU_SIMPLE, USE_GPU_JINHA, USE_GPU_JINHA_MULTI_EXPRESSION_BATCH, or USE_GPU_ASYNC_JINHA"
#endif

#endif // EVALUATOR_H
