# Use system CUDA instead of conda/miniforge CUDA to avoid incomplete installations
# NVCC ?= /usr/local/cuda/bin/nvcc
NVCC = nvcc
# Target NVIDIA L40S (Ada, SM 89). Embed PTX for forward-compat JIT.
# Use GCC 12 for NVCC to avoid compatibility issues with GCC 13+

# Adjust -arch based on your GPU:
# - sm_60: Pascal (GTX 10 series, Tesla P100)
# - sm_70: Volta (Tesla V100)
# - sm_75: Turing (RTX 20 series, Tesla T4)
# - sm_80: Ampere (RTX 30 series, A100)
# - sm_86: RTX 3090
# - sm_89: RTX 4090, L40S
GPU_ARCH ?= sm_75

NVFLAGS ?= -O3 \
  -gencode arch=compute_89,code=sm_89 \
  -gencode arch=compute_89,code=compute_89 \
  -Xptxas -v

# NVIDIA H200 
# NVFLAGS ?= -O3 \
#   -gencode arch=compute_90,code=sm_90 \
#   -gencode arch=compute_90,code=compute_90 \
#   -Wno-deprecated-gpu-targets
TARGET := libevaltree.so
SRC := eval_tree.cu

# Test and benchmark directories
TEST_DIR := src/test
BENCH_DIR := src/bench
TEST_BIN := test_eval
BENCH_BIN := benchmark_eval

all: $(TARGET) run_gpu

$(TARGET): $(SRC)
	$(NVCC) $(NVFLAGS) -Xcompiler -fPIC -shared -o $@ $<

run_gpu: eval_tree.cu run_gpu.cu
	$(NVCC) $(NVFLAGS) -o $@ run_gpu.cu eval_tree.cu

run_async_test: eval_tree.cu run_async_test.cu
	$(NVCC) $(NVFLAGS) -o $@ run_async_test.cu eval_tree.cu

.PHONY: test_async
test_async: run_async_test
	./run_async_test

# Test target
$(TEST_BIN): $(TEST_DIR)/test_eval.cpp eval_tree.cu
	$(NVCC) $(NVFLAGS) -I$(TEST_DIR) -o $@ $(TEST_DIR)/test_eval.cpp eval_tree.cu

test: $(TEST_BIN)
	./$(TEST_BIN)

# Benchmark target
$(BENCH_BIN): $(BENCH_DIR)/benchmark_eval.cpp eval_tree.cu
	$(NVCC) $(NVFLAGS) -I$(BENCH_DIR) -o $@ $(BENCH_DIR)/benchmark_eval.cpp eval_tree.cu

bench: $(BENCH_BIN)
	./$(BENCH_BIN)

# Convenience targets
run_test: test

run_bench: bench

# ============================================================================
# CPU Build All Target
# ============================================================================
all_cpu: cpu_eval cpu_multi_eval cpu_common

# ============================================================================
# CPU Single-threaded Evaluation Targets
# ============================================================================
CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -pthread

BUILD_DIR = build
CPU_EVAL_BIN = $(BUILD_DIR)/cpu_eval
UTILS_SRC = src/utils/utils.cpp src/utils/detect.cpp
UTILS_HDR = src/utils/utils.h src/utils/detect.h
MAIN_SRC = src/main.cpp
CPU_EVAL_SRC = src/eval/cpu_simple_single.cpp src/eval/common_eval.cpp
EVALUATOR_HDR = src/eval/evaluator.h

$(CPU_EVAL_BIN): $(MAIN_SRC) $(UTILS_SRC) $(UTILS_HDR) $(CPU_EVAL_SRC) $(EVALUATOR_HDR)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -DUSE_CPU_SIMPLE -o $@ $(MAIN_SRC) $(UTILS_SRC) $(CPU_EVAL_SRC)

cpu_eval: $(CPU_EVAL_BIN)

# Test with single expression
run_cpu_eval_single: $(CPU_EVAL_BIN)
	$(CPU_EVAL_BIN) data/ai_feyn/singles/input_001.txt

# Test with multiple expressions
run_cpu_eval_multi: $(CPU_EVAL_BIN)
	$(CPU_EVAL_BIN) data/ai_feyn/multi/input_100_100k.txt

# Test with mutation file (100 mutations, shared data)
run_cpu_eval_mutations: $(CPU_EVAL_BIN)
	$(CPU_EVAL_BIN) data/ai_feyn/mutations/input_base056_100mut_100k.txt

# Test with sample input (2 expressions, 1000 data points each)
run_cpu_eval_sample: $(CPU_EVAL_BIN)
	$(CPU_EVAL_BIN) data/examples/sample_input.txt

# Default run target (single expression)
run_cpu_eval: run_cpu_eval_single

# ============================================================================
# CPU Common Subtree Evaluation Targets
# ============================================================================
CPU_COMMON_BIN = $(BUILD_DIR)/cpu_common_subtree
CPU_COMMON_SRC = src/eval/cpu_subtree.cpp

$(CPU_COMMON_BIN): $(MAIN_SRC) $(UTILS_SRC) $(UTILS_HDR) $(CPU_COMMON_SRC) $(EVALUATOR_HDR)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -DUSE_CPU_SUBTREE -o $@ $(MAIN_SRC) $(UTILS_SRC) $(CPU_COMMON_SRC)

# Test with mutation file (ideal use case)
run_cpu_common_mutations: $(CPU_COMMON_BIN)
	$(CPU_COMMON_BIN) data/ai_feyn/mutations/input_base056_100mut_100k.txt

# Convenience target
cpu_common: $(CPU_COMMON_BIN)

# ============================================================================
# CPU Multi-threaded Evaluation Targets
# ============================================================================
CPU_MULTI_EVAL_BIN = $(BUILD_DIR)/cpu_multi_eval
CPU_MULTI_EVAL_SRC = src/eval/cpu_simple_multi.cpp src/eval/common_eval.cpp

# Number of CPU worker threads (configurable at compile time)
CPU_EVAL_THREADS ?= 8

$(CPU_MULTI_EVAL_BIN): $(MAIN_SRC) $(UTILS_SRC) $(UTILS_HDR) $(CPU_MULTI_EVAL_SRC) $(EVALUATOR_HDR)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -DUSE_CPU_MULTI -DCPU_EVAL_THREADS=$(CPU_EVAL_THREADS) -o $@ $(MAIN_SRC) $(UTILS_SRC) $(CPU_MULTI_EVAL_SRC)

cpu_multi_eval: $(CPU_MULTI_EVAL_BIN)

# Test with single expression
run_cpu_multi_eval_single: $(CPU_MULTI_EVAL_BIN)
	$(CPU_MULTI_EVAL_BIN) data/ai_feyn/singles/input_001.txt

# Test with multiple expressions
run_cpu_multi_eval_multi: $(CPU_MULTI_EVAL_BIN)
	$(CPU_MULTI_EVAL_BIN) data/ai_feyn/multi/input_100_100k.txt

# Test with sample input (2 expressions, 1000 data points each)
run_cpu_multi_eval_sample: $(CPU_MULTI_EVAL_BIN)
	$(CPU_MULTI_EVAL_BIN) data/examples/sample_input.txt

run_cpu_multi_eval_mutations: $(CPU_MULTI_EVAL_BIN)
	$(CPU_MULTI_EVAL_BIN) data/ai_feyn/mutations/input_base056_100mut_100k.txt

# Default run target (multi expression)
run_cpu_multi_eval: run_cpu_multi_eval_multi

# ============================================================================
# GPU Evaluation Targets (for Feynman equation evaluation)
# ============================================================================
GPU_EVAL_BIN = $(BUILD_DIR)/gpu_eval
GPU_EVAL_SRC = src/eval/gpu_simple.cu

# Adjust -arch based on your GPU:
# - sm_60: Pascal (GTX 10 series, Tesla P100)
# - sm_70: Volta (Tesla V100)
# - sm_75: Turing (RTX 20 series, Tesla T4)
# - sm_80: Ampere (RTX 30 series, A100)
# - sm_86: RTX 3090
# - sm_89: RTX 4090, L40S
GPU_ARCH ?= sm_89
# GPU_ARCH ?= sm_90
# Use GCC 12 for NVCC to avoid compatibility issues with GCC 13+
# Suppress deprecated GPU target warnings
NVCCFLAGS = -std=c++17 -O3 -arch=$(GPU_ARCH) \
	-Wno-deprecated-gpu-targets -Xptxas -v


$(GPU_EVAL_BIN): $(MAIN_SRC) $(UTILS_SRC) $(UTILS_HDR) $(GPU_EVAL_SRC) src/eval/common_eval.cpp $(EVALUATOR_HDR)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -DUSE_GPU_SIMPLE -o $@ $(MAIN_SRC) $(UTILS_SRC) $(GPU_EVAL_SRC) src/eval/common_eval.cpp

# Test with single expression
run_gpu_eval_single: $(GPU_EVAL_BIN)
	$(GPU_EVAL_BIN) data/ai_feyn/singles/input_001.txt

# Test with multiple expressions
run_gpu_eval_multi: $(GPU_EVAL_BIN)
	$(GPU_EVAL_BIN) data/ai_feyn/multi/input_100_1000k.txt

# Test with sample input (2 expressions, 1000 data points each)
run_gpu_eval_sample: $(GPU_EVAL_BIN)
	$(GPU_EVAL_BIN) data/examples/sample_input.txt

# Test with mutation file (100 mutations, shared data)
run_gpu_eval_mutations: $(GPU_EVAL_BIN)
	$(GPU_EVAL_BIN) data/ai_feyn/mutations/input_base056_100mut_1000k.txt

# Default GPU run target (multi expression)
run_gpu_eval: run_gpu_eval_multi

# ============================================================================
# GPU Evolve Simple Evaluation (uses gpu_simple.cu eval_kernel)
# ============================================================================
GPU_EVOLVE_SIMPLE_BIN = $(BUILD_DIR)/gpu_evolve_simple
GPU_EVOLVE_SIMPLE_SRC = src/eval/gpu_simple.cu



$(GPU_EVOLVE_SIMPLE_BIN): $(MAIN_SRC) $(UTILS_SRC) $(UTILS_HDR) $(GPU_EVOLVE_SIMPLE_SRC) src/eval/common_eval.cpp src/utils/generate.cu $(EVALUATOR_HDR)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -DUSE_GPU_EVOLVE_SIMPLE -o $@ $(MAIN_SRC) $(UTILS_SRC) $(GPU_EVOLVE_SIMPLE_SRC) src/eval/common_eval.cpp src/utils/generate.cu

# Test with single expression
run_gpu_evolve_simple_single: $(GPU_EVOLVE_SIMPLE_BIN)
	$(GPU_EVOLVE_SIMPLE_BIN) data/ai_feyn/singles/input_001.txt

# Test with multiple expressions
run_gpu_evolve_simple_multi: $(GPU_EVOLVE_SIMPLE_BIN)
	$(GPU_EVOLVE_SIMPLE_BIN) data/ai_feyn/multi/input_100_100k.txt

# Test with sample input (2 expressions, 1000 data points each)
run_gpu_evolve_simple_sample: $(GPU_EVOLVE_SIMPLE_BIN)
	$(GPU_EVOLVE_SIMPLE_BIN) data/examples/sample_input.txt

# Default run target (single expression)
run_gpu_evolve_simple: run_gpu_evolve_simple_sample

# ============================================================================
# GPU Jinha Multi-Expression Batch Evaluation (uses batch kernel)
# ============================================================================
GPU_JINHA_BATCH_BIN = $(BUILD_DIR)/gpu_jinha_batch_eval
GPU_JINHA_BATCH_SRC = src/eval/gpu_simple_jinha.cu

$(GPU_JINHA_BATCH_BIN): $(MAIN_SRC) $(UTILS_SRC) $(UTILS_HDR) $(UTILS_CU_SRC) $(GPU_JINHA_BATCH_SRC) src/eval/common_eval.cpp $(EVALUATOR_HDR)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -DUSE_GPU_JINHA_MULTI_EXPRESSION_BATCH -o $@ $(MAIN_SRC) $(UTILS_SRC) $(GPU_JINHA_BATCH_SRC) src/eval/common_eval.cpp $(UTILS_CU_SRC)

# Test with single expression (batch kernel)
run_gpu_jinha_batch_single: $(GPU_JINHA_BATCH_BIN)
	$(GPU_JINHA_BATCH_BIN) data/ai_feyn/singles/input_001.txt

# Test with multiple expressions (batch kernel)
run_gpu_jinha_batch_multi: $(GPU_JINHA_BATCH_BIN)
	$(GPU_JINHA_BATCH_BIN) data/ai_feyn/multi/input_100_100k.txt

# Test with sample input (batch kernel)
run_gpu_jinha_batch_sample: $(GPU_JINHA_BATCH_BIN)
	$(GPU_JINHA_BATCH_BIN) data/examples/sample_input.txt

# Test with mutation file (batch kernel)
run_gpu_jinha_batch_mutations: $(GPU_JINHA_BATCH_BIN)
	$(GPU_JINHA_BATCH_BIN) data/ai_feyn/mutations/input_base056_100mut_1000k.txt

# Test with 100 mutations file (1M datapoints per expression - batch kernel)
run_gpu_jinha_batch_100_mutations_1000k: $(GPU_JINHA_BATCH_BIN)
	$(GPU_JINHA_BATCH_BIN) data/ai_feyn/mutations/input_base056_100mut_1000k.txt

# Default GPU Jinha Batch run target
run_gpu_jinha_batch: run_gpu_jinha_batch_sample

# ============================================================================
# GPU Jinha Evaluation (using eval_tree.cu library)
# ============================================================================
GPU_JINHA_BIN = $(BUILD_DIR)/gpu_jinha_eval
GPU_JINHA_SRC = src/eval/gpu_simple_jinha.cu
UTILS_CU_SRC = src/utils/utils.cu

$(GPU_JINHA_BIN): $(MAIN_SRC) $(UTILS_SRC) $(UTILS_HDR) $(UTILS_CU_SRC) $(GPU_JINHA_SRC) src/eval/common_eval.cpp $(EVALUATOR_HDR)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -DUSE_GPU_JINHA -o $@ $(MAIN_SRC) $(UTILS_SRC) $(GPU_JINHA_SRC) src/eval/common_eval.cpp $(UTILS_CU_SRC)

# Test with single expression
run_gpu_jinha_eval_single: $(GPU_JINHA_BIN)
	$(GPU_JINHA_BIN) data/ai_feyn/singles/input_001.txt

# Test with multiple expressions
run_gpu_jinha_eval_multi: $(GPU_JINHA_BIN)
	$(GPU_JINHA_BIN) data/ai_feyn/multi/input_100_100k.txt

# Test with sample input (2 expressions, 1000 data points each)
run_gpu_jinha_eval_sample: $(GPU_JINHA_BIN)
	$(GPU_JINHA_BIN) data/examples/sample_input.txt

# Test with mutation file (100 mutations, shared data)
run_gpu_jinha_eval_mutations: $(GPU_JINHA_BIN)
	$(GPU_JINHA_BIN) data/ai_feyn/mutations/input_base056_100mut_1000k.txt

# Test with 100 mutations file (1M datapoints per expression - multi-expression per kernel)
run_gpu_jinha_eval_100_mutations_1000k_batch_expressions: $(GPU_JINHA_BATCH_BIN)
	$(GPU_JINHA_BATCH_BIN) data/ai_feyn/mutations/input_base056_100mut_1000k.txt

# Default GPU Jinha run target
run_gpu_jinha_eval: run_gpu_jinha_eval_sample

# ============================================================================
# GPU Evolve Jinha Evaluation (experimental, uses gpu_simple_jinha_with_evolve.cu)
# ============================================================================
GPU_EVOLVE_JINHA_BIN = $(BUILD_DIR)/gpu_evolve_jinha_eval
GPU_EVOLVE_JINHA_SRC = src/eval/gpu_simple_jinha_with_evolve.cu
EVOLVE_KERNEL_SRC = src/utils/generate.cu src/utils/mutation.cu src/utils/crossover.cu

$(GPU_EVOLVE_JINHA_BIN): $(MAIN_SRC) $(UTILS_SRC) $(UTILS_HDR) $(UTILS_CU_SRC) $(GPU_EVOLVE_JINHA_SRC) src/eval/common_eval.cpp $(EVOLVE_KERNEL_SRC) $(EVALUATOR_HDR)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -DUSE_GPU_EVOLVE_JINHA -o $@ $(MAIN_SRC) $(UTILS_SRC) $(GPU_EVOLVE_JINHA_SRC) src/eval/common_eval.cpp $(UTILS_CU_SRC) $(EVOLVE_KERNEL_SRC)

# PTX output for gpu_evolve_jinha_eval (for sanity checking)
GPU_EVOLVE_JINHA_PTX = $(BUILD_DIR)/gpu_evolve_jinha_eval.ptx

# Generate PTX only for the CUDA utility kernels (eval_prefix_kernel_batch, etc.)
# nvcc -ptx with -o requires exactly one input source file.
$(GPU_EVOLVE_JINHA_PTX): $(UTILS_CU_SRC)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -DUSE_GPU_EVOLVE_JINHA -ptx -o $@ $(UTILS_CU_SRC)

.PHONY: ptx_gpu_evolve_jinha
ptx_gpu_evolve_jinha: $(GPU_EVOLVE_JINHA_PTX)

# Test with single expression
run_gpu_evolve_jinha_eval_single: $(GPU_EVOLVE_JINHA_BIN)
	$(GPU_EVOLVE_JINHA_BIN) data/ai_feyn/singles/input_001.txt

# Test with multiple expressions
run_gpu_evolve_jinha_eval_multi: $(GPU_EVOLVE_JINHA_BIN)
	$(GPU_EVOLVE_JINHA_BIN) data/ai_feyn/multi/input_100_1000k.txt

# Test with sample input (2 expressions, 1000 data points each)
run_gpu_evolve_jinha_eval_sample: $(GPU_EVOLVE_JINHA_BIN)
	$(GPU_EVOLVE_JINHA_BIN) data/examples/sample_input.txt

# Default GPU Evolve Jinha run target
run_gpu_evolve_jinha_eval: run_gpu_evolve_jinha_eval_sample

# Test with 100k mutations file (evolve version)
run_gpu_evolve_jinha_eval_100k_mutations: $(GPU_EVOLVE_JINHA_BIN)
	$(GPU_EVOLVE_JINHA_BIN) data/ai_feyn/mutations/input_base056_100kmut_1000k.txt

# ============================================================================
# GPU Evolve Jinha Multi-Expressions Batch Single Kernel Evaluation (uses multi-expression batch evolve)
# ============================================================================
GPU_EVOLVE_JINHA_BATCH_MULTI_EXPRS_SINGLE_KERNEL_BIN = $(BUILD_DIR)/gpu_evolve_jinha_batch_multi_expressions_single_kernel_eval

$(GPU_EVOLVE_JINHA_BATCH_MULTI_EXPRS_SINGLE_KERNEL_BIN): $(MAIN_SRC) $(UTILS_SRC) $(UTILS_HDR) $(UTILS_CU_SRC) $(GPU_EVOLVE_JINHA_SRC) src/eval/common_eval.cpp $(EVOLVE_KERNEL_SRC) $(EVALUATOR_HDR)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -DUSE_GPU_EVOLVE_JINHA_MULTI_EXPRS -o $@ $(MAIN_SRC) $(UTILS_SRC) $(GPU_EVOLVE_JINHA_SRC) src/eval/common_eval.cpp $(UTILS_CU_SRC) $(EVOLVE_KERNEL_SRC)

# Test with single expression (multi-expressions batch single kernel)
run_gpu_evolve_jinha_batch_multi_expressions_single_kernel_single: $(GPU_EVOLVE_JINHA_BATCH_MULTI_EXPRS_SINGLE_KERNEL_BIN)
	$(GPU_EVOLVE_JINHA_BATCH_MULTI_EXPRS_SINGLE_KERNEL_BIN) data/ai_feyn/singles/input_001.txt

# Test with multiple expressions (multi-expressions batch single kernel)
run_gpu_evolve_jinha_batch_multi_expressions_single_kernel_multi: $(GPU_EVOLVE_JINHA_BATCH_MULTI_EXPRS_SINGLE_KERNEL_BIN)
	$(GPU_EVOLVE_JINHA_BATCH_MULTI_EXPRS_SINGLE_KERNEL_BIN) data/ai_feyn/multi/input_100_1000k.txt

# Test with sample input (2 expressions, 1000 data points each, multi-expressions batch single kernel)
run_gpu_evolve_jinha_batch_multi_expressions_single_kernel_sample: $(GPU_EVOLVE_JINHA_BATCH_MULTI_EXPRS_SINGLE_KERNEL_BIN)
	$(GPU_EVOLVE_JINHA_BATCH_MULTI_EXPRS_SINGLE_KERNEL_BIN) data/examples/sample_input.txt

# Default GPU Evolve Jinha batch multi-expressions single kernel run target
run_gpu_evolve_jinha_batch_multi_expressions_single_kernel: run_gpu_evolve_jinha_batch_multi_expressions_single_kernel_sample

# =========================================================================
# GPU Custom-Kernel-Per-Expression Evolve (uses gpu_custom_kernel_per_expression.cu)
# =========================================================================
GPU_CUSTOM_PEREXPR_BIN = $(BUILD_DIR)/gpu_custom_kernel_perexpression_evolve
GPU_CUSTOM_PEREXPR_SRC = src/eval/gpu_custom_kernel_per_expression.cu

$(GPU_CUSTOM_PEREXPR_BIN): $(MAIN_SRC) $(UTILS_SRC) $(UTILS_HDR) $(UTILS_CU_SRC) $(GPU_CUSTOM_PEREXPR_SRC) src/eval/common_eval.cpp $(EVOLVE_KERNEL_SRC) $(EVALUATOR_HDR)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -DUSE_GPU_CUSTOM_PEREXPR_EVOLVE -o $@ $(MAIN_SRC) $(UTILS_SRC) $(GPU_CUSTOM_PEREXPR_SRC) src/eval/common_eval.cpp $(UTILS_CU_SRC) $(EVOLVE_KERNEL_SRC) \
		-lnvrtc -lcuda

# Test with single expression (evolve)
run_gpu_custom_kernel_perexpression_evolve_single: $(GPU_CUSTOM_PEREXPR_BIN)
	$(GPU_CUSTOM_PEREXPR_BIN) data/ai_feyn/singles/input_001.txt

# Test with multiple expressions (evolve)
run_gpu_custom_kernel_perexpression_evolve_multi: $(GPU_CUSTOM_PEREXPR_BIN)
	$(GPU_CUSTOM_PEREXPR_BIN) data/ai_feyn/multi/input_100_1000k.txt

# Test with sample input (2 expressions, 1000 data points each, evolve)
run_gpu_custom_kernel_perexpression_evolve_sample: $(GPU_CUSTOM_PEREXPR_BIN)
	$(GPU_CUSTOM_PEREXPR_BIN) data/examples/sample_input.txt

# Default GPU custom-kernel-per-expression evolve run target
run_gpu_custom_kernel_perexpression_evolve: run_gpu_custom_kernel_perexpression_evolve_sample

# =========================================================================
# GPU Custom-Kernel-Per-Expression Non-Evolve Multi (PTX batch path)
# =========================================================================
GPU_CUSTOM_PEREXPR_MULTI_BIN = $(BUILD_DIR)/gpu_custom_kernel_perexpression_multi

$(GPU_CUSTOM_PEREXPR_MULTI_BIN): $(MAIN_SRC) $(UTILS_SRC) $(UTILS_HDR) $(UTILS_CU_SRC) $(GPU_CUSTOM_PEREXPR_SRC) src/eval/common_eval.cpp $(EVOLVE_KERNEL_SRC) $(EVALUATOR_HDR)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -DUSE_GPU_CUSTOM_PEREXPR_MULTI -o $@ $(MAIN_SRC) $(UTILS_SRC) $(GPU_CUSTOM_PEREXPR_SRC) src/eval/common_eval.cpp $(UTILS_CU_SRC) $(EVOLVE_KERNEL_SRC) \
		-lnvrtc -lcuda

GPU_CUSTOM_PEREXPR_MULTI_NVRTC_BIN = $(BUILD_DIR)/gpu_custom_kernel_perexpression_multi_nvrtc

$(GPU_CUSTOM_PEREXPR_MULTI_NVRTC_BIN): $(MAIN_SRC) $(UTILS_SRC) $(UTILS_HDR) $(UTILS_CU_SRC) $(GPU_CUSTOM_PEREXPR_SRC) src/eval/common_eval.cpp $(EVOLVE_KERNEL_SRC) $(EVALUATOR_HDR)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -DUSE_GPU_CUSTOM_PEREXPR_MULTI_NVRTC -o $@ $(MAIN_SRC) $(UTILS_SRC) $(GPU_CUSTOM_PEREXPR_SRC) src/eval/common_eval.cpp $(UTILS_CU_SRC) $(EVOLVE_KERNEL_SRC) \
		-lnvrtc -lcuda

# Test with single expression (non-evolve custom per-expression)
run_gpu_custom_kernel_perexpression_single: $(GPU_CUSTOM_PEREXPR_MULTI_BIN)
	$(GPU_CUSTOM_PEREXPR_MULTI_BIN) data/ai_feyn/singles/input_001.txt

# Test with multiple expressions (non-evolve custom per-expression)
run_gpu_custom_kernel_perexpression_multi: $(GPU_CUSTOM_PEREXPR_MULTI_BIN)
	$(GPU_CUSTOM_PEREXPR_MULTI_BIN) data/ai_feyn/multi/input_100_100k.txt

# Test with sample input (non-evolve custom per-expression)
run_gpu_custom_kernel_perexpression_sample: $(GPU_CUSTOM_PEREXPR_MULTI_BIN)
	$(GPU_CUSTOM_PEREXPR_MULTI_BIN) data/examples/sample_input.txt

# Default non-evolve custom per-expression run target
run_gpu_custom_kernel_perexpression: run_gpu_custom_kernel_perexpression_sample

# Test with mutation file (100 mutations, shared data) using PTX multi-expression path
run_gpu_custom_kernel_perexpression_mutations_ptx: $(GPU_CUSTOM_PEREXPR_MULTI_BIN)
	$(GPU_CUSTOM_PEREXPR_MULTI_BIN) data/ai_feyn/mutations/input_base056_100mut_1000k.txt

# Convenience build target for PTX multi-expression batch evaluator
.PHONY: eval_multi_expr_ptx_batch
eval_multi_expr_ptx_batch: $(GPU_CUSTOM_PEREXPR_MULTI_BIN)

# Compare all three implementations
compare_evals: $(CPU_EVAL_BIN) $(GPU_EVAL_BIN) $(GPU_JINHA_BIN)
	@echo "=== CPU Simple ==="
	@time $(CPU_EVAL_BIN) data/examples/sample_input.txt
	@echo ""
	@echo "=== GPU Simple ==="
	@time $(GPU_EVAL_BIN) data/examples/sample_input.txt
	@echo ""
	@echo "=== GPU Jinha (eval_tree) ==="
	@time $(GPU_JINHA_BIN) data/examples/sample_input.txt

# ============================================================================
# GPU Async Jinha Evaluation (using async double-buffer from eval_tree.cu)
# ============================================================================
GPU_ASYNC_JINHA_BIN = $(BUILD_DIR)/gpu_async_jinha_eval
GPU_ASYNC_JINHA_SRC = src/eval/gpu_async_jinha.cu

$(GPU_ASYNC_JINHA_BIN): $(MAIN_SRC) $(UTILS_SRC) $(UTILS_HDR) $(UTILS_CU_SRC) $(GPU_ASYNC_JINHA_SRC) src/eval/common_eval.cpp $(EVALUATOR_HDR)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -DUSE_GPU_ASYNC_JINHA -o $@ $(MAIN_SRC) $(UTILS_SRC) $(GPU_ASYNC_JINHA_SRC) src/eval/common_eval.cpp $(UTILS_CU_SRC)

# Test with single expression
run_gpu_async_jinha_eval_single: $(GPU_ASYNC_JINHA_BIN)
	$(GPU_ASYNC_JINHA_BIN) data/ai_feyn/singles/input_001.txt

# Test with multiple expressions
run_gpu_async_jinha_eval_multi: $(GPU_ASYNC_JINHA_BIN)
	$(GPU_ASYNC_JINHA_BIN) data/ai_feyn/multi/input_100_100k.txt

# Test with sample input (2 expressions, 1000 data points each)
run_gpu_async_jinha_eval_sample: $(GPU_ASYNC_JINHA_BIN)
	$(GPU_ASYNC_JINHA_BIN) data/examples/sample_input.txt

# Default GPU Async Jinha run target
run_gpu_async_jinha_eval: run_gpu_async_jinha_eval_sample

# Compare all implementations
compare_all_evals: $(CPU_EVAL_BIN) $(CPU_MULTI_EVAL_BIN) $(GPU_EVAL_BIN) $(GPU_JINHA_BIN) $(GPU_ASYNC_JINHA_BIN)
	@echo "=== CPU Single-threaded ==="
	@time $(CPU_EVAL_BIN) data/examples/sample_input.txt
	@echo ""
	@echo "=== CPU Multi-threaded ($(CPU_EVAL_THREADS) workers) ==="
	@time $(CPU_MULTI_EVAL_BIN) data/examples/sample_input.txt
	@echo ""
	@echo "=== GPU Simple ==="
	@time $(GPU_EVAL_BIN) data/examples/sample_input.txt
	@echo ""
	@echo "=== GPU Jinha (batch) ==="
	@time $(GPU_JINHA_BIN) data/examples/sample_input.txt
	@echo ""
	@echo "=== GPU Async Jinha (double-buffer) ==="
	@time $(GPU_ASYNC_JINHA_BIN) data/examples/sample_input.txt

# ============================================================================
# GPU Baseline Evaluator (Two-Stage with Persistent Cache)
# ============================================================================
GPU_SIMPLE_JINHA_BIN = $(BUILD_DIR)/gpu_simple_jinha
GPU_SIMPLE_JINHA_SRC = src/eval/gpu_simple_jinha.cu

$(GPU_SIMPLE_JINHA_BIN): $(MAIN_SRC) $(UTILS_SRC) $(UTILS_HDR) $(GPU_SIMPLE_JINHA_SRC) $(EVALUATOR_HDR)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -DUSE_GPU_JINHA -o $@ \
		$(MAIN_SRC) $(UTILS_SRC) $(GPU_SIMPLE_JINHA_SRC) src/eval/common_eval.cpp

# Default GPU Subtree run target
gpu_simple_jinha: $(GPU_SIMPLE_JINHA_BIN)

# ============================================================================
# GPU Subtree Evaluator (Two-Stage with Persistent Cache)
# ============================================================================
GPU_SUBTREE_BIN = $(BUILD_DIR)/gpu_subtree_eval
GPU_SUBTREE_SRC = src/eval/gpu_subtree.cu

$(GPU_SUBTREE_BIN): $(MAIN_SRC) $(UTILS_SRC) $(UTILS_HDR) $(GPU_SUBTREE_SRC) $(EVALUATOR_HDR)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -DUSE_GPU_SUBTREE -o $@ \
		$(MAIN_SRC) $(UTILS_SRC) $(GPU_SUBTREE_SRC) src/eval/common_eval.cpp

# Default GPU Subtree run target
gpu_subtree_eval: $(GPU_SUBTREE_BIN)

# ============================================================================
# GPU Subtree Evaluator (Stateful, No Batching Across Datapoints)
# ============================================================================
GPU_SUBTREE_STATE_BIN = $(BUILD_DIR)/gpu_subtree_state_eval
GPU_SUBTREE_STATE_SRC = src/eval/gpu_subtree_batch_state.cu src/eval/gpu_subtree_batch_state_main.cpp

$(GPU_SUBTREE_STATE_BIN): $(GPU_SUBTREE_STATE_SRC) $(UTILS_SRC) $(UTILS_HDR)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $(GPU_SUBTREE_STATE_SRC) $(UTILS_SRC)

gpu_subtree_state_eval: $(GPU_SUBTREE_STATE_BIN)

# ============================================================================
# Data Generation
# ============================================================================

PREPROCESS_SCRIPT := src/script/preprocess_ai_feyn.py
MULTI_DIR := data/ai_feyn/multi
SINGLES_DIR := data/ai_feyn/singles
MUTATIONS_DIR := data/ai_feyn/mutations

.PHONY: data_gen data_gen_multi data_gen_singles data_gen_mutations

# Generate all data files
data_gen: data_gen_multi data_gen_singles data_gen_mutations data_gen_evo
	@echo ""
	@echo "=========================================="
	@echo "All data generation complete!"
	@echo "=========================================="

# Generate evolution benchmark data
data_gen_evo: gen_data_short gen_data_long
	@echo ""
	@echo "Evolution data generated."

# Generate multi-expression files with different datapoint counts
data_gen_multi:
	@echo "=========================================="
	@echo "Generating multi-expression files..."
	@echo "=========================================="
	@mkdir -p $(MULTI_DIR)
	@echo ""
	@echo "--- Generating 100_10k (10,000 datapoints) ---"
	python $(PREPROCESS_SCRIPT) --multi --dps 10000
	@echo ""
	@echo "--- Generating 100_100k (100,000 datapoints) ---"
	python $(PREPROCESS_SCRIPT) --multi --dps 100000
	@echo ""
	@echo "--- Generating 100_1000k (1,000,000 datapoints) ---"
	python $(PREPROCESS_SCRIPT) --multi --dps 1000000
	@echo ""
	@echo "Multi-expression files generated in $(MULTI_DIR)"

# Generate all single-expression files
data_gen_singles:
	@echo ""
	@echo "=========================================="
	@echo "Generating single-expression files..."
	@echo "=========================================="
	@mkdir -p $(SINGLES_DIR)
	python $(PREPROCESS_SCRIPT) --write --quiet
	@echo ""
	@echo "Single-expression files generated in $(SINGLES_DIR)"

# Generate mutation files (100 mutations of equation 56 with 100k datapoints)
data_gen_mutations:
	@echo ""
	@echo "=========================================="
	@echo "Generating mutation files..."
	@echo "=========================================="
	@mkdir -p $(MUTATIONS_DIR)
	python $(PREPROCESS_SCRIPT) --mutation
	python $(PREPROCESS_SCRIPT) --mutation --dps 1000000
	@echo ""
	@echo "Mutation files generated in $(MUTATIONS_DIR)"

# ============================================================================
# Evolution Tests
# ============================================================================
TEST_EVOLVE_BIN = evolution_tests
TEST_EVOLVE_SRC = src/test/evolution_tests.cpp
KERNEL_SRC = src/utils/generate.cu src/utils/mutation.cu src/utils/crossover.cu

$(TEST_EVOLVE_BIN): $(TEST_EVOLVE_SRC) $(KERNEL_SRC)
	$(NVCC) $(NVFLAGS) -I./src/utils -o $@ $(TEST_EVOLVE_SRC) $(KERNEL_SRC)

run_test_evolve: $(TEST_EVOLVE_BIN)
	./$(TEST_EVOLVE_BIN)

# ============================================================================
# Mutation Detection Tests
# ============================================================================
TEST_DETECT_BIN = $(BUILD_DIR)/test_detect_mutations
TEST_DETECT_SRC = src/test/test_detect_mutations.cpp

$(TEST_DETECT_BIN): $(TEST_DETECT_SRC) $(UTILS_SRC) $(UTILS_HDR)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I./src/utils -o $@ $(TEST_DETECT_SRC) $(UTILS_SRC)
# ============================================================================
# Evolution Benchmark Targets
# ============================================================================

# Data Generation
gen_data_long:
	python3 src/script/generate_evolution_data.py --gens 20 --dps 500000 --out data/evolution_long_large

gen_data_short:
	python3 src/script/generate_evolution_data.py --gens 5 --dps 100000 --out data/evolution_short_small

gen_data_test20:
	python3 src/script/generate_evolution_test20.py --gens 20 --pop 1000 --dps 500000 --out data/evolution_test20

bench_test20: cpu_common cpu_multi_eval
	@echo "--- Running Test20 Benchmark (Klein-Nishina, 20 gens, 1M dps) ---"
	@echo ">> Stateful Subtree"
	./build/cpu_common_subtree -evolution 0 19 data/evolution_test20
	@echo ">> CPU Multi"
	./build/cpu_multi_eval -evolution 0 19 data/evolution_test20

bench_test20_gpu: gpu_subtree_eval
	@echo "--- Running Test20 Benchmark (Klein-Nishina, 20 gens, 1M dps) on GPU ---"
	@echo ">> GPU Subtree"
	./build/gpu_subtree_eval -evolution 0 19 data/evolution_test20

bench_test20_gpu_state: gpu_subtree_state_eval
	@echo "--- Running Test20 Benchmark (Klein-Nishina, 20 gens, 1M dps) on GPU (Stateful) ---"
	@echo ">> GPU Subtree (Stateful)"
	./build/gpu_subtree_state_eval 0 19 data/evolution_test20

bench_test20_gpu_baseline: gpu_simple_jinha
	@echo "--- Running Test20 Benchmark (Klein-Nishina, 20 gens, 1M dps) on GPU (Baseline eval_tree_device) ---"
	@echo ">> GPU Jinha Baseline (Multi-Expr Batch)"
	./build/gpu_simple_jinha -evolution 0 19 data/evolution_test20

# Run Benchmarks (Stateful vs Stateless Multi-threaded)
bench_long: cpu_common cpu_multi_eval
	@echo "--- Running Long & Large Benchmark (20 gens, 500k dps) ---"
	@echo ">> Stateful Subtree"
	./build/cpu_common_subtree -evolution 0 19 data/evolution_long_large
	@echo ">> CPU Multi"
	./build/cpu_multi_eval -evolution 0 19 data/evolution_long_large

bench_long_gpu: gpu_subtree_eval
	@echo "--- Running Long & Large Benchmark (20 gens, 500k dps) on GPU ---"
	@echo ">> GPU Subtree"
	./build/gpu_subtree_eval -evolution 0 19 data/evolution_long_large

bench_long_gpu_state: gpu_subtree_state_eval
	@echo "--- Running Long & Large Benchmark (20 gens, 500k dps) on GPU (Stateful, no DP batching) ---"
	@echo ">> GPU Subtree (Stateful)"
	./build/gpu_subtree_state_eval 0 19 data/evolution_long_large

bench_short: cpu_common cpu_multi_eval
	@echo "--- Running Short & Small Benchmark (5 gens, 100k dps) ---"
	@echo ">> Stateful Subtree"
	./build/cpu_common_subtree -evolution 0 4 data/evolution_short_small
	@echo ">> CPU Multi"
	./build/cpu_multi_eval -evolution 0 4 data/evolution_short_small

bench_short_gpu: gpu_subtree_eval
	@echo "--- Running Short & Small Benchmark (5 gens, 100k dps) on GPU ---"
	@echo ">> GPU Subtree"
	./build/gpu_subtree_eval -evolution 0 4 data/evolution_short_small

bench_short_gpu_state: gpu_subtree_state_eval
	@echo "--- Running Short & Small Benchmark (5 gens, 100k dps) on GPU (Stateful, no DP batching) ---"
	@echo ">> GPU Subtree (Stateful)"
	./build/gpu_subtree_state_eval 0 4 data/evolution_short_small

# ============================================================================
# GPU Jinha Evolution Benchmark
# ============================================================================
GPU_JINHA_EVOLVE_BIN = $(BUILD_DIR)/gpu_jinha_evolve
GPU_JINHA_EVOLVE_SRC = src/eval/gpu_simple_jinha.cu

$(GPU_JINHA_EVOLVE_BIN): $(MAIN_SRC) $(UTILS_SRC) $(UTILS_HDR) $(UTILS_CU_SRC) $(GPU_JINHA_EVOLVE_SRC) $(EVALUATOR_HDR)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -DUSE_GPU_JINHA_MULTI_EXPRESSION_BATCH -o $@ \
		$(MAIN_SRC) $(UTILS_SRC) $(GPU_JINHA_EVOLVE_SRC) $(UTILS_CU_SRC) src/eval/common_eval.cpp

gpu_jinha_evolve: $(GPU_JINHA_EVOLVE_BIN)

bench_short_gpu_jinha: $(GPU_JINHA_EVOLVE_BIN)
	@echo "--- Running Short & Small Benchmark (5 gens, 100k dps) on GPU (Jinha) ---"
	@echo ">> GPU Jinha (Multi-Expr Batch)"
	./build/gpu_jinha_evolve -evolution 0 4 data/evolution_short_small

bench_long_gpu_jinha: $(GPU_JINHA_EVOLVE_BIN)
	@echo "--- Running Long & Large Benchmark (20 gens, 500k dps) on GPU (Jinha) ---"
	@echo ">> GPU Jinha (Multi-Expr Batch)"
	./build/gpu_jinha_evolve -evolution 0 19 data/evolution_long_large

bench_all_evolution: gen_data_short gen_data_long bench_short bench_long
run_test_detect: $(TEST_DETECT_BIN)
	./$(TEST_DETECT_BIN) data/ai_feyn/mutations/input_base056_100mut_1000k.txt 3 2

clean:
	rm -f $(TARGET) run_gpu $(TEST_BIN) $(BENCH_BIN) $(TEST_EVOLVE_BIN) $(TEST_DETECT_BIN)
	rm -rf $(BUILD_DIR)

.PHONY: all test run_bench run_test_evolve
