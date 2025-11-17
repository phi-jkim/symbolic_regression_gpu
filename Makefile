# Use system CUDA instead of conda/miniforge CUDA to avoid incomplete installations
# NVCC ?= /usr/local/cuda/bin/nvcc
NVCC = nvcc
# Target NVIDIA L40S (Ada, SM 89). Embed PTX for forward-compat JIT.
# Use GCC 12 for NVCC to avoid compatibility issues with GCC 13+
NVFLAGS ?= -O3 \
  -gencode arch=compute_89,code=sm_89 \
  -gencode arch=compute_89,code=compute_89 \
  -Wno-deprecated-gpu-targets
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
# CPU Evaluation Targets (for Feynman equation evaluation)
# ============================================================================
CXX = g++
CXXFLAGS = -std=c++11 -O3 -Wall -pthread

BUILD_DIR = build
CPU_EVAL_BIN = $(BUILD_DIR)/cpu_eval
UTILS_SRC = src/utils/utils.cpp
MAIN_SRC = src/main.cpp
CPU_EVAL_SRC = src/eval/cpu_simple_single.cpp

$(CPU_EVAL_BIN): $(MAIN_SRC) $(UTILS_SRC) $(CPU_EVAL_SRC)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -DUSE_CPU_SIMPLE -o $@ $(MAIN_SRC) $(UTILS_SRC) $(CPU_EVAL_SRC)

# Test with single expression
run_cpu_eval_single: $(CPU_EVAL_BIN)
	$(CPU_EVAL_BIN) data/ai_feyn/singles/input_001.txt

# Test with multiple expressions
run_cpu_eval_multi: $(CPU_EVAL_BIN)
	$(CPU_EVAL_BIN) data/ai_feyn/multi/input_100_100k.txt

# Test with sample input (2 expressions, 1000 data points each)
run_cpu_eval_sample: $(CPU_EVAL_BIN)
	$(CPU_EVAL_BIN) data/examples/sample_input.txt

# Default run target (single expression)
run_cpu_eval: run_cpu_eval_single

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
# Use GCC 12 for NVCC to avoid compatibility issues with GCC 13+
# Suppress deprecated GPU target warnings
NVCCFLAGS = -std=c++11 -O3 -arch=$(GPU_ARCH) \
	-Wno-deprecated-gpu-targets

$(GPU_EVAL_BIN): $(MAIN_SRC) $(UTILS_SRC) $(GPU_EVAL_SRC)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -DUSE_GPU_SIMPLE -o $@ $(MAIN_SRC) $(UTILS_SRC) $(GPU_EVAL_SRC)

# Test with single expression
run_gpu_eval_single: $(GPU_EVAL_BIN)
	$(GPU_EVAL_BIN) data/ai_feyn/singles/input_001.txt

# Test with multiple expressions
run_gpu_eval_multi: $(GPU_EVAL_BIN)
	$(GPU_EVAL_BIN) data/ai_feyn/multi/input_100_100k.txt

# Test with sample input (2 expressions, 1000 data points each)
run_gpu_eval_sample: $(GPU_EVAL_BIN)
	$(GPU_EVAL_BIN) data/examples/sample_input.txt

# Default GPU run target (multi expression)
run_gpu_eval: run_gpu_eval_multi

# ============================================================================
# GPU Jinha Evaluation (using eval_tree.cu library)
# ============================================================================
GPU_JINHA_BIN = $(BUILD_DIR)/gpu_jinha_eval
GPU_JINHA_SRC = src/eval/gpu_simple_jinha.cu
UTILS_CU_SRC = src/utils/utils.cu

$(GPU_JINHA_BIN): $(MAIN_SRC) $(UTILS_SRC) $(UTILS_CU_SRC) $(GPU_JINHA_SRC)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -DUSE_GPU_JINHA -o $@ \
		$(MAIN_SRC) $(UTILS_SRC) $(GPU_JINHA_SRC) $(UTILS_CU_SRC)

# Test with single expression
run_gpu_jinha_eval_single: $(GPU_JINHA_BIN)
	$(GPU_JINHA_BIN) data/ai_feyn/singles/input_001.txt

# Test with multiple expressions
run_gpu_jinha_eval_multi: $(GPU_JINHA_BIN)
	$(GPU_JINHA_BIN) data/ai_feyn/multi/input_100_100k.txt

# Test with sample input (2 expressions, 1000 data points each)
run_gpu_jinha_eval_sample: $(GPU_JINHA_BIN)
	$(GPU_JINHA_BIN) data/examples/sample_input.txt

# Default GPU Jinha run target
run_gpu_jinha_eval: run_gpu_jinha_eval_sample

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

$(GPU_ASYNC_JINHA_BIN): $(MAIN_SRC) $(UTILS_SRC) $(UTILS_CU_SRC) $(GPU_ASYNC_JINHA_SRC)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -DUSE_GPU_ASYNC_JINHA -o $@ \
		$(MAIN_SRC) $(UTILS_SRC) $(GPU_ASYNC_JINHA_SRC) $(UTILS_CU_SRC)

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

# Compare all four implementations
compare_all_evals: $(CPU_EVAL_BIN) $(GPU_EVAL_BIN) $(GPU_JINHA_BIN) $(GPU_ASYNC_JINHA_BIN)
	@echo "=== CPU Simple ==="
	@time $(CPU_EVAL_BIN) data/examples/sample_input.txt
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
# Data Generation
# ============================================================================

PREPROCESS_SCRIPT := src/script/preprocess_ai_feyn.py
MULTI_DIR := data/ai_feyn/multi
SINGLES_DIR := data/ai_feyn/singles

.PHONY: data_gen data_gen_multi data_gen_singles

# Generate all data files
data_gen: data_gen_multi data_gen_singles
	@echo ""
	@echo "=========================================="
	@echo "All data generation complete!"
	@echo "=========================================="

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

# ============================================================================

clean:
	rm -f $(TARGET) run_gpu $(TEST_BIN) $(BENCH_BIN)
	rm -rf $(BUILD_DIR)
