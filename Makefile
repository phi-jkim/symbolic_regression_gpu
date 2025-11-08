NVCC ?= nvcc
# Target NVIDIA L40S (Ada, SM 89). Embed PTX for forward-compat JIT.
NVFLAGS ?= -O3 \
  -gencode arch=compute_89,code=sm_89 \
  -gencode arch=compute_89,code=compute_89
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

clean:
	rm -f $(TARGET) run_gpu $(TEST_BIN) $(BENCH_BIN)
