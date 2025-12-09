NVCC = nvcc
CXX = g++
NVCC_FLAGS = -O3 -arch=sm_89 -std=c++14 --compiler-options -Wall
CXX_FLAGS = -O3 -std=c++14 -Wall -fopenmp

INCLUDES = -I../utils

BUILD_DIR = ../../build
# Ensure build dir exists
$(shell mkdir -p $(BUILD_DIR))

# Objects
OBJS = $(BUILD_DIR)/cl_main.o $(BUILD_DIR)/cl_gpu_eval.o $(BUILD_DIR)/cl_gpu_simple.o $(BUILD_DIR)/cl_cpu_eval.o $(BUILD_DIR)/detect.o $(BUILD_DIR)/utils.o

# Targets
all: $(BUILD_DIR)/cl_bench $(BUILD_DIR)/cl_sr_opt $(BUILD_DIR)/cl_sr_simple $(BUILD_DIR)/cl_sr_cpu

$(BUILD_DIR)/cl_bench: $(OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ -lgomp

# GPU Optimized System
$(BUILD_DIR)/cl_sr_opt: $(BUILD_DIR)/cl_sr_system_opt.o $(BUILD_DIR)/cl_gpu_eval.o $(BUILD_DIR)/cl_gpu_simple.o $(BUILD_DIR)/cl_cpu_eval.o $(BUILD_DIR)/detect.o $(BUILD_DIR)/utils.o
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ -lgomp

$(BUILD_DIR)/cl_sr_system_opt.o: cl_sr_system.cpp
	$(CXX) $(CXX_FLAGS) -DUSE_GPU_SUBTREE $(INCLUDES) -c $< -o $@

# GPU Simple System
$(BUILD_DIR)/cl_sr_simple: $(BUILD_DIR)/cl_sr_system_simple.o $(BUILD_DIR)/cl_gpu_eval.o $(BUILD_DIR)/cl_gpu_simple.o $(BUILD_DIR)/cl_cpu_eval.o $(BUILD_DIR)/detect.o $(BUILD_DIR)/utils.o
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ -lgomp

$(BUILD_DIR)/cl_sr_system_simple.o: cl_sr_system.cpp
	$(CXX) $(CXX_FLAGS) -DUSE_GPU_SIMPLE $(INCLUDES) -c $< -o $@

# CPU System
$(BUILD_DIR)/cl_sr_cpu: $(BUILD_DIR)/cl_sr_system_cpu.o $(BUILD_DIR)/cl_gpu_eval.o $(BUILD_DIR)/cl_gpu_simple.o $(BUILD_DIR)/cl_cpu_eval.o $(BUILD_DIR)/detect.o $(BUILD_DIR)/utils.o
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ -lgomp

$(BUILD_DIR)/cl_sr_system_cpu.o: cl_sr_system.cpp
	$(CXX) $(CXX_FLAGS) -DUSE_CPU $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/cl_main.o: cl_main.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/cl_cpu_eval.o: cl_cpu_eval.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/cl_gpu_eval.o: cl_gpu_eval.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/cl_gpu_simple.o: cl_gpu_simple.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/detect.o: ../utils/detect.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/utils.o: ../utils/utils.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $< -o $@

# Convenience Targets (SR System)
# Usage: make run_sr_opt POP=2000 GENS=100 THREADS=16
# Usage: make run_sr_simple POP=2000 GENS=100 THREADS=16
# Usage: make run_sr_cpu POP=2000 GENS=100 THREADS=16

POP ?= 2000
GENS ?= 100
DATA ?= ../../data/evolution_test20
THREADS ?= 8

.PHONY: run_sr_opt run_sr_simple run_sr_cpu

run_sr_opt: $(BUILD_DIR)/cl_sr_opt
	OMP_NUM_THREADS=$(THREADS) srun --gres=gpu:1 $(BUILD_DIR)/cl_sr_opt $(POP) $(GENS) $(DATA)

run_sr_simple: $(BUILD_DIR)/cl_sr_simple
	OMP_NUM_THREADS=$(THREADS) srun --gres=gpu:1 $(BUILD_DIR)/cl_sr_simple $(POP) $(GENS) $(DATA)

run_sr_cpu: $(BUILD_DIR)/cl_sr_cpu
	OMP_NUM_THREADS=$(THREADS) $(BUILD_DIR)/cl_sr_cpu $(POP) $(GENS) $(DATA)

clean:
	rm -f $(OBJS) $(BUILD_DIR)/cl_bench $(BUILD_DIR)/cl_sr_*
