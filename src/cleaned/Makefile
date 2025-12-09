NVCC = nvcc
CXX = g++
NVCC_FLAGS = -O3 -arch=sm_89 -std=c++14 --compiler-options -Wall
CXX_FLAGS = -O3 -std=c++14 -Wall -pthread

INCLUDES = -I../utils

BUILD_DIR = ../../build
# Ensure build dir exists
$(shell mkdir -p $(BUILD_DIR))

# Objects
OBJS = $(BUILD_DIR)/cl_main.o $(BUILD_DIR)/cl_gpu_eval.o $(BUILD_DIR)/cl_gpu_simple.o $(BUILD_DIR)/cl_cpu_eval.o $(BUILD_DIR)/detect.o $(BUILD_DIR)/utils.o $(BUILD_DIR)/cl_gpu_ptx.o

# Targets
all: $(BUILD_DIR)/cl_bench $(BUILD_DIR)/cl_sr_opt $(BUILD_DIR)/cl_sr_simple $(BUILD_DIR)/cl_sr_cpu $(BUILD_DIR)/cl_sr_ptx

$(BUILD_DIR)/cl_bench: $(OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ -lpthread -lnvrtc -lcuda

# GPU Optimized System
$(BUILD_DIR)/cl_sr_opt: $(BUILD_DIR)/cl_sr_system_opt.o $(BUILD_DIR)/cl_gpu_eval.o $(BUILD_DIR)/cl_gpu_simple.o $(BUILD_DIR)/cl_cpu_eval.o $(BUILD_DIR)/detect.o $(BUILD_DIR)/utils.o
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ -lpthread

$(BUILD_DIR)/cl_sr_system_opt.o: cl_sr_system.cpp
	$(CXX) $(CXX_FLAGS) -DUSE_GPU_SUBTREE $(INCLUDES) -c $< -o $@

# GPU Simple System
$(BUILD_DIR)/cl_sr_simple: $(BUILD_DIR)/cl_sr_system_simple.o $(BUILD_DIR)/cl_gpu_eval.o $(BUILD_DIR)/cl_gpu_simple.o $(BUILD_DIR)/cl_cpu_eval.o $(BUILD_DIR)/detect.o $(BUILD_DIR)/utils.o
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ -lpthread

$(BUILD_DIR)/cl_sr_system_simple.o: cl_sr_system.cpp
	$(CXX) $(CXX_FLAGS) -DUSE_GPU_SIMPLE $(INCLUDES) -c $< -o $@

# CPU System
$(BUILD_DIR)/cl_sr_cpu: $(BUILD_DIR)/cl_sr_system_cpu.o $(BUILD_DIR)/cl_cpu_eval.o $(BUILD_DIR)/cl_cpu_eval.o $(BUILD_DIR)/detect.o $(BUILD_DIR)/utils.o
	$(CXX) $(CXX_FLAGS) -o $@ $^ -lpthread

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
DATA ?= ../../data/evolution_test20_1M
LIMIT_DPS ?= 1000000
THREADS ?= 8

.PHONY: run_sr_opt run_sr_simple run_sr_cpu

run_sr_opt: $(BUILD_DIR)/cl_sr_opt
	THREADS=$(THREADS) srun --gres=gpu:1 $(BUILD_DIR)/cl_sr_opt $(POP) $(GENS) $(DATA) $(LIMIT_DPS)

run_sr_simple: $(BUILD_DIR)/cl_sr_simple
	THREADS=$(THREADS) srun --gres=gpu:1 $(BUILD_DIR)/cl_sr_simple $(POP) $(GENS) $(DATA) $(LIMIT_DPS)

run_sr_cpu: $(BUILD_DIR)/cl_sr_cpu
	THREADS=$(THREADS) $(BUILD_DIR)/cl_sr_cpu $(POP) $(GENS) $(DATA) $(LIMIT_DPS)

# GPU PTX System
$(BUILD_DIR)/cl_sr_ptx: $(BUILD_DIR)/cl_sr_system_ptx.o $(BUILD_DIR)/cl_gpu_eval.o $(BUILD_DIR)/cl_gpu_simple.o $(BUILD_DIR)/cl_gpu_ptx.o $(BUILD_DIR)/cl_cpu_eval.o $(BUILD_DIR)/detect.o $(BUILD_DIR)/utils.o
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ -lpthread -lnvrtc -lcuda

$(BUILD_DIR)/cl_sr_system_ptx.o: cl_sr_system.cpp
	$(CXX) $(CXX_FLAGS) -DUSE_GPU_PTX $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/cl_gpu_ptx.o: cl_gpu_ptx.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

run_sr_ptx: $(BUILD_DIR)/cl_sr_ptx
	THREADS=$(THREADS) srun --gres=gpu:1 $(BUILD_DIR)/cl_sr_ptx $(POP) $(GENS) $(DATA) $(LIMIT_DPS)

run_sr_task_gen_pop100_gpu:
	@for i in 1 2 3 4 5; do \
		echo "Run $$i/3: GPU Optimized"; \
		$(MAKE) run_sr_opt POP=100; \
		echo "Run $$i/3: GPU Simple"; \
		$(MAKE) run_sr_simple POP=100; \
		echo "Run $$i/3: GPU PTX"; \
		$(MAKE) run_sr_ptx POP=100; \
	done

run_sr_task_gen_pop2000_gpu:
	@for i in 1 2 3 4 5; do \
		echo "Run $$i/3: GPU Optimized"; \
		$(MAKE) run_sr_opt POP=2000; \
		echo "Run $$i/3: GPU Simple"; \
		$(MAKE) run_sr_simple POP=2000; \
# 		echo "Run $$i/3: GPU PTX"; \
# 		$(MAKE) run_sr_ptx POP=2000; \
	done

run_sr_task_gen_pop100_cpu:
	@for i in 1 2 3 4 5; do \
		echo "Run $$i/3: CPU Simple"; \
		$(MAKE) run_sr_cpu POP=100 THREADS=96; \
	done

run_sr_task_gen_pop2000_cpu:
	@for i in 1 2 3 4 5; do \
		echo "Run $$i/3: CPU Simple"; \
		$(MAKE) run_sr_cpu POP=2000 THREADS=96; \
	done



# Benchmark Task: Scale Population GPU
# Pops: 50, 100, 200, 400, 800, 1600, then +800 up to 8000
# Runs: 3 times each
run_sr_task_scale_pop_gpu:
	@for pop in 50 100 200 400 800 1600 2400 3200 4800 6400 8000; do \
		echo "Scale Pop Test: POP=$$pop"; \
		for i in 1 2 3 4 5; do \
			echo "  Run $$i/3: GPU Optimized (Pop $$pop)"; \
			$(MAKE) run_sr_opt POP=$$pop GENS=64; \
			echo "  Run $$i/3: GPU Simple (Pop $$pop)"; \
			$(MAKE) run_sr_simple POP=$$pop GENS=64; \
			# echo "  Run $$i/3: GPU PTX (Pop $$pop)"; \
			# $(MAKE) run_sr_ptx POP=$$pop GENS=64; \
		done; \
	done


# Benchmark Task: Scale DPS GPU
# DPS: 50k, 100k, 200k, 500k, 1M
# Runs: 3 times each
# Fixed Pop=2000, Gens=100
run_sr_task_scale_dps_gpu:
	@for dps in 50000 100000 200000 500000 1000000; do \
		echo "Scale DPS Test: DPS=$$dps"; \
		for i in 1 2 3 4 5; do \
			echo "  Run $$i/3: GPU Optimized (DPS $$dps)"; \
			$(MAKE) run_sr_opt POP=2000 GENS=100 LIMIT_DPS=$$dps; \
			echo "  Run $$i/3: GPU Simple (DPS $$dps)"; \
			$(MAKE) run_sr_simple POP=2000 GENS=100 LIMIT_DPS=$$dps; \
		done; \
	done



# Data Generation
gen_data_test20_1M:
	python3 ../script/generate_evolution_test20.py --out ../../data/evolution_test20_1M --dps 1000000 --gens 200 --pop 2000

clean:
	rm -f $(OBJS) $(BUILD_DIR)/cl_bench $(BUILD_DIR)/cl_sr_*
