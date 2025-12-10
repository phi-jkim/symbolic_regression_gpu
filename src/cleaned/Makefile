NVCC = nvcc
CXX = g++
NVCC_FLAGS = -O3 -arch=sm_89 -std=c++14 --compiler-options -Wall
CXX_FLAGS = -O3 -std=c++14 -Wall -pthread

INCLUDES = -I../utils

BUILD_DIR = ../../build
# Ensure build dir exists
$(shell mkdir -p $(BUILD_DIR))

# Objects
OBJS = $(BUILD_DIR)/cl_main.o $(BUILD_DIR)/cl_gpu_eval.o $(BUILD_DIR)/cl_gpu_simple.o $(BUILD_DIR)/cl_cpu_eval.o $(BUILD_DIR)/detect.o $(BUILD_DIR)/utils.o $(BUILD_DIR)/cl_gpu_ptx.o $(BUILD_DIR)/cl_gpu_nvrtc.o

# Targets
all: $(BUILD_DIR)/cl_bench $(BUILD_DIR)/cl_sr_opt $(BUILD_DIR)/cl_sr_simple $(BUILD_DIR)/cl_sr_cpu $(BUILD_DIR)/cl_sr_ptx $(BUILD_DIR)/cl_sr_nvrtc

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
SEED ?=

.PHONY: run_sr_opt run_sr_simple run_sr_cpu

run_sr_opt: $(BUILD_DIR)/cl_sr_opt
	THREADS=$(THREADS) srun --gres=gpu:1 $(BUILD_DIR)/cl_sr_opt $(POP) $(GENS) $(DATA) $(LIMIT_DPS) $(SEED)

run_sr_simple: $(BUILD_DIR)/cl_sr_simple
	THREADS=$(THREADS) srun --gres=gpu:1 $(BUILD_DIR)/cl_sr_simple $(POP) $(GENS) $(DATA) $(LIMIT_DPS) $(SEED)

run_sr_cpu: $(BUILD_DIR)/cl_sr_cpu
	THREADS=$(THREADS) $(BUILD_DIR)/cl_sr_cpu $(POP) $(GENS) $(DATA) $(LIMIT_DPS) $(SEED)

# GPU PTX System
$(BUILD_DIR)/cl_sr_ptx: $(BUILD_DIR)/cl_sr_system_ptx.o $(BUILD_DIR)/cl_gpu_eval.o $(BUILD_DIR)/cl_gpu_simple.o $(BUILD_DIR)/cl_gpu_ptx.o $(BUILD_DIR)/cl_cpu_eval.o $(BUILD_DIR)/detect.o $(BUILD_DIR)/utils.o
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ -lpthread -lnvrtc -lcuda

$(BUILD_DIR)/cl_sr_system_ptx.o: cl_sr_system.cpp
	$(CXX) $(CXX_FLAGS) -DUSE_GPU_PTX $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/cl_gpu_ptx.o: cl_gpu_ptx.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

run_sr_ptx: $(BUILD_DIR)/cl_sr_ptx
	THREADS=$(THREADS) srun --gres=gpu:1 $(BUILD_DIR)/cl_sr_ptx $(POP) $(GENS) $(DATA) $(LIMIT_DPS) $(SEED)

# GPU NVRTC System
$(BUILD_DIR)/cl_sr_nvrtc: $(BUILD_DIR)/cl_sr_system_nvrtc.o $(BUILD_DIR)/cl_gpu_eval.o $(BUILD_DIR)/cl_gpu_simple.o $(BUILD_DIR)/cl_gpu_ptx.o $(BUILD_DIR)/cl_gpu_nvrtc.o $(BUILD_DIR)/cl_cpu_eval.o $(BUILD_DIR)/detect.o $(BUILD_DIR)/utils.o
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ -lpthread -lnvrtc -lcuda

$(BUILD_DIR)/cl_sr_system_nvrtc.o: cl_sr_system.cpp
	$(CXX) $(CXX_FLAGS) -DUSE_GPU_NVRTC $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/cl_gpu_nvrtc.o: cl_gpu_nvrtc.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

run_sr_nvrtc: $(BUILD_DIR)/cl_sr_nvrtc
	THREADS=$(THREADS) srun --gres=gpu:1 $(BUILD_DIR)/cl_sr_nvrtc $(POP) $(GENS) $(DATA) $(LIMIT_DPS) $(SEED)

run_sr_task_gen_pop100_gpu:
	@for i in 1 2 3 4 5; do \
		seed=$$RANDOM; \
		echo "Run $$i/5: GPU Optimized (Seed $$seed)"; \
		$(MAKE) run_sr_opt POP=100 SEED=$$seed; \
		echo "Run $$i/5: GPU Simple (Seed $$seed)"; \
		$(MAKE) run_sr_simple POP=100 SEED=$$seed; \
		echo "Run $$i/5: GPU PTX (Seed $$seed)"; \
		$(MAKE) run_sr_ptx GEN=50 POP=100 SEED=$$seed; \
		echo "Run $$i/5: GPU NVRTC (Seed $$seed)"; \
		$(MAKE) run_sr_nvrtc GEN=50 POP=100 SEED=$$seed; \
	done

run_sr_task_gen_pop2000_gpu:
	@for i in 1 2 3 4 5; do \
		seed=$$RANDOM; \
		echo "Run $$i/5: GPU Optimized (Seed $$seed)"; \
		$(MAKE) run_sr_opt POP=2000 SEED=$$seed; \
		echo "Run $$i/5: GPU Simple (Seed $$seed)"; \
		$(MAKE) run_sr_simple POP=2000 SEED=$$seed; \
	done

run_sr_task_gen_pop100_cpu:
	@for i in 1 2 3 4 5; do \
		seed=$$RANDOM; \
		echo "Run $$i/5: CPU Simple (Seed $$seed)"; \
		$(MAKE) run_sr_cpu POP=100 THREADS=96 SEED=$$seed; \
	done

run_sr_task_gen_pop2000_cpu:
	@for i in 1 2 3; do \
		seed=$$RANDOM; \
		echo "Run $$i/5: CPU Simple (Seed $$seed)"; \
		$(MAKE) run_sr_cpu POP=2000 THREADS=96 SEED=$$seed; \
	done



# Benchmark Task: Scale Population GPU
# Pops: 50, 100, 200, 400, 800, 1600, then +800 up to 8000
# Runs: 3 times each
run_sr_task_scale_pop_gpu:
	@for pop in 50 100 200; do \
		echo "Scale Pop Test: POP=$$pop"; \
		for i in 1 2 3 4 5; do \
			seed=$$RANDOM; \
			echo "  Run $$i/5: GPU Optimized (Pop $$pop, Seed $$seed)"; \
			$(MAKE) run_sr_opt POP=$$pop GENS=30 SEED=$$seed; \
			echo "  Run $$i/5: GPU Simple (Pop $$pop, Seed $$seed)"; \
			$(MAKE) run_sr_simple POP=$$pop GENS=30 SEED=$$seed; \
			echo "  Run $$i/5: GPU PTX (Pop $$pop, Seed $$seed)"; \
			$(MAKE) run_sr_ptx POP=$$pop GENS=30 SEED=$$seed; \
		done; \
	done
	@for pop in 400 800 1600 2400 3200 4800 6400 8000; do \
		echo "Scale Pop Test: POP=$$pop"; \
		for i in 1 2 3 4 5; do \
			seed=$$RANDOM; \
			echo "  Run $$i/5: GPU Optimized (Pop $$pop, Seed $$seed)"; \
			$(MAKE) run_sr_opt POP=$$pop GENS=30 SEED=$$seed; \
			echo "  Run $$i/5: GPU Simple (Pop $$pop, Seed $$seed)"; \
			$(MAKE) run_sr_simple POP=$$pop GENS=30 SEED=$$seed; \
		done; \
	done

run_sr_task_scale_pop_cpu:
	@for pop in 50 100 200 400; do \
		echo "Scale Pop Test: POP=$$pop"; \
		for i in 1 2 3 4 5; do \
			seed=$$RANDOM; \
			echo "  Run $$i/5: CPU Simple (Pop $$pop, Seed $$seed)"; \
			$(MAKE) run_sr_cpu POP=$$pop GENS=50 THREADS=96 SEED=$$seed; \
		done; \
	done


# Benchmark Task: Scale DPS GPU
# DPS: 50k, 100k, 200k, 500k, 1M
# Runs: 3 times each
# Fixed Pop=2000, Gens=100
run_sr_task_scale_dps_gpu:
	@for dps in 40000 80000 160000 320000 480000 640000 800000 1000000; do \
		echo "Scale DPS Test: DPS=$$dps"; \
		for i in 1 2 3 4 5; do \
			seed=$$RANDOM; \
			echo "  Run $$i/5: GPU Optimized (DPS $$dps, Seed $$seed)"; \
			$(MAKE) run_sr_opt POP=2000 GENS=50 LIMIT_DPS=$$dps SEED=$$seed; \
			echo "  Run $$i/5: GPU Simple (DPS $$dps, Seed $$seed)"; \
			$(MAKE) run_sr_simple POP=2000 GENS=50 LIMIT_DPS=$$dps SEED=$$seed; \
		done; \
	done



# Data Generation
gen_data_test20_1M:
	python3 ../script/generate_evolution_test20.py --out ../../data/evolution_test20_1M --dps 1000000 --gens 1 --pop 2000

clean:
	rm -f $(OBJS) $(BUILD_DIR)/cl_bench $(BUILD_DIR)/cl_sr_*

# Plotting Targets
plots:
	python3 ../utils/plot_results.py --dir ../../data/output/sr_results/plot-A1
	python3 ../utils/plot_results.py --dir ../../data/output/sr_results/plot-A2 --filter GPU
	python3 ../utils/plot_results_pop.py --dir ../../data/output/sr_results/plot-B --filter GPU
	python3 ../utils/plot_results_dps.py --dir ../../data/output/sr_results/plot-C
	python3 ../utils/plot_results_kernel.py --dir ../../data/output/sr_results/plot-A1 --filter GPU
	python3 ../utils/plot_results_breakdown.py --dir ../../data/output/sr_results/plot-A1
	python3 ../utils/plot_results_breakdown_pop.py --dir ../../data/output/sr_results/plot-B --filter GPU
