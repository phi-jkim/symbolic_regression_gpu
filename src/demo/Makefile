# GPU Timing Demo Makefile
# NOTE: Requires NVIDIA GPU and CUDA toolkit

NVCC = nvcc
NVCCFLAGS = -std=c++11 -O3 -arch=sm_70
# Adjust -arch based on your GPU:
# - sm_60: Pascal (GTX 10 series, Tesla P100)
# - sm_70: Volta (Tesla V100)
# - sm_75: Turing (RTX 20 series, Tesla T4)
# - sm_80: Ampere (RTX 30 series, A100)
# - sm_86: RTX 3090
# - sm_89: RTX 4090

TARGETS = gpu_timing_demo minimal_example

all: $(TARGETS)

gpu_timing_demo: gpu_timing_demo.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

minimal_example: minimal_example.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

run: gpu_timing_demo
	./gpu_timing_demo

run-minimal: minimal_example
	./minimal_example

clean:
	rm -f $(TARGETS)

# Check CUDA installation
check:
	@echo "Checking CUDA installation..."
	@which nvcc || echo "ERROR: nvcc not found. Install CUDA toolkit."
	@nvidia-smi || echo "ERROR: nvidia-smi not found. Check GPU driver."

# Query GPU properties
gpu-info:
	nvidia-smi

.PHONY: all run clean check gpu-info
