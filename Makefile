NVCC ?= nvcc
# Target NVIDIA L40S (Ada, SM 89). Embed PTX for forward-compat JIT.
NVFLAGS ?= -O3 \
  -gencode arch=compute_89,code=sm_89 \
  -gencode arch=compute_89,code=compute_89
TARGET := libevaltree.so
SRC := eval_tree.cu

all: $(TARGET) run_gpu run_evogp

$(TARGET): $(SRC)
	$(NVCC) $(NVFLAGS) -Xcompiler -fPIC -shared -o $@ $<

run_gpu: eval_tree.cu run_gpu.cu
	$(NVCC) $(NVFLAGS) -o $@ run_gpu.cu eval_tree.cu

# Benchmark evogp's treeGPEvalKernel via evogp::evaluate on the same dataset
EVOGP_FWD := ../evogp/src/evogp/cuda/forward.cu
EVOGP_INC := -I../evogp/src/evogp/cuda
run_evogp: run_evogp.cu eval_tree.cu $(EVOGP_FWD)
	$(NVCC) $(NVFLAGS) $(EVOGP_INC) -o $@ run_evogp.cu eval_tree.cu $(EVOGP_FWD)

clean:
	rm -f $(TARGET) run_gpu run_evogp
