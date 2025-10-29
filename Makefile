NVCC ?= nvcc
NVFLAGS ?= -O3
TARGET := libevaltree.so
SRC := eval_tree.cu

all: $(TARGET) run_gpu

$(TARGET): $(SRC)
	$(NVCC) $(NVFLAGS) -Xcompiler -fPIC -shared -o $@ $<

run_gpu: eval_tree.cu run_gpu.cu
	$(NVCC) $(NVFLAGS) -o $@ run_gpu.cu eval_tree.cu

clean:
	rm -f $(TARGET) run_gpu
