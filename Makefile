NVCC ?= nvcc
CXXFLAGS ?= -O3 -Xcompiler -fPIC
LDFLAGS ?= -shared
TARGET := libevaltree.so
SRC := eval_tree.cu

all: $(TARGET) run_gpu

$(TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) $(LDFLAGS) -o $@ $<

run_gpu: eval_tree.cu run_gpu.cu
	$(NVCC) -O3 -o $@ run_gpu.cu eval_tree.cu

clean:
	rm -f $(TARGET) run_gpu
