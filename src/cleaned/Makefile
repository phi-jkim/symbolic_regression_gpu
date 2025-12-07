NVCC = nvcc
CXX = g++
NVCC_FLAGS = -O3 -arch=sm_70 -std=c++14 --compiler-options -Wall
CXX_FLAGS = -O3 -std=c++14 -Wall -fopenmp

INCLUDES = -I../utils

BUILD_DIR = ../../build
# Ensure build dir exists
$(shell mkdir -p $(BUILD_DIR))

# Objects
OBJS = $(BUILD_DIR)/cl_main.o $(BUILD_DIR)/cl_gpu_eval.o $(BUILD_DIR)/cl_gpu_simple.o $(BUILD_DIR)/cl_cpu_eval.o $(BUILD_DIR)/detect.o $(BUILD_DIR)/utils.o

# Targets
all: $(BUILD_DIR)/cl_bench

$(BUILD_DIR)/cl_bench: $(OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ -lgomp

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

clean:
	rm -f $(OBJS) $(BUILD_DIR)/cl_bench
