#!/bin/bash
# check_cuda.sh - Validate CUDA environment and GPU availability
#
# Usage: ./check_cuda.sh [--verbose]

set -e

VERBOSE=0
if [[ "$1" == "--verbose" ]]; then
    VERBOSE=1
fi

echo "=== CUDA Environment Check ==="
echo

# Check nvcc availability
echo -n "Checking nvcc... "
if command -v nvcc &> /dev/null; then
    NVCC_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    echo "Found (version $NVCC_VERSION)"
    if [[ $VERBOSE -eq 1 ]]; then
        echo "  Path: $(which nvcc)"
    fi
else
    echo "NOT FOUND"
    echo "ERROR: nvcc not found in PATH"
    echo "Please install CUDA toolkit or add it to PATH"
    exit 1
fi

# Check CUDA_HOME
echo -n "Checking CUDA_HOME... "
if [[ -n "$CUDA_HOME" ]]; then
    echo "$CUDA_HOME"
    if [[ ! -d "$CUDA_HOME" ]]; then
        echo "WARNING: CUDA_HOME is set but directory does not exist"
    fi
else
    echo "Not set (optional)"
fi

# Check nvidia-smi
echo -n "Checking nvidia-smi... "
if command -v nvidia-smi &> /dev/null; then
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
    echo "Found (driver $DRIVER_VERSION)"
else
    echo "NOT FOUND"
    echo "WARNING: nvidia-smi not found. GPU may not be available on this machine."
    echo "This is OK if you're building for remote execution."
    exit 0
fi

# Get GPU information
echo
echo "=== GPU Information ==="
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "Number of GPUs: $GPU_COUNT"

if [[ $GPU_COUNT -gt 0 ]]; then
    nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv,noheader | \
    while IFS=, read -r idx name compute_cap memory; do
        echo
        echo "GPU $idx:"
        echo "  Name:            $name"
        echo "  Compute Cap:     $compute_cap"
        echo "  Memory:          $memory"

        # Suggest SM architecture
        COMPUTE_MAJOR=$(echo $compute_cap | cut -d'.' -f1)
        COMPUTE_MINOR=$(echo $compute_cap | cut -d'.' -f2)
        SM_ARCH="${COMPUTE_MAJOR}${COMPUTE_MINOR}"
        echo "  Recommended SM:  -gencode arch=compute_${SM_ARCH},code=sm_${SM_ARCH}"
    done
fi

# Check GPU health
if [[ $VERBOSE -eq 1 ]]; then
    echo
    echo "=== GPU Health ==="
    nvidia-smi --query-gpu=index,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.free --format=csv
fi

echo
echo "=== Check Complete ==="
echo "CUDA environment is ready for building."
