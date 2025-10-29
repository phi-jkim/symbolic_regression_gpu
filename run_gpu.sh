#!/usr/bin/env bash
set -euo pipefail

# Build shared lib and CLI (unset conda toolchain flags that nvcc doesn't accept)
env -u CFLAGS -u CXXFLAGS -u LDFLAGS -u CPPFLAGS \
    make -C "$(dirname "$0")"

# Run the CUDA CLI examples
"$(dirname "$0")"/run_gpu
