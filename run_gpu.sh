#!/usr/bin/env bash
set -euo pipefail

# Build shared lib and CLI
make -C "$(dirname "$0")"

# Run the CUDA CLI examples
"$(dirname "$0")"/run_gpu
