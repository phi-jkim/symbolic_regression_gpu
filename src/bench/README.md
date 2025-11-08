# Benchmarking

## Overview

We need to run benchmark in various type of settings:

- Language: C++, CUDA, (todo: Julia, Python)
- Environment: Single-core CPU, Multi-core CPU, GPU-only, CPU+GPU (Engaging Cluster)

To be in fair ground, and to measure conveniently, there will be a benchmark runner in Python, which will only coordinate running executables and reporting.
i.e. Python runner script should be completely irrelevant to the benchmark results, algorithms, etc. Only coordinating and reporting.

Each executable should work as follows:

- Get two command-line arguments, input_file and output_file
- Input and output files will be simple TOML format file, supporting only comments, single and double quoted strings, decimal ints and floats.
- The point is to keep only two arguments. All input/output related to the main logic (e.g. data name, xs, ys, estimated formula, etc) should be derived from the input file.
  - For example, the input TOML file may have `input_data_file = 'data/ai_feyn/Feynman_with_units/I.6.2'`, then the executable will read the file and extract the data by itself.
- The output file should contain both the algorithm results and the benchmark info. (e.g. estimated formula, timing, number of nodes, etc)
- `stdout`, `stderr` will be used to show any run-time temporary results (progress info, any exceptions, etc), which won't be important for benchmark, but might be useful for inspection.

Notes

- Ideally, we should describe the input and output file's content here as well, to standardize across different implementations.
- There will be multiple problems to be benchmarked (e.g. Evaluation, SR, GP, etc), and ideally they should use all the same framework.
- `stdout` and `stderr` may be logged, for debugging or inspection.

## How to run

For single executable:

```bash
echo TODO
```
