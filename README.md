Wwe present the following contributions to the field of accelerated symbolic regression:
(1) Semantic-Aware Subtree Caching: We introduce a mechanism to detect and reuse common sub-
expressions across the population, reducing redundant floating-point operations. (notable speedup compared to state of art)
(2) JIT Compilation Analysis: We provide a comparative analysis of runtime compilation (PTX to SASS,
or C++ to SASS) versus manual stack-based evaluation, illustrating the trade-offs between compilation
latency and execution throughput.
(3) Comprehensive Benchmarking: We validate our system on the AI Feynman benchmark dataset,
demonstrating performance gains over both CPU and GPU baseline implementations.

<img width="591" height="282" alt="Screenshot 2026-01-16 at 1 17 20 AM" src="https://github.com/user-attachments/assets/38cbbedc-a25c-4f87-ab60-1e2fc72854ed" />
<img width="581" height="151" alt="Screenshot 2026-01-16 at 1 17 53 AM" src="https://github.com/user-attachments/assets/5f400912-c001-4b1c-b9fb-f693b0448221" />

## 1) MIT Engaging setup for SR evolution benchmarks 

This repo also contains a full symbolic-regression evolution benchmark (GPU + CPU) under `src/cleaned` and `src/eval`. On MIT Engaging you typically want:

```bash
# 1) Get a GPU node (example: 1 L40S for 2 hours)
salloc -p mit_normal_gpu --gres=gpu:1 -N 1 -t 02:00:00

# 2) Load CUDA (system toolkit) and Python stack
module load cuda/12.4.0

# Option A: use system-wide Miniforge/Conda
module load miniforge          # if available on your cluster
conda create -n sr-bench python=3.11 -y
conda activate sr-bench

pip install -r requirements.txt

# Option B: bring your own Miniforge
# (outside scope of this README; just ensure nvcc and python are available)

# 3) Optional: start a shell on the allocated node
srun --pty bash -l

# 4) Verify GPU + nvcc
nvidia-smi
nvcc --version
```

Notes:
- The *top-level* `Makefile` uses `nvcc` directly and sets `-arch=$(GPU_ARCH)` with a default of `sm_89` (L40S / RTX 4090). Override with e.g. `GPU_ARCH=sm_90` if needed.
- The **cleaned SR system** under `src/cleaned` has its own `Makefile` (also using `nvcc` + `g++`) and expects a working CUDA 12.x toolkit.
 - If you change CUDA version, `GPU_ARCH`, or branches, it can help to run `make clean` in the relevant directory before rebuilding.


## 2) Datasets for evaluation

There are two main dataset families used in the benchmarks:

- **AI Feynman datasets** (for expression evaluation benchmarks).
- **Synthetic evolution datasets** (for long-evolution SR benchmarks like "test20").

All commands below assume you are in the project root: `symbolic_regression_gpu/`.

### 2.1 Download AI Feynman raw data

This downloads the Feynman/Bonus CSVs and tarballs into `data/ai_feyn/` and untars them.

```bash
cd src/script
python download_ai_feyn.py
```

After this you should have files like:

- `data/ai_feyn/Feynman_with_units/...`
- `data/ai_feyn/Feynman_without_units/...`
- `data/ai_feyn/FeynmanEquations.csv`, etc.

### 2.2 Preprocess AI Feynman into evaluator input format

`preprocess_ai_feyn.py` converts symbolic formulas into the prefix-encoded token format used by the C++ / CUDA evaluators and writes digest files.

The easiest way to generate all AI Feynman digests is via the top-level `Makefile` (from the project root):

```bash
cd symbolic_regression_gpu

# Generate everything (singles + multi + mutations)
make data_gen

# Or only specific families
make data_gen_singles    # data/ai_feyn/singles/
make data_gen_multi      # data/ai_feyn/multi/
make data_gen_mutations  # data/ai_feyn/mutations/
```

Under the hood, these `make` targets call `src/script/preprocess_ai_feyn.py` roughly as:

```bash
cd src/script

# (A) Generate single-expression digest files (1e6 datapoints each)
python preprocess_ai_feyn.py --write --quiet \
    --csv ../data/ai_feyn/FeynmanEquations.csv

# (B) Generate multi-expression files with different datapoint counts
python preprocess_ai_feyn.py --multi --dps 10000   # 10k dps
python preprocess_ai_feyn.py --multi --dps 100000  # 100k dps
python preprocess_ai_feyn.py --multi --dps 1000000 # 1M dps

# (C) Generate mutation benchmark files (100 mutated expressions)
python preprocess_ai_feyn.py --mutation            # 100k dps
python preprocess_ai_feyn.py --mutation --dps 1000000
```

These commands (or the `make data_gen*` wrappers) populate:

- `data/ai_feyn/singles/` – single-expression digest files.
- `data/ai_feyn/multi/` – multi-expression benchmark files.
- `data/ai_feyn/mutations/` – mutation benchmark files.

> The `make data_gen_*` targets in the top-level `Makefile` simply wrap these scripts.

### 2.3 Generate synthetic evolution datasets 

These datasets are used by the SR *evolution* benchmarks (as opposed to pure evaluation). From the project root you can generate them via `make`:

```bash
cd symbolic_regression_gpu

# Short & small evolution benchmark (5 gens, 100k dps)
make gen_data_short    # -> data/evolution_short_small

# Long & large evolution benchmark (20 gens, 500k dps)
make gen_data_long     # -> data/evolution_long_large

# Klein–Nishina "test20" benchmark (default: 20 gens, 1000 pop, 500k dps)
make gen_data_test20   # -> data/evolution_test20
```

The **cleaned SR system** under `src/cleaned` expects a `data/evolution_test20*` style directory containing:

- `shared_data.txt` – [num_dps rows] × [num_vars+1 columns], inputs + target.
- `gen_0.txt, gen_1.txt, ...` – prefix-encoded populations per generation.


## (used for Final Comparison)
For the **1M-DPS** "test20" dataset used by the default cleaned SR Makefile (`DATA=../../data/evolution_test20_1M`), run:

```bash
cd src/cleaned
make gen_data_test20_1M   # -> ../../data/evolution_test20_1M
```


## 3) Code overview: `src/cleaned` (SR system, cleaned benchmark)

The `src/cleaned` directory contains a self-contained SR evolution system and its GPU/CPU backends:

- `Makefile`
  - Builds SR system binaries into `../../build/`.
  - Main targets:
    - `cl_sr_opt`   – GPU Optimized Subtree evaluator (`USE_GPU_SUBTREE`).
    - `cl_sr_simple` – GPU Simple baseline (`USE_GPU_SIMPLE`).
    - `cl_sr_cpu`   – CPU baseline (`USE_CPU`).
    - `cl_sr_ptx`   – GPU PTX JIT version (`USE_GPU_PTX`).
    - `cl_sr_nvrtc` – GPU NVRTC JIT version (`USE_GPU_NVRTC`).
    - `cl_bench`    – combined micro-benchmark driver for GPU/CPU variants.

- `cl_main.cpp`
  - Minimal benchmark driver that:
    - Loads an evolution dataset (e.g., `data/evolution_test20`),
    - Converts shared data to float,
    - Calls multiple GPU backends (Simple, PTX, NVRTC, Optimized Subtree),
    - Optionally verifies against the CPU baseline,
    - Optionally writes a CSV with per-generation timings and MSEs.

- `cl_sr_system.cpp`
  - Full SR *evolution* loop used by `cl_sr_opt`, `cl_sr_simple`, `cl_sr_cpu`, `cl_sr_ptx`, `cl_sr_nvrtc`.
  - Given `POP`, `GENS`, a dataset dir, and DPS limit, it:
    - Initializes a random population of expressions.
    - Evaluates each generation on CPU or GPU (depending on compile-time flag).
    - Tracks timing breakdowns via `RunStats`.
    - Logs per-generation stats and writes `sr_results_*.csv` into `../../data/output/sr_results/`.

- `cl_gpu_eval.cu`
  - GPU **Optimized Subtree** evaluator with subtree detection and caching in GPU memory.
  - Two-stage kernels:
    - Subtree evaluation kernel (caches repeated sub-expressions).
    - Skeleton evaluation kernel (reuses subtree cache + evaluates remaining tree "skeleton").
  - Provides `create_gpu_context`, `destroy_gpu_context`, and `evaluate_gpu_mse_wrapper` used by `cl_sr_system.cpp`.

- `cl_gpu_simple.cu`
  - GPU **Simple** baseline: evaluates each expression directly with a stack-based prefix interpreter (no subtree caching).
  - Single evaluation kernel + reduction kernel for MSE.
  - Provides `create_gpu_simple_context`, `destroy_gpu_simple_context`, and `evaluate_gpu_simple_wrapper`.

- `cl_cpu_eval.cpp`
  - Multi-threaded CPU evaluator used by `cl_sr_system.cpp` when compiled with `USE_CPU`.
  - Uses a per-thread loop over expressions and datapoints, with environment variable `THREADS` controlling the number of CPU workers.

- `cl_gpu_ptx.cu`
  - GPU evaluator that **generates PTX at runtime** for each expression batch.
  - Uses NVRTC to compile helper code and the CUDA driver API (`cuModuleLoadData`) to JIT PTX → SASS.
  - Evaluates many expressions in a single multi-expression kernel; measures compile/JIT time separately.

- `cl_gpu_nvrtc.cu`
  - GPU evaluator that uses **NVRTC-generated C++ kernels** per expression batch.
  - Similar to `cl_gpu_ptx.cu` but stays at the NVRTC C++ level (PTX handled implicitly).

- `cl_gpu_common.h`
  - Shared GPU-side data structures and helpers for the cleaned SR GPU backends (buffer management, context structs, etc.).


## 4) Code overview: `src/eval` (evaluation backends)

The `src/eval` directory contains reusable evaluation backends (used by the **top-level** `Makefile` and evolution benchmarks):

- `evaluator.h`
  - Central header that defines `eval_batch(...)` via a preprocessor switch (`USE_CPU_SIMPLE`, `USE_GPU_SIMPLE`, `USE_GPU_JINHA`, `USE_GPU_CUSTOM_PEREXPR_*`, etc.).
  - Lets you compile different binaries (CPU, GPU simple, GPU subtree, Jinha batch, async, custom-kernel-per-expression, etc.) from the same `main.cpp`.

- CPU backends
  - `cpu_simple_single.cpp` – single-threaded CPU evaluator (prefix interpreter).
  - `cpu_simple_multi.cpp` – multi-threaded CPU evaluator (`USE_CPU_MULTI`), used for high-throughput baselines.
  - `cpu_subtree.cpp` – CPU evaluator with common-subtree detection and caching.

- GPU simple / subtree backends
  - `gpu_simple.cu` – basic GPU evaluator that mirrors the CPU prefix interpreter on GPU.
  - `gpu_subtree.cu` – GPU subtree evaluator with caching and batched execution over datapoints.
  - `gpu_subtree_batch_state.cu` + `gpu_subtree_batch_state_main.cpp` – stateful subtree evaluator variant (no datapoint batching), used for specific evolution benchmarks.

- GPU Jinha / async / evolve variants
  - `gpu_simple_jinha.cu` – multi-expression batch evaluator using the `eval_tree.cu` library (both per-expression and multi-expression batch kernels).
  - `gpu_async_jinha.cu` – async double-buffered GPU evaluator.
  - `gpu_simple_jinha_with_evolve.cu` – evolution-oriented Jinha evaluator (multi-expressions, evolve kernels).

- Custom-kernel-per-expression backends
  - `gpu_custom_kernel_per_expression.cu` – experimental path that **JIT-compiles one custom kernel per expression** (NVRTC/PTX), supporting:
    - Non-evolve multi-expression PTX batch.
    - Evolve variants.
    - NVRTC-based C++→PTX multi-expression batch.

- `common_eval.cpp`
  - Shared logic for running evolution benchmarks over multiple generations, used by both CPU and GPU variants via `eval_batch`.


## 5) Important test and benchmark commands

This section lists core commands for building and running benchmarks. Paths are relative to the project root unless stated otherwise.

### 5.1 Top-level evaluation benchmarks

From the project root (`symbolic_regression_gpu/`):

```bash
# Build CPU & GPU eval binaries (simple + multi-threaded + GPU simple)
make all_cpu        # builds cpu_eval, cpu_multi_eval, cpu_common

# Run single-expression CPU eval
make run_cpu_eval_single

# Run multi-expression CPU eval (100 expressions, 100k dps)
make run_cpu_eval_multi

# Run GPU simple eval (multi-expression, 1M dps)
make gpu_eval       # build GPU evaluator
make run_gpu_eval_multi

# Compare CPU vs GPU vs Jinha vs Async Jinha on a sample file
make compare_all_evals
```

For evolution-style benchmarks (using `data/evolution_*`):

```bash
# Generate evolution datasets (short/long)
make gen_data_short
make gen_data_long

# CPU long/short benchmarks (stateful subtree vs CPU multi)
make bench_long
make bench_short

# GPU evolution benchmarks (subtree, stateful, simple, Jinha, custom kernel, etc.)
make bench_test20
make bench_test20_gpu
make bench_test20_gpu_state
make bench_test20_gpu_simple_original
make bench_test20_gpu_jinha
make bench_test20_gpu_custom
make bench_test20_gpu_custom_nvrtc
make bench_test20_gpu_baseline
```

### 5.2 Primary commands for the cleaned SR system (`src/cleaned`)

From the cleaned SR system directory:

```bash
cd src/cleaned

# (Optional but recommended) clean previous builds if you changed CUDA/flags/branch
make clean

# Build all cleaned SR binaries (GPU Optimized, GPU Simple, CPU, PTX, NVRTC, and cl_bench)
make            # or: make all
```

The main runtime wrappers are:

```make
run_sr_opt      # GPU Optimized Subtree SR system
run_sr_simple   # GPU Simple SR system
run_sr_cpu      # CPU baseline SR system
run_sr_ptx      # GPU PTX JIT SR system
run_sr_nvrtc    # GPU NVRTC JIT SR system
```

Each wrapper ultimately runs the corresponding binary (`cl_sr_opt`, `cl_sr_simple`, etc.) via `srun` on an Engaging GPU node, with arguments:

```bash
THREADS=<threads> srun --gres=gpu:1 ../../build/cl_sr_opt   <POP> <GENS> <DATA_DIR> <LIMIT_DPS> <SEED>
THREADS=<threads> srun --gres=gpu:1 ../../build/cl_sr_simple <POP> <GENS> <DATA_DIR> <LIMIT_DPS> <SEED>
THREADS=<threads>          ../../build/cl_sr_cpu    <POP> <GENS> <DATA_DIR> <LIMIT_DPS> <SEED>
```

Internally, `cl_sr_system.cpp` runs a simple genetic algorithm over prefix-encoded expression trees:

- **Initialization**: `POP` random expressions are generated via `generate_random_expression` (tokens + per-node values).
- **Fitness**: in each generation, all individuals are evaluated on the shared dataset and assigned an MSE-based fitness.
- **Selection & breeding**:
  - Top ~5% of the population are copied unchanged as *elites*.
  - The remaining slots are filled using 3-way tournament selection:
    - With 20% probability, a selected parent is **mutated** using `mutate_expression(...)`.
    - With 80% probability, two parents are selected and combined via **crossover** using `crossover_expressions(...)`.

This evolution logic is identical across GPU and CPU backends; only the evaluator used to compute MSE differs between `cl_sr_opt`, `cl_sr_simple`, `cl_sr_ptx`, `cl_sr_nvrtc`, and `cl_sr_cpu`.

Key cleaned tasks (as defined in `src/cleaned/Makefile`):

- Fixed-population comparison (time vs gen + breakdown plots)

  ```bash
  # GPU: POP=100 (5 runs each of Opt/Simple/PTX/NVRTC)
  make run_sr_task_gen_pop100_gpu

  # GPU: POP=2000 (5 runs each of Opt/Simple)
  make run_sr_task_gen_pop2000_gpu

  # CPU: POP=100 and POP=2000 baselines
  make run_sr_task_gen_pop100_cpu
  make run_sr_task_gen_pop2000_cpu
  ```

  These produce `sr_results_*.csv` used by:

  ```bash
  # Time vs generation, full + GPU-only, plus kernel-only breakdowns
  make plots   # invokes plot_results.py, plot_results_kernel.py, plot_results_breakdown.py, etc.
  ```

- Population scaling benchmarks (Plot-B, time vs population + breakdown)

  ```bash
  # GPU population scaling (50 → 8000)
  make run_sr_task_scale_pop_gpu

  # CPU population scaling (50 → 400)
  make run_sr_task_scale_pop_cpu
  ```

  These write CSVs consumed by:

  ```bash
  python ../utils/plot_results_pop.py           --dir ../../data/output/sr_results/plot-B --filter GPU
  python ../utils/plot_results_breakdown_pop.py --dir ../../data/output/sr_results/plot-B --filter GPU
  ```

- DPS scaling benchmarks (Plot-C, time vs datapoints)

  ```bash
  # GPU DPS scaling at fixed POP=2000
  make run_sr_task_scale_dps_gpu
  ```

  Used by:

  ```bash
  python ../utils/plot_results_dps.py --dir ../../data/output/sr_results/plot-C
  ```


## 6) Summary

- The top-level CUDA evaluator (`eval_tree.cu`) and CLI (`run_gpu`) demonstrate single-expression prefix-tree evaluation on CPU and GPU.
- The evaluation backends in `src/eval` generalize this to multi-expression, multi-backend benchmarks (CPU simple/multi/subtree, GPU simple/subtree/Jinha, async, custom-kernel-per-expression, etc.).
- The cleaned SR system in `src/cleaned` implements a full symbolic-regression evolution loop with multiple GPU/CPU backends and produces detailed CSV logs and breakdown plots for the benchmarks used in the experiments.
