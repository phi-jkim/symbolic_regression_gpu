# symbolic_regression_gpu

Optimized single-point prefix-tree evaluation on CPU and a single-thread CUDA kernel, plus a Julia example using SymbolicRegression.jl.

This doc covers:
- Starting an interactive GPU session on MIT Engaging (SLURM)
- Building and running the CUDA implementation
- Running the Julia single-point example

sin(x0)+cos(x1)+3 @ [0.5,1.2] -> cpu=3.8417833 gpu=3.8417833
sin(x0)+cos(x1)+3 @ [0.5,1.2] timings: cpu_avg_us=0.090 (iters=1000) gpu_avg_us=135.350 (iters=100)
sin(x0)+cos(x1)+3 @ [0.5,0.3] -> cpu=4.4347620 gpu=4.4347620
sin(x0)+cos(x1)+3 @ [0.5,0.3] timings: cpu_avg_us=0.074 (iters=1000) gpu_avg_us=134.060 (iters=100)

sr_eval  = 3.8417833
complete = true
eval_time_ms = 283.679
avg_time_us (iters=1000) = 0.476
sr_eval  = 4.434762
complete = true
eval_time_ms = 0.001
avg_time_us (iters=1000) = 0.481


## 1) MIT Engaging: start an interactive GPU session

On MIT Engaging (SLURM-based), request an interactive node with a GPU, e.g.:

```bash
# Example: 1 GPU for 20 minutes on the 'gpu' partition
salloc -p mit_normal_gpu --gres=gpu:1 -N 1 -t 00:20:00

module load miniforge 

# or load conda environment 
conda activate ml_train 

# Once the session starts, run commands with srun (or just use the shell if configured)
srun --pty bash -l

# Verify GPU visibility on the allocated node
nvidia-smi
```

Notes:
- Partitions/flags may differ. Common flags: `-p gpu`, `-G 1` (number of GPUs), `--time=HH:MM:SS`.
- If modules are used, you may need to `module load cuda/<version>` on the compute node.

## 2) Build and run the CUDA implementation

This project provides a shared library (`libevaltree.so`) and a small CLI (`run_gpu`). The CLI evaluates a few prefix-encoded trees using both CPU and GPU paths.

Prereqs:
- CUDA toolkit with `nvcc` in PATH (e.g., via module load or conda `cuda-toolkit`/`cuda-nvcc`).

Build and run:

```bash
# From the project directory
cd /home/jinhakim/symbolic_regression_gpu

# Build (Makefile is configured to avoid conda-injected flags nvcc doesn't accept)
bash ./run_gpu.sh

# Or build manually (equivalent)
# env -u CFLAGS -u CXXFLAGS -u LDFLAGS -u CPPFLAGS make

# Run the GPU CLI directly (on a GPU node)
./run_gpu
```

Expected output (GPU should match CPU):

```
sin(x0)+cos(x1)+3 @ [0.5,1.2] -> cpu=3.8417833 gpu=3.8417833
sin(x0)+cos(x1)+3 @ [0.5,0.3] -> cpu=4.4347620 gpu=4.4347620
```

## 3) Run the Julia example (no GPU required)

The Julia script demonstrates evaluating an expression using `SymbolicRegression.eval_tree_array` on a single point. It does not require CUDA.

One-time setup:

```bash
# Install Julia (e.g., via conda)
conda install -c conda-forge julia

# Instantiate the SymbolicRegression.jl project (downloads Julia dependencies)
julia --project=/home/jinhakim/SymbolicRegression.jl -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

Run the example:

```bash
julia --project=/home/jinhakim/SymbolicRegression.jl /home/jinhakim/symbolic_regression_gpu/run_single_point.jl
```

It prints the evaluation for two points with the expression `sin(x1) + cos(x2) + 3` (variable names are 1-based in that script).

## 4) Token encoding used by the CUDA evaluator

- `TOK_CONST = 0` (values[i] holds the constant value)
- `TOK_VAR = -1` (values[i] holds the zero-based feature index)
- Operators: `1=ADD, 2=SUB, 3=MUL, 4=DIV, 5=SIN, 6=COS, 7=EXP, 8=LOG`
- Trees are prefix-encoded (operators before their operands). The GPU kernel uses a stack to compute the result.

## 5) Notes on optimizations

- Fused multiply-add (FMA) for `(a*b) +/- c` patterns using a lightweight tagged stack for multiplicative subexpressions.
- Safe variants for DIV/LOG to reduce NaNs/Infs.


Eval 