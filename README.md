# symbolic_regression_gpu

Optimized single-point prefix-tree evaluation on CPU and a single-thread CUDA kernel.

The CUDA implementation fuses simple patterns similar in spirit to `eval_tree_array` from SymbolicRegression.jl:
- Fuse add/sub with a child multiplication via FMA when possible.
- Track multiplicative subexpressions on the stack using a light "tagged" representation so chained multiplies can collapse.

The library exports two C symbols:
- `float eval_tree_cpu(const int* tokens, const float* values, const float* features, int len, int num_features)`
- `void  eval_tree_gpu(const int* tokens, const float* values, const float* features, int len, int num_features, float* out_host)`

Token encoding:
- `TOK_CONST = 0` with `values[i]` holding the constant value.
- `TOK_VAR = -1` with `values[i]` holding the zero-based feature index.
- Operators (positive): `1=ADD, 2=SUB, 3=MUL, 4=DIV, 5=SIN, 6=COS, 7=EXP, 8=LOG`.

## Build

From this directory:

```bash
default: make
```

This produces `libevaltree.so` using NVCC.

## Run (examples)

You can call the library from Julia (or any FFI). Below are minimal Julia examples.

- Example model y = sin(x0) + 3 evaluated on a single point x0 = 0.5:

```julia
lib = joinpath(@__DIR__, "libevaltree.so")
# Prefix: [ADD, SIN, VAR, CONST] with var index 0 and const 3
tokens = Int32[1, 5, -1, 0]
values = Float32[0, 0, 0, 3]
x = Float32[0.5]

cpu = ccall((:eval_tree_cpu, lib), Cfloat,
            (Ptr{Cint}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cint),
            pointer(tokens), pointer(values), pointer(x), Cint(length(tokens)), Cint(length(x)))

out_ref = Ref{Cfloat}(0)
ccall((:eval_tree_gpu, lib), Cvoid,
      (Ptr{Cint}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cint, Ptr{Cfloat}),
      pointer(tokens), pointer(values), pointer(x), Cint(length(tokens)), Cint(length(x)), out_ref)
println((cpu=cpu, gpu=out_ref[]))
```

Notes:
- The GPU kernel runs with a single block and single thread and uses dynamic shared memory sized to `4*len*sizeof(float) + len*sizeof(int)` internally; the host function computes this for you.
- Division and log are guarded with small epsilons to avoid nans/infs.

## Relationship to eval_tree_array optimizations

This implementation mirrors a subset of `eval_tree_array`'s micro-optimizations:
- Fused multiply-add for `a*b + c` and `a*b - c` (and commuted variants), using `fmaf` / `__fmaf_rn`.
- Collapsing chained products by carrying a multiplicative tag `(pa * pb)` on the stack and reusing it across parents.

It is not a drop-in replacement for all SymbolicRegression.jl evaluation modes, but is intended for fast single-point evaluation on a prefix-encoded tree.
