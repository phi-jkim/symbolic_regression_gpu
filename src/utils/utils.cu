// eval_tree.cu
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <cmath>

// ---- Low-level async copy helpers (Ampere+) ----

__device__ __forceinline__ void cp_async4(void *smem_ptr, const void *glob_ptr) {
    const int BYTES = 16;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{\n"
        "   cp.async.cg.shared.global [%0], [%1], %2;\n"
        "}\n" : : "r"(smem),
        "l"(glob_ptr),
        "n"(BYTES));
}

__device__ __forceinline__ void async_commit_group() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N> __device__ __forceinline__ void async_wait_pending() {
    // waits until there are <= N groups still pending (PTX semantics)
    asm volatile("cp.async.wait_group %0;\n" : : "n"(N));
}

template<int BYTES>
__device__ __forceinline__ void cp_async(void* smem_ptr, const void* glob_ptr) {
  static_assert(BYTES==4 || BYTES==8 || BYTES==16, "cp.async supports 4/8/16 byte ops");
  uint32_t smem = (uint32_t)__cvta_generic_to_shared(smem_ptr);
  asm volatile("{\n\tcp.async.ca.shared.global [%0], [%1], %2;\n}"
               :: "r"(smem), "l"(glob_ptr), "n"(BYTES));
}

// ---- Token / operator encoding ----

static const int TOK_CONST = 0;
static const int TOK_VAR   = -1;

// Operator codes (positive) - Updated to match preprocessing encoding
static const int OP_ADD = 1;
static const int OP_SUB = 2;
static const int OP_MUL = 3;
static const int OP_DIV = 4;
static const int OP_POW = 5;
static const int OP_MIN = 6;
static const int OP_MAX = 7;
static const int OP_LOOSE_DIV = 8;
static const int OP_LOOSE_POW = 9;
static const int OP_SIN = 10;
static const int OP_COS = 11;
static const int OP_TAN = 12;
static const int OP_SINH = 13;
static const int OP_COSH = 14;
static const int OP_TANH = 15;
static const int OP_EXP = 16;
static const int OP_LOG = 17;
static const int OP_INV = 18;
static const int OP_ASIN = 19;
static const int OP_ACOS = 20;
static const int OP_ATAN = 21;
static const int OP_LOOSE_LOG = 22;
static const int OP_LOOSE_INV = 23;
static const int OP_ABS = 24;
static const int OP_NEG = 25;
static const int OP_SQRT = 26;
static const int OP_LOOSE_SQRT = 27;
// Note: OP_IF (29) removed for now as not used in preprocessing

__host__ __device__ inline float madd(float a, float b, float c) {
#ifdef __CUDA_ARCH__
    return __fmaf_rn(a, b, c);
#else
    return fmaf(a, b, c);
#endif
}

__host__ __device__ inline int op_arity(int op);
__host__ __device__ inline float apply_op(int op, float a, float b);
__host__ __device__ inline float safe_div(float a, float b);

#ifndef MAX_EVAL_STACK
#define MAX_EVAL_STACK 128
#endif

// ---- Batched (datapoint-parallel) evaluator ----

__global__ void eval_prefix_kernel_batch(const int* __restrict__ tokens,
                                         const float* __restrict__ values,
                                         const float* __restrict__ X, // [dataPoints, num_features]
                                         int len,
                                         int num_features,
                                         int dataPoints,
                                         float* __restrict__ out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dataPoints) return;
    if (len > MAX_EVAL_STACK) {
        if (idx == 0) out[0] = NAN;
        return;
    }

    float s_val[MAX_EVAL_STACK];
    int   s_tag[MAX_EVAL_STACK];
    float s_pa[MAX_EVAL_STACK];
    float s_pb[MAX_EVAL_STACK];
    int sp = 0;

    const float* x = X + (size_t)idx * (size_t)num_features;

    // Evaluate one prefix expression for datapoint idx
    for (int i = len - 1; i >= 0; --i) {
        const int t = tokens[i];
        if (t == TOK_CONST) {
            s_val[sp] = values[i]; s_tag[sp] = 0; s_pa[sp] = 0.0f; s_pb[sp] = 0.0f; ++sp;
        } else if (t == TOK_VAR) {
            int fidx = (int)values[i];
            float v = (fidx >= 0 && fidx < num_features) ? x[fidx] : 0.0f;
            s_val[sp] = v; s_tag[sp] = 0; s_pa[sp] = 0.0f; s_pb[sp] = 0.0f; ++sp;
        } else {
            const int ar = op_arity(t);
            if (ar == 1) {
                float a = s_val[--sp];
                float r = apply_op(t, a, 0.0f);
                s_val[sp] = r; s_tag[sp] = 0; s_pa[sp] = 0.0f; s_pb[sp] = 0.0f; ++sp;
            } else if (ar == 2) {
                float a_v = s_val[--sp]; int a_t = s_tag[sp]; float a_pa = s_pa[sp]; float a_pb = s_pb[sp];
                float b_v = s_val[--sp]; int b_t = s_tag[sp]; float b_pa = s_pa[sp]; float b_pb = s_pb[sp];
                (void)a_t; (void)b_t; (void)a_pa; (void)a_pb; (void)b_pa; (void)b_pb;
                float r;
                if (t == OP_ADD)      r = a_v + b_v;
                else if (t == OP_SUB) r = a_v - b_v;
                else if (t == OP_MUL) r = a_v * b_v;
                else if (t == OP_DIV) r = a_v / b_v;
                else                  r = apply_op(t, a_v, b_v);
                s_val[sp] = r; s_tag[sp] = 0; s_pa[sp] = 0.0f; s_pb[sp] = 0.0f; ++sp;
            } else { // ternary IF
                float c = s_val[--sp];
                float b = s_val[--sp];
                float a = s_val[--sp];
                float r = (a > 0.0f) ? b : c;
                s_val[sp] = r; s_tag[sp] = 0; s_pa[sp] = 0.0f; s_pb[sp] = 0.0f; ++sp;
            }
        }
    }
    out[idx] = (sp > 0) ? s_val[sp - 1] : 0.0f;
}

extern "C" void eval_tree_gpu_batch(const int* tokens,
                                    const float* values,
                                    const float* X,
                                    int len,
                                    int num_features,
                                    int dataPoints,
                                    float* out_dev,
                                    int blocks,
                                    int threads) {
    if (threads <= 0) threads = 256;
    if (blocks  <= 0) blocks  = (dataPoints + threads - 1) / threads;
    eval_prefix_kernel_batch<<<blocks, threads>>>(tokens, values, X, len, num_features, dataPoints, out_dev);
}

// ---- Single-expression evaluators (GPU + CPU) ----

__host__ __device__ inline int op_arity(int op) {
    // Unary operators (10-27)
    if (op == OP_SIN || op == OP_COS || op == OP_TAN ||
        op == OP_SINH || op == OP_COSH || op == OP_TANH ||
        op == OP_EXP || op == OP_LOG || op == OP_INV ||
        op == OP_ASIN || op == OP_ACOS || op == OP_ATAN ||
        op == OP_LOOSE_LOG || op == OP_LOOSE_INV ||
        op == OP_ABS || op == OP_NEG || op == OP_SQRT || op == OP_LOOSE_SQRT)
        return 1;
    // Binary operators (1-9)
    return 2;
}

static const float DELTA = 1e-9f;
static const float MAX_VAL = 1e9f;

__host__ __device__ inline float clamp_val(float r) { return r; }

__host__ __device__ inline float safe_div(float a, float b) {
    const float eps = 1e-12f;
    float denom = fabsf(b) < eps ? (b < 0 ? -eps : eps) : b;
    return a / denom;
}

__host__ __device__ inline float loose_div(float a, float b) {
    float denom = fabsf(b) <= DELTA ? (b < 0 ? -DELTA : DELTA) : b;
    return a / denom;
}

__host__ __device__ inline float safe_log(float a) {
    const float eps = 1e-12f;
    return logf(fabsf(a) + eps);
}

__host__ __device__ inline float loose_log(float a) {
    if (a == 0.0f) return -MAX_VAL;
    return logf(fabsf(a));
}

__host__ __device__ inline float safe_inv(float a) {
    if (a == 0.0f) return NAN;
    return 1.0f / a;
}

__host__ __device__ inline float loose_inv(float a) {
    float denom = fabsf(a) <= DELTA ? (a < 0 ? -DELTA : DELTA) : a;
    return 1.0f / denom;
}

__host__ __device__ inline float apply_op(int op, float a, float b) {
    switch (op) {
        // Binary operators (1-9)
        case OP_ADD: return clamp_val(a + b);
        case OP_SUB: return clamp_val(a - b);
        case OP_MUL: return clamp_val(a * b);
        case OP_DIV: return (b == 0.0f) ? NAN : clamp_val(a / b);
        case OP_POW: return clamp_val(powf(a, b));
        case OP_MIN: return clamp_val(a <= b ? a : b);
        case OP_MAX: return clamp_val(a >= b ? a : b);
        case OP_LOOSE_DIV: return clamp_val(loose_div(a, b));
        case OP_LOOSE_POW: return (a == 0.0f && b == 0.0f) ? 0.0f : clamp_val(powf(fabsf(a), b));
        // Unary operators (10-27)
        case OP_SIN: return clamp_val(sinf(a));
        case OP_COS: return clamp_val(cosf(a));
        case OP_TAN: return clamp_val(tanf(a));
        case OP_SINH: return clamp_val(sinhf(a));
        case OP_COSH: return clamp_val(coshf(a));
        case OP_TANH: return clamp_val(tanhf(a));
        case OP_EXP: return clamp_val(expf(a));
        case OP_LOG: return clamp_val(logf(a));
        case OP_INV: return clamp_val(safe_inv(a));
        case OP_ASIN: return clamp_val(asinf(a));
        case OP_ACOS: return clamp_val(acosf(a));
        case OP_ATAN: return clamp_val(atanf(a));
        case OP_LOOSE_LOG: return clamp_val(loose_log(a));
        case OP_LOOSE_INV: return clamp_val(loose_inv(a));
        case OP_ABS: return clamp_val(fabsf(a));
        case OP_NEG: return clamp_val(-a);
        case OP_SQRT: return clamp_val(sqrtf(a));
        case OP_LOOSE_SQRT: return clamp_val(sqrtf(fabsf(a)));
        default: return 0.0f;
    }
}

struct TagVal {
    float v;
    int tag;    // 0 = plain value (placeholder for future algebraic tags)
    float pa;
    float pb;
};


// ---- Single-expression GPU launcher (shared-memory stack) ----

__global__ void eval_prefix_kernel(const int* __restrict__ tokens,
                                   const float* __restrict__ values,
                                   const float* __restrict__ x,
                                   int len,
                                   int num_features,
                                   float* __restrict__ out) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    extern __shared__ unsigned char smem[];
    float* s_val = reinterpret_cast<float*>(smem);
    int*   s_tag = reinterpret_cast<int*>(s_val + len);
    float* s_pa  = reinterpret_cast<float*>(s_tag + len);
    float* s_pb  = reinterpret_cast<float*>(s_pa + len);
    int sp = 0;

    for (int i = len - 1; i >= 0; --i) {
        const int t = tokens[i];
        if (t == TOK_CONST) {
            s_val[sp] = values[i]; s_tag[sp] = 0; s_pa[sp] = 0.0f; s_pb[sp] = 0.0f; ++sp;
        } else if (t == TOK_VAR) {
            int idx = (int)values[i];
            float v = 0.0f;
            if (idx >= 0 && idx < num_features) v = x[idx];
            s_val[sp] = v; s_tag[sp] = 0; s_pa[sp] = 0.0f; s_pb[sp] = 0.0f; ++sp;
        } else {
            const int ar = op_arity(t);
            if (ar == 1) {
                float a = s_val[--sp];
                float r = apply_op(t, a, 0.0f);
                s_val[sp] = r; s_tag[sp] = 0; s_pa[sp] = 0.0f; s_pb[sp] = 0.0f; ++sp;
            } else if (ar == 2) {
                float a_v = s_val[--sp];
                float b_v = s_val[--sp];
                float r;
                if (t == OP_ADD)      r = clamp_val(a_v + b_v);
                else if (t == OP_SUB) r = clamp_val(a_v - b_v);
                else if (t == OP_MUL) r = clamp_val(a_v * b_v);
                else if (t == OP_DIV) r = (b_v == 0.0f) ? NAN : a_v / b_v;
                else                   r = apply_op(t, a_v, b_v);
                s_val[sp] = r; s_tag[sp] = 0; s_pa[sp] = 0.0f; s_pb[sp] = 0.0f; ++sp;
            } else {
                float cond = s_val[--sp];
                float then_v = s_val[--sp];
                float else_v = s_val[--sp];
                float r = (cond > 0.0f) ? then_v : else_v;
                s_val[sp] = r; s_tag[sp] = 0; s_pa[sp] = 0.0f; s_pb[sp] = 0.0f; ++sp;
            }
        }
    }
    out[0] = (sp > 0) ? s_val[sp - 1] : 0.0f;
}

extern "C" void eval_tree_gpu(const int* tokens,
                               const float* values,
                               const float* features,
                               int len,
                               int num_features,
                               float* out_host) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        *out_host = NAN;
        return;
    }
    int *d_tokens = nullptr;
    float *d_values = nullptr, *d_x = nullptr, *d_out = nullptr;

    cudaMalloc(&d_tokens, sizeof(int) * len);
    cudaMalloc(&d_values, sizeof(float) * len);
    cudaMalloc(&d_x, sizeof(float) * num_features);
    cudaMalloc(&d_out, sizeof(float));

    cudaMemcpy(d_tokens, tokens, sizeof(int) * len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values, sizeof(float) * len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, features, sizeof(float) * num_features, cudaMemcpyHostToDevice);

    size_t smem = sizeof(float) * (size_t)len /* s_val */
                + sizeof(int)   * (size_t)len /* s_tag */
                + sizeof(float) * (size_t)len /* s_pa */
                + sizeof(float) * (size_t)len /* s_pb */;
    {
        float init = -123.456f;
        cudaMemcpy(d_out, &init, sizeof(float), cudaMemcpyHostToDevice);
    }
    eval_prefix_kernel<<<1, 1, smem>>>(d_tokens, d_values, d_x, len, num_features, d_out);
    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        *out_host = NAN;
        cudaFree(d_tokens); cudaFree(d_values); cudaFree(d_x); cudaFree(d_out);
        return;
    }
    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        *out_host = NAN;
        cudaFree(d_tokens); cudaFree(d_values); cudaFree(d_x); cudaFree(d_out);
        return;
    }

    float out_val = 0.0f;
    cudaMemcpy(&out_val, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    *out_host = out_val;

    cudaFree(d_tokens);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_out);
}

// ---- Async double-buffer evaluator (single expression, multiple datapoints) ----

__global__ void eval_async_kernel(const int* __restrict__ tokens,
                                   const float* __restrict__ values,
                                   const float* __restrict__ X,
                                   int len,
                                   int num_features,
                                   int dataPoints,
                                   float* __restrict__ out) {
    if (len <= 0 || len > MAX_EVAL_STACK) return;

    extern __shared__ float shmem[];
    float* buf0 = shmem;
    float* buf1 = shmem + (size_t)blockDim.x * (size_t)num_features;

    const int tile = blockDim.x;
    const int tiles_per_grid_y = gridDim.y > 0 ? gridDim.y : 1;
    int t0 = blockIdx.y * tile;
    const int stride = tile * tiles_per_grid_y;

    const int tid = threadIdx.x;

    // Lambda to copy a tile of data
    // Uses cp.async on sm_80+ (Ampere), falls back to sync copy on older GPUs
    auto copy_tile = [&](int start, float* dst) {
        const int dp = start + tid;
        if (dp < dataPoints) {
            const float* src = X + (size_t)dp * (size_t)num_features;
#if __CUDA_ARCH__ >= 800
            // Ampere+ (sm_80): Use async copy for true double-buffering
            int f = 0;
            for (; f + 3 < num_features; f += 4) {
                void* sptr = (void*)(&dst[(size_t)tid * (size_t)num_features + f]);
                const void* gptr = (const void*)(&src[f]);
                if ((((uintptr_t)sptr | (uintptr_t)gptr) & 0xF) == 0) {
                    cp_async4(sptr, gptr);
                } else {
                    cp_async<4>((void*)((float*)sptr + 0), (const void*)((const float*)gptr + 0));
                    cp_async<4>((void*)((float*)sptr + 1), (const void*)((const float*)gptr + 1));
                    cp_async<4>((void*)((float*)sptr + 2), (const void*)((const float*)gptr + 2));
                    cp_async<4>((void*)((float*)sptr + 3), (const void*)((const float*)gptr + 3));
                }
            }
            for (; f < num_features; ++f) {
                cp_async<4>(&dst[(size_t)tid * (size_t)num_features + f], &src[f]);
            }
#else
            // Pre-Ampere (sm_75 and below): Fallback to synchronous copy
            for (int f = 0; f < num_features; ++f) {
                dst[(size_t)tid * (size_t)num_features + f] = src[f];
            }
#endif
        } else {
            // Zero-pad when past the end
            for (int f = 0; f < num_features; ++f) {
                dst[(size_t)tid * (size_t)num_features + f] = 0.0f;
            }
        }
    };

    if (t0 >= dataPoints) return;

    // Stage 0: prefetch first tile into buf0
    copy_tile(t0, buf0);
#if __CUDA_ARCH__ >= 800
    async_commit_group();
    async_wait_pending<0>();
#endif
    __syncthreads();

    int t = t0;
    float* cur = buf0;
    float* nxt = buf1;

    // Double-buffer pipeline
    // sm_80+: Async copy overlaps with compute
    // sm_75: Synchronous copy, still benefits from double-buffering pattern
    while (t < dataPoints) {
        const int next_t = t + stride;
        if (next_t < dataPoints) {
            copy_tile(next_t, nxt);
#if __CUDA_ARCH__ >= 800
            async_commit_group();
#endif
        }

        const int dp = t + tid;
        if (dp < dataPoints) {
            float s_val[MAX_EVAL_STACK];
            int   s_tag[MAX_EVAL_STACK];
            float s_pa[MAX_EVAL_STACK];
            float s_pb[MAX_EVAL_STACK];
            int sp = 0;

            // Evaluate expression tree
            for (int i = len - 1; i >= 0; --i) {
                const int tt = tokens[i];
                if (tt == TOK_CONST) {
                    s_val[sp] = values[i]; s_tag[sp] = 0; s_pa[sp] = 0.0f; s_pb[sp] = 0.0f; ++sp;
                } else if (tt == TOK_VAR) {
                    int fidx = (int)values[i];
                    float v = (fidx >= 0 && fidx < num_features) ? cur[(size_t)tid * (size_t)num_features + fidx] : 0.0f;
                    s_val[sp] = v; s_tag[sp] = 0; s_pa[sp] = 0.0f; s_pb[sp] = 0.0f; ++sp;
                } else {
                    const int ar = op_arity(tt);
                    if (ar == 1) {
                        float a = s_val[--sp];
                        float r = apply_op(tt, a, 0.0f);
                        s_val[sp] = r; s_tag[sp] = 0; s_pa[sp] = 0.0f; s_pb[sp] = 0.0f; ++sp;
                    } else if (ar == 2) {
                        float a_v = s_val[--sp];
                        float b_v = s_val[--sp];
                        float r;
                        if (tt == OP_ADD)      r = clamp_val(a_v + b_v);
                        else if (tt == OP_SUB) r = clamp_val(a_v - b_v);
                        else if (tt == OP_MUL) r = clamp_val(a_v * b_v);
                        else if (tt == OP_DIV) r = (b_v == 0.0f) ? NAN : a_v / b_v;
                        else                    r = apply_op(tt, a_v, b_v);
                        s_val[sp] = r; s_tag[sp] = 0; s_pa[sp] = 0.0f; s_pb[sp] = 0.0f; ++sp;
                    }
                }
            }
            out[dp] = (sp > 0) ? s_val[sp - 1] : 0.0f;
        }

        // Wait for async copy to complete (sm_80+) and sync before swapping buffers
#if __CUDA_ARCH__ >= 800
        async_wait_pending<0>();
#endif
        __syncthreads();
        float* tmp = cur; cur = nxt; nxt = tmp;
        t = next_t;
    }
}

extern "C" void eval_tree_gpu_async(const int* tokens,
                                    const float* values,
                                    const float* X,
                                    int len,
                                    int num_features,
                                    int dataPoints,
                                    float* out_dev,
                                    int blocks_y,
                                    int threads) {
    if (threads <= 0) threads = 256;
    if (blocks_y <= 0) blocks_y = (dataPoints + threads - 1) / threads;
    dim3 grid(1, blocks_y, 1);
    size_t smem = sizeof(float) * (size_t)threads * (size_t)num_features * 2u;
    eval_async_kernel<<<grid, threads, smem>>>(tokens, values, X, len, num_features, dataPoints, out_dev);
}
