#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <vector>
#include <algorithm>
#include <stdio.h>

// Token encoding
// TOK_CONST = 0 -> values[i] holds the constant value
// TOK_VAR   = -1 -> values[i] holds the variable index (0..num_features-1)
// Operators (positive):
//  Existing: 1=ADD, 2=SUB, 3=MUL, 4=DIV, 5=SIN, 6=COS, 7=EXP, 8=LOG
//  Added (to mirror EVOGP's Function set):
//   9=LOOSE_DIV, 10=POW, 11=LOOSE_POW, 12=MAX, 13=MIN,
//   14=LT, 15=GT, 16=LE, 17=GE,
//   18=TAN, 19=SINH, 20=COSH, 21=TANH,
//   22=LOOSE_LOG,
//   23=INV, 24=LOOSE_INV, 25=NEG, 26=ABS, 27=SQRT, 28=LOOSE_SQRT,
//   29=IF (ternary: cond>0 ? a : b)

static const int TOK_CONST = 0;
static const int TOK_VAR   = -1;

// Operator codes (positive)
static const int OP_ADD = 1;
static const int OP_SUB = 2;
static const int OP_MUL = 3;
static const int OP_DIV = 4;
static const int OP_SIN = 5;
static const int OP_COS = 6;
static const int OP_EXP = 7;
static const int OP_LOG = 8;
static const int OP_LOOSE_DIV = 9;
static const int OP_POW = 10;
static const int OP_LOOSE_POW = 11;
static const int OP_MAX = 12;
static const int OP_MIN = 13;
static const int OP_LT  = 14;
static const int OP_GT  = 15;
static const int OP_LE  = 16;
static const int OP_GE  = 17;
static const int OP_TAN = 18;
static const int OP_SINH= 19;
static const int OP_COSH= 20;
static const int OP_TANH= 21;
static const int OP_LOOSE_LOG = 22;
static const int OP_INV = 23;
static const int OP_LOOSE_INV = 24;
static const int OP_NEG = 25;
static const int OP_ABS = 26;
static const int OP_SQRT= 27;
static const int OP_LOOSE_SQRT = 28;
static const int OP_IF  = 29; // ternary

__host__ __device__ inline float madd(float a, float b, float c) {
#ifdef __CUDA_ARCH__
    return __fmaf_rn(a, b, c);
#else
    return fmaf(a, b, c);
#endif
}

// ------------ Batched GPU evaluator (configurable grid/block) ------------

// Forward declarations for helpers used inside kernels defined below
__host__ __device__ inline int op_arity(int op);
__host__ __device__ inline float apply_op(int op, float a, float b);
__host__ __device__ inline float safe_div(float a, float b);

#ifndef MAX_EVAL_STACK
#define MAX_EVAL_STACK 128
#endif

__global__ void eval_prefix_kernel_batch(const int* __restrict__ tokens,
                                         const float* __restrict__ values,
                                         const float* __restrict__ X, // [dataPoints, num_features]
                                         int len,
                                         int num_features,
                                         int dataPoints,
                                         float* __restrict__ out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dataPoints) return;
    // Guard supported lengths
    if (len > MAX_EVAL_STACK) {
        if (idx == 0) {
            // best-effort: write NaN to indicate overflow
            out[0] = NAN;
        }
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
                float r; int out_t = 0; float pa = 0.0f, pb = 0.0f;
                if (t == OP_ADD) {
                    if (a_t == 1) r = madd(a_pa, a_pb, b_v);
                    else if (b_t == 1) r = madd(b_pa, b_pb, a_v);
                    else r = a_v + b_v;
                } else if (t == OP_SUB) {
                    if (a_t == 1) r = madd(a_pa, a_pb, -b_v);
                    else if (b_t == 1) r = madd(-b_pa, b_pb, a_v);
                    else r = a_v - b_v;
                } else if (t == OP_MUL) {
                    if (a_t == 1 && b_t == 1) { pa = a_pa * a_pb; pb = b_pa * b_pb; out_t = 1; r = (pa * pb); }
                    else if (a_t == 1)       { pa = a_pa; pb = a_pb * b_v; out_t = 1; r = (pa * pb); }
                    else if (b_t == 1)       { pa = a_v * b_pa; pb = b_pb; out_t = 1; r = (pa * pb); }
                    else                      { pa = a_v; pb = b_v; out_t = 1; r = (pa * pb); }
                } else if (t == OP_DIV) {
                    r = safe_div(a_v, b_v);
                } else {
                    r = apply_op(t, a_v, b_v);
                }
                s_val[sp] = r; s_tag[sp] = out_t; s_pa[sp] = pa; s_pb[sp] = pb; ++sp;
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
                                      const float* X, // device pointer [dataPoints, num_features]
                                      int len,
                                      int num_features,
                                      int dataPoints,
                                      float* out_dev, // device pointer [dataPoints]
                                      int blocks,
                                      int threads) {
    if (threads <= 0) threads = 256;
    if (blocks  <= 0) blocks  = (dataPoints + threads - 1) / threads;
    eval_prefix_kernel_batch<<<blocks, threads>>>(tokens, values, X, len, num_features, dataPoints, out_dev);
}

__host__ __device__ inline int op_arity(int op) {
    // ternary IF
    if (op == OP_IF) return 3;
    // unary ops
    if (op == OP_SIN || op == OP_COS || op == OP_EXP || op == OP_LOG ||
        op == OP_TAN || op == OP_SINH || op == OP_COSH || op == OP_TANH ||
        op == OP_LOOSE_LOG || op == OP_INV || op == OP_LOOSE_INV ||
        op == OP_NEG || op == OP_ABS || op == OP_SQRT || op == OP_LOOSE_SQRT)
        return 1;
    // binary default
    return 2;
}

static const float DELTA = 1e-9f;
static const float MAX_VAL = 1e9f;

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
        // binary
        case OP_ADD: return a + b;
        case OP_SUB: return a - b;
        case OP_MUL: return a * b;
        case OP_DIV: return safe_div(a, b);
        case OP_LOOSE_DIV: return loose_div(a, b);
        case OP_POW: return powf(a, b);
        case OP_LOOSE_POW: return powf(fabsf(a), b);
        case OP_MAX: return a >= b ? a : b;
        case OP_MIN: return a <= b ? a : b;
        case OP_LT:  return a <  b ? 1.0f : -1.0f;
        case OP_GT:  return a >  b ? 1.0f : -1.0f;
        case OP_LE:  return a <= b ? 1.0f : -1.0f;
        case OP_GE:  return a >= b ? 1.0f : -1.0f;
        // unary
        case OP_SIN: return sinf(a);
        case OP_COS: return cosf(a);
        case OP_TAN: return tanf(a);
        case OP_SINH: return sinhf(a);
        case OP_COSH: return coshf(a);
        case OP_TANH: return tanhf(a);
        case OP_LOG: return safe_log(a);
        case OP_LOOSE_LOG: return loose_log(a);
        case OP_EXP: return expf(a);
        case OP_INV: return safe_inv(a);
        case OP_LOOSE_INV: return loose_inv(a);
        case OP_NEG: return -a;
        case OP_ABS: return fabsf(a);
        case OP_SQRT: return sqrtf(a);
        case OP_LOOSE_SQRT: return sqrtf(fabsf(a));
        default: return 0.0f;
    }
}

struct TagVal {
    float v;
    int tag;    // 0 = plain value, 1 = v == pa * pb
    float pa;
    float pb;
};

static inline float eval_prefix_cpu_impl(const int* tokens,
                                         const float* values,
                                         const float* x,
                                         int len,
                                         int num_features) {
    std::vector<TagVal> stack;
    stack.reserve(len);

    for (int i = len - 1; i >= 0; --i) {
        const int t = tokens[i];
        if (t == TOK_CONST) {
            stack.push_back({values[i], 0, 0.0f, 0.0f});
        } else if (t == TOK_VAR) {
            int idx = (int)values[i];
            float v = 0.0f;
            if (idx >= 0 && idx < num_features) v = x[idx];
            stack.push_back({v, 0, 0.0f, 0.0f});
        } else {
            const int ar = op_arity(t);
            if (ar == 1) {
                TagVal a = stack.back(); stack.pop_back();
                float r = apply_op(t, a.v, 0.0f);
                stack.push_back({r, 0, 0.0f, 0.0f});
            } else if (ar == 2) {
                TagVal a = stack.back(); stack.pop_back();
                TagVal b = stack.back(); stack.pop_back();
                float r;
                int out_tag = 0;
                float pa = 0.0f, pb = 0.0f;
                if (t == OP_ADD) {
                    // recognizes patterns like (a*b) + c and use fused multiply add 
                    if (a.tag == 1) r = madd(a.pa, a.pb, b.v);
                    else if (b.tag == 1) r = madd(b.pa, b.pb, a.v);
                    else r = a.v + b.v;
                } else if (t == OP_SUB) {
                    if (a.tag == 1) r = madd(a.pa, a.pb, -b.v);
                    else if (b.tag == 1) r = madd(-b.pa, b.pb, a.v);
                    else r = a.v - b.v;
                } else if (t == OP_MUL) {
                    if (a.tag == 1 && b.tag == 1) {
                        pa = a.pa * a.pb;
                        pb = b.pa * b.pb;
                        out_tag = 1;
                        r = (pa * pb);
                    } else if (a.tag == 1) {
                        pa = a.pa;
                        pb = a.pb * b.v;
                        out_tag = 1;
                        r = (pa * pb);
                    } else if (b.tag == 1) {
                        pa = a.v * b.pa;
                        pb = b.pb;
                        out_tag = 1;
                        r = (pa * pb);
                    } else {
                        pa = a.v;
                        pb = b.v;
                        out_tag = 1;
                        r = (pa * pb);
                    }
                } else if (t == OP_DIV) {
                    r = safe_div(a.v, b.v);
                } else {
                    r = apply_op(t, a.v, b.v);
                }
                stack.push_back({r, out_tag, pa, pb});
            } else {
                // ternary (IF)
                TagVal a = stack.back(); stack.pop_back(); // cond
                TagVal b = stack.back(); stack.pop_back(); // then
                TagVal c = stack.back(); stack.pop_back(); // else
                float r = (a.v > 0.0f) ? b.v : c.v;
                stack.push_back({r, 0, 0.0f, 0.0f});
            }
        }
    }
    return stack.empty() ? 0.0f : stack.back().v;
}

extern "C" float eval_tree_cpu(const int* tokens,
                                const float* values,
                                const float* features,
                                int len,
                                int num_features) {
    return eval_prefix_cpu_impl(tokens, values, features, len, num_features);
}

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

    // examples
    // tokens : [ADD(1), MUL(3), VAR(-1), VAR(-1), VAR(-1)] 
    // values: [0, 0, 0, 1, 2]
    // x    : [1.5, 2.0, 0.25]

    for (int i = len - 1; i >= 0; --i) {
        const int t = tokens[i];
        if (t == TOK_CONST) {
            s_val[sp] = values[i];
            s_tag[sp] = 0;
            s_pa[sp] = 0.0f; s_pb[sp] = 0.0f;
            ++sp;
        } else if (t == TOK_VAR) {
            int idx = (int)values[i];
            float v = 0.0f;
            if (idx >= 0 && idx < num_features) v = x[idx];
            s_val[sp] = v;
            s_tag[sp] = 0;
            s_pa[sp] = 0.0f; s_pb[sp] = 0.0f;
            ++sp;
        } else {
            const int ar = op_arity(t);
            if (ar == 1) {
                float a = s_val[--sp];
                float r = apply_op(t, a, 0.0f);
                s_val[sp] = r;
                s_tag[sp] = 0; s_pa[sp] = 0.0f; s_pb[sp] = 0.0f;
                ++sp;
            } else if (ar == 2) {
                float a_v = s_val[--sp];
                int   a_t = s_tag[sp];
                float a_pa = s_pa[sp];
                float a_pb = s_pb[sp];
                float b_v = s_val[--sp];
                int   b_t = s_tag[sp];
                float b_pa = s_pa[sp];
                float b_pb = s_pb[sp];

                float r; int out_t = 0; float pa = 0.0f, pb = 0.0f;
                if (t == OP_ADD) {
                    if (a_t == 1) r = madd(a_pa, a_pb, b_v);
                    else if (b_t == 1) r = madd(b_pa, b_pb, a_v);
                    else r = a_v + b_v;
                } else if (t == OP_SUB) {
                    if (a_t == 1) r = madd(a_pa, a_pb, -b_v);
                    else if (b_t == 1) r = madd(-b_pa, b_pb, a_v);
                    else r = a_v - b_v;
                } else if (t == OP_MUL) {
                    if (a_t == 1 && b_t == 1) {
                        pa = a_pa * a_pb; pb = b_pa * b_pb; out_t = 1; r = (pa * pb);
                    } else if (a_t == 1) {
                        pa = a_pa; pb = a_pb * b_v; out_t = 1; r = (pa * pb);
                    } else if (b_t == 1) {
                        pa = a_v * b_pa; pb = b_pb; out_t = 1; r = (pa * pb);
                    } else {
                        pa = a_v; pb = b_v; out_t = 1; r = (pa * pb);
                    }
                } else if (t == OP_DIV) {
                    r = safe_div(a_v, b_v);
                } else {
                    r = apply_op(t, a_v, b_v);
                }
                s_val[sp] = r;
                s_tag[sp] = out_t;
                s_pa[sp]  = pa;
                s_pb[sp]  = pb;
                ++sp;
            } else {
                // ternary (IF)
                float cond = s_val[--sp];
                float then_v = s_val[--sp];
                float else_v = s_val[--sp];
                float r = (cond > 0.0f) ? then_v : else_v;
                s_val[sp] = r;
                s_tag[sp] = 0; s_pa[sp] = 0.0f; s_pb[sp] = 0.0f;
                ++sp;
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
