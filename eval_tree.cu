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
// 1=ADD, 2=SUB, 3=MUL, 4=DIV, 5=SIN, 6=COS, 7=EXP, 8=LOG

static const int TOK_CONST = 0;
static const int TOK_VAR   = -1;

__host__ __device__ inline float madd(float a, float b, float c) {
#ifdef __CUDA_ARCH__
    return __fmaf_rn(a, b, c);
#else
    return fmaf(a, b, c);
#endif
}

__host__ __device__ inline int op_arity(int op) {
    if (op == 5 || op == 6 || op == 7 || op == 8) return 1; // unary
    return 2; // binary for 1..4
}

__host__ __device__ inline float safe_div(float a, float b) {
    const float eps = 1e-12f;
    float denom = fabsf(b) < eps ? (b < 0 ? -eps : eps) : b;
    return a / denom;
}

__host__ __device__ inline float safe_log(float a) {
    const float eps = 1e-12f;
    return logf(fabsf(a) + eps);
}

__host__ __device__ inline float apply_op(int op, float a, float b) {
    switch (op) {
        case 1: return a + b; // ADD
        case 2: return a - b; // SUB
        case 3: return a * b; // MUL
        case 4: return safe_div(a, b); // DIV
        case 5: return sinf(a); // SIN (b ignored)
        case 6: return cosf(a); // COS (b ignored)
        case 7: return expf(a); // EXP (b ignored)
        case 8: return safe_log(a); // LOG (b ignored)
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
            } else {
                TagVal a = stack.back(); stack.pop_back();
                TagVal b = stack.back(); stack.pop_back();
                float r;
                int out_tag = 0;
                float pa = 0.0f, pb = 0.0f;
                if (t == 1) {
                    // recognizes patterns like (a*b) + c and use fused multiply add 
                    if (a.tag == 1) r = madd(a.pa, a.pb, b.v);
                    else if (b.tag == 1) r = madd(b.pa, b.pb, a.v);
                    else r = a.v + b.v;
                } else if (t == 2) {
                    if (a.tag == 1) r = madd(a.pa, a.pb, -b.v);
                    else if (b.tag == 1) r = madd(-b.pa, b.pb, a.v);
                    else r = a.v - b.v;
                } else if (t == 3) {
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
                } else if (t == 4) {
                    r = safe_div(a.v, b.v);
                } else {
                    r = apply_op(t, a.v, b.v);
                }
                stack.push_back({r, out_tag, pa, pb});
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
            } else {
                float a_v = s_val[--sp];
                int   a_t = s_tag[sp];
                float a_pa = s_pa[sp];
                float a_pb = s_pb[sp];
                float b_v = s_val[--sp];
                int   b_t = s_tag[sp];
                float b_pa = s_pa[sp];
                float b_pb = s_pb[sp];

                float r; int out_t = 0; float pa = 0.0f, pb = 0.0f;
                if (t == 1) {
                    if (a_t == 1) r = madd(a_pa, a_pb, b_v);
                    else if (b_t == 1) r = madd(b_pa, b_pb, a_v);
                    else r = a_v + b_v;
                } else if (t == 2) {
                    if (a_t == 1) r = madd(a_pa, a_pb, -b_v);
                    else if (b_t == 1) r = madd(-b_pa, b_pb, a_v);
                    else r = a_v - b_v;
                } else if (t == 3) {
                    if (a_t == 1 && b_t == 1) {
                        pa = a_pa * a_pb; pb = b_pa * b_pb; out_t = 1; r = (pa * pb);
                    } else if (a_t == 1) {
                        pa = a_pa; pb = a_pb * b_v; out_t = 1; r = (pa * pb);
                    } else if (b_t == 1) {
                        pa = a_v * b_pa; pb = b_pb; out_t = 1; r = (pa * pb);
                    } else {
                        pa = a_v; pb = b_v; out_t = 1; r = (pa * pb);
                    }
                } else if (t == 4) {
                    r = safe_div(a_v, b_v);
                } else {
                    r = apply_op(t, a_v, b_v);
                }
                s_val[sp] = r;
                s_tag[sp] = out_t;
                s_pa[sp]  = pa;
                s_pb[sp]  = pb;
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
