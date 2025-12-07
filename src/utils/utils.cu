// eval_tree.cu
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <cmath>

// ---- Low-level async copy helpers (Ampere+) ----

// #ifndef MAX_EVAL_STACK
// #define MAX_EVAL_STACK 60
// #endif
constexpr int MAX_EVAL_STACK = 60;

#ifndef MAX_NUM_FEATURES
#define MAX_NUM_FEATURES 16
#endif

// Threads per block (we'll actually use 128 as you requested)
#ifndef EVAL_MAX_THREADS_PER_BLOCK
#define EVAL_MAX_THREADS_PER_BLOCK 128
#endif

// Fixed number of blocks we use to partition [0, dataPoints)
#ifndef EVAL_NUM_BLOCKS
#define EVAL_NUM_BLOCKS 48
#endif

__host__ __device__ __forceinline__
int ceil_div_int(int a, int b) {
    // assumes a >= 0, b > 0
    return (a + b - 1) / b;
}

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

__device__ inline void static_stack_push(float *stk, float val, int &sp)
{
    // stk[sp] = val;
    // sp++;
    #pragma unroll
    for(int i=MAX_EVAL_STACK-2; i>=0; i--)
        stk[i+1] = stk[i];
    stk[0] = val;
}

__device__ inline float static_stack_pop(float *stk, int &sp)
{
    // sp--;
    // return stk[sp];
    float val = stk[0];
    #pragma unroll
    for(int i=0; i<=MAX_EVAL_STACK-2; i++)
        stk[i] = stk[i+1];
    return val;
}

// 32-element register-backed top-of-stack cache.
// Implemented as a small fixed-size array so the compiler can scalarize to registers.
// struct TopCache32 {
//     float data[32];
//     int   tc;  // how many live in data[0..tc-1]

//     __device__ __forceinline__ void clear() { tc = 0; }
// };

__device__ __forceinline__
void push_cached(float v,
                 float* __restrict__ stack,
                 int& sp,          // backing-stack pointer
                 float* __restrict__ cache, // size 32
                 int* __restrict__ tc_ptr)  // how many live in cache[0..tc-1]
{
    int tc = *tc_ptr;
    if (tc == 32) {
        // Spill deepest cached element (logical bottom of the cache)
        stack[sp] = cache[0];
        ++sp;
        // Shift cache down: cache[i] = cache[i+1] for i=0..30
        #pragma unroll 
        for (int i = 0; i < 31; ++i) {
            cache[i] = cache[i + 1];
        }
        cache[31] = v;
    } else {
        // Just grow cache: shift live prefix down one slot
        #pragma unroll 
        for (int i = 0; i < 32; ++i) {
            if (i == tc) { 
                cache[i] = v;
                break;
            }
        }
        ++tc;
        *tc_ptr = tc;
    }
}

__device__ __forceinline__
float pop_cached(float* __restrict__ stack,
                 int& sp,
                 float* __restrict__ cache,
                 int* __restrict__ tc_ptr)
{
    int tc = *tc_ptr;
    // Pop from cache if we have any cached values
    if (tc > 0) {
        float v = 0.0; 
        #pragma unroll 
        for (int i = 0; i < 32; ++i) {
            if (i + 1 == tc) {
                v = cache[i];
                break; 
            }
        }
        --tc;
        *tc_ptr = tc;
        return v;
    }

    // No cached values, fall back to backing stack
    float v = 0.0f;
    if (sp > 0) {
        --sp;
        v = stack[sp];
    }
    return v;
}

// struct RegCache12 {
//     float c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11;

//     // Push with backing stack + spill when full.
//     __device__ __forceinline__ void push(float v,
//                                          float* __restrict__ stack,
//                                          int& sp, 
//                                          int& tc)
//     {
//         if (tc == 12) {
//             // Spill the *bottom* (logical oldest) value.
//             stack[sp++] = c0;

//             // Shift the cache down: c0<-c1, ..., c10<-c11, c11<-v
//             c0  = c1;   c1  = c2;   c2  = c3;   c3  = c4;
//             c4  = c5;   c5  = c6;   c6  = c7;   c7  = c8;
//             c8  = c9;   c9  = c10;  c10 = c11;  c11 = v;
//             // tc stays 12
//             return;
//         }

//         // tc < 12: put v at index tc, then increment tc
//         switch (tc) {
//             case  0: c0  = v; break; 
//             case  1: c1  = v; break;
//             case  2: c2  = v; break; 
//             case  3: c3  = v; break;
//             case  4: c4  = v; break; 
//             case  5: c5  = v; break;
//             case  6: c6  = v; break; 
//             case  7: c7  = v; break;
//             case  8: c8  = v; break; 
//             case  9: c9  = v; break;
//             case 10: c10 = v; break;
//             case 11: c11 = v; break;
//             default: break; // should not happen
//         }
//         ++tc;
//     }

//     // Pop returns top of cache if available; else read from backing stack.
//     __device__ __forceinline__ float pop(float* __restrict__ stack,
//                                          int& sp,
//                                          int& tc)
//     {
//         if (tc > 0) {
//             --tc;
//             switch (tc) {
//                 case  0: return c0;   
//                 case  1: return c1;
//                 case  2: return c2;   
//                 case  3: return c3;
//                 case  4: return c4;   
//                 case  5: return c5;
//                 case  6: return c6;   
//                 case  7: return c7;
//                 case  8: return c8;   
//                 case  9: return c9;
//                 case 10: return c10;
//                 case 11: return c11;
//                 default: return 0.0f; // shouldn't happen
//             }
//         }

//         // Fallback: use backing stack if cache is empty.
//         if (sp > 0) {
//             --sp;
//             return stack[sp];
//         }
//         return 0.0f;
//     }
// };

// struct RegCache6 {
//     float c0, c1, c2, c3, c4, c5;

//     // Push with backing stack + spill when full.
//     __device__ __forceinline__ void push(float v,
//                                          float* __restrict__ stack,
//                                          int& sp, 
//                                          int& tc)
//     {
//         if (tc == 6) {
//             // Spill the *bottom* (logical oldest) value.
//             stack[sp++] = c0;

//             // Shift the cache down: c0<-c1, ..., c4<-c5, c5<-v
//             c0 = c1;   c1 = c2;   c2 = c3;
//             c3 = c4;   c4 = c5;   c5 = v;
//             // tc stays 6
//             return;
//         }

//         // tc < 6: put v at index tc, then increment tc
//         switch (tc) {
//             case 0: c0 = v; break;
//             case 1: c1 = v; break;
//             case 2: c2 = v; break;
//             case 3: c3 = v; break;
//             case 4: c4 = v; break;
//             case 5: c5 = v; break;
//             default: break; // should not happen
//         }
//         ++tc;
//     }

//     // Pop returns top of cache if available; else read from backing stack.
//     __device__ __forceinline__ float pop(float* __restrict__ stack,
//                                          int& sp,
//                                          int& tc)
//     {
//         if (tc > 0) {
//             --tc;
//             switch (tc) {
//                 case 0: return c0;
//                 case 1: return c1;
//                 case 2: return c2;
//                 case 3: return c3;
//                 case 4: return c4;
//                 case 5: return c5;
//                 default: return 0.0f; // shouldn't happen
//             }
//         }

//         // Fallback: use backing stack if cache is empty.
//         if (sp > 0) {
//             --sp;
//             return stack[sp];
//         }
//         return 0.0f;
//     }
// };

struct RegCache3 {
    float c0, c1, c2;

    // Push with backing stack + spill when full.
    __device__ __forceinline__ void push(float v,
                                         float* __restrict__ stack,
                                         int& sp, 
                                         int& tc)
    {
        if (tc == 3) {
            // Spill the *bottom* (logical oldest) value.
            stack[sp++] = c0;

            // Shift the cache down: c0<-c1, c1<-c2, c2<-v
            c0 = c1;
            c1 = c2;
            c2 = v;
            // tc stays 3
            return;
        }

        // tc < 3: put v at index tc, then increment tc
        switch (tc) {
            case 0: c0 = v; break;
            case 1: c1 = v; break;
            case 2: c2 = v; break;
            default: break; // should not happen
        }
        ++tc;
    }

    // Pop returns top of cache if available; else read from backing stack.
    __device__ __forceinline__ float pop(float* __restrict__ stack,
                                         int& sp,
                                         int& tc)
    {
        if (tc > 0) {
            --tc;
            switch (tc) {
                case 0: return c0;
                case 1: return c1;
                case 2: return c2;
                default: return 0.0f; // shouldn't happen
            }
        }

        // Fallback: use backing stack if cache is empty.
        if (sp > 0) {
            --sp;
            return stack[sp];
        }
        return 0.0f;
    }
};


__launch_bounds__(128, 12)
// 16
__global__ void eval_prefix_kernel_batch(const int* __restrict__ tokens,
                                         const float* __restrict__ values,
                                         const float* __restrict__ X, // [dataPoints, num_features]
                                         int len,
                                         int num_features,
                                         int dataPoints,
                                         float* s_val_old,
                                         float* __restrict__ out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dataPoints) return;
    
    float stack[MAX_EVAL_STACK];   // backing stack (likely local memory)
    int   sp = 0;
    int   tc = 0;

    // RegCache6 cache;
    // RegCache3 cache;

    const float* x = X + (size_t)idx * (size_t)num_features;
    // float r;

    // for (int i = len - 1; i >= 0; --i) {
    //     const int t = tokens[i];
    //     if (t == TOK_CONST) {
    //         cache.push(values[i], stack, sp, tc);
    //     } else if (t == TOK_VAR) {
    //         int fidx = (int)values[i];
    //         cache.push((fidx >= 0 && fidx < num_features) ? x[fidx] : 0.0f, stack, sp, tc);
    //     } else {
    //         const int ar = op_arity(t);
    //         if (ar == 1) {
    //             cache.push(apply_op(t, cache.pop(stack, sp, tc), 0.0f), stack, sp, tc);
    //         } else if (ar == 2) {
    //             float a_v = cache.pop(stack, sp, tc);
    //             float b_v = cache.pop(stack, sp, tc);
    //             if (t == OP_ADD)      r = a_v + b_v;
    //             else if (t == OP_SUB) r = a_v - b_v;
    //             else if (t == OP_MUL) r = a_v * b_v;
    //             else if (t == OP_DIV) r = a_v / b_v;
    //             else                  r = apply_op(t, a_v, b_v);
    //             cache.push(r, stack, sp, tc);
    //         } else { // ternary IF
    //             float c = cache.pop(stack, sp, tc);
    //             float b = cache.pop(stack, sp, tc);
    //             float a = cache.pop(stack, sp, tc);
    //             r = (a > 0.0f) ? b : c;
    //             cache.push(r, stack, sp, tc);
    //         }
    //     }
    // }

    // // If expression evaluates to a single value without extra backing entries:
    // // float result = 0.0f;
    // out[idx] = (tc == 1 && sp == 0) ? cache.pop(stack, sp, tc) : 0.0f;

    for (int i = len - 1; i >= 0; --i) {
        const int t = tokens[i];
        if (t == TOK_CONST) {
            stack[sp] = values[i]; ++sp;
        } else if (t == TOK_VAR) {
            int fidx = (int)values[i];
            float v = (fidx >= 0 && fidx < num_features) ? x[fidx] : 0.0f;
            stack[sp] = v; ++sp;
        } else {
            const int ar = op_arity(t);
            if (ar == 1) {
                float a = stack[--sp];
                float r = apply_op(t, a, 0.0f);
                stack[sp] = r; ++sp;
            } else if (ar == 2) {
                float a_v = stack[--sp];
                float b_v = stack[--sp];
                float r;
                if (t == OP_ADD)      r = a_v + b_v;
                else if (t == OP_SUB) r = a_v - b_v;
                else if (t == OP_MUL) r = a_v * b_v;
                else if (t == OP_DIV) r = a_v / b_v;
                else                  r = apply_op(t, a_v, b_v);
                stack[sp] = r; ++sp;
            } else { // ternary IF
                float c = stack[--sp];
                float b = stack[--sp];
                float a = stack[--sp];
                float r = (a > 0.0f) ? b : c;
                stack[sp] = r; ++sp;
            }
        }
    }
    out[idx] = (sp > 0) ? stack[sp - 1] : 0.0f;
}

// ---- Batched (datapoint-parallel) evaluator ----

// Register-backed stack with 60 float slots (no dynamic indexing).
struct RegStack60 {
  // 60 scalars => eligible for registers
  float s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,
        s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,
        s20,s21,s22,s23,s24,s25,s26,s27,s28,s29,
        s30,s31,s32,s33,s34,s35,s36,s37,s38,s39,
        s40,s41,s42,s43,s44,s45,s46,s47,s48,s49,
        s50,s51,s52,s53,s54,s55,s56,s57,s58,s59;
  int sp;

  __device__ __forceinline__ void clear() { sp = 0; }

  __device__ __forceinline__ void push(float v) {
    switch (sp) {
      case  0: s0=v; break;  case  1: s1=v; break;  case  2: s2=v; break;  case  3: s3=v; break;
      case  4: s4=v; break;  case  5: s5=v; break;  case  6: s6=v; break;  case  7: s7=v; break;
      case  8: s8=v; break;  case  9: s9=v; break;  case 10: s10=v; break; case 11: s11=v; break;
      case 12: s12=v; break; case 13: s13=v; break; case 14: s14=v; break; case 15: s15=v; break;
      case 16: s16=v; break; case 17: s17=v; break; case 18: s18=v; break; case 19: s19=v; break;
      case 20: s20=v; break; case 21: s21=v; break; case 22: s22=v; break; case 23: s23=v; break;
      case 24: s24=v; break; case 25: s25=v; break; case 26: s26=v; break; case 27: s27=v; break;
      case 28: s28=v; break; case 29: s29=v; break; case 30: s30=v; break; case 31: s31=v; break;
      case 32: s32=v; break; case 33: s33=v; break; case 34: s34=v; break; case 35: s35=v; break;
      case 36: s36=v; break; case 37: s37=v; break; case 38: s38=v; break; case 39: s39=v; break;
      case 40: s40=v; break; case 41: s41=v; break; case 42: s42=v; break; case 43: s43=v; break;
      case 44: s44=v; break; case 45: s45=v; break; case 46: s46=v; break; case 47: s47=v; break;
      case 48: s48=v; break; case 49: s49=v; break; case 50: s50=v; break; case 51: s51=v; break;
      case 52: s52=v; break; case 53: s53=v; break; case 54: s54=v; break; case 55: s55=v; break;
      case 56: s56=v; break; case 57: s57=v; break; case 58: s58=v; break; case 59: s59=v; break;
      default: break; // ignore overflow if expression is malformed
    }
    ++sp;
  }

  __device__ __forceinline__ float pop() {
    --sp;
    switch (sp) {
      case  0: return s0;   case  1: return s1;   case  2: return s2;   case  3: return s3;
      case  4: return s4;   case  5: return s5;   case  6: return s6;   case  7: return s7;
      case  8: return s8;   case  9: return s9;   case 10: return s10;  case 11: return s11;
      case 12: return s12;  case 13: return s13;  case 14: return s14;  case 15: return s15;
      case 16: return s16;  case 17: return s17;  case 18: return s18;  case 19: return s19;
      case 20: return s20;  case 21: return s21;  case 22: return s22;  case 23: return s23;
      case 24: return s24;  case 25: return s25;  case 26: return s26;  case 27: return s27;
      case 28: return s28;  case 29: return s29;  case 30: return s30;  case 31: return s31;
      case 32: return s32;  case 33: return s33;  case 34: return s34;  case 35: return s35;
      case 36: return s36;  case 37: return s37;  case 38: return s38;  case 39: return s39;
      case 40: return s40;  case 41: return s41;  case 42: return s42;  case 43: return s43;
      case 44: return s44;  case 45: return s45;  case 46: return s46;  case 47: return s47;
      case 48: return s48;  case 49: return s49;  case 50: return s50;  case 51: return s51;
      case 52: return s52;  case 53: return s53;  case 54: return s54;  case 55: return s55;
      case 56: return s56;  case 57: return s57;  case 58: return s58;  case 59: return s59;
      default: return 0.f;  // underflow guard for malformed expressions
    }
  }
};

// ---- Batched (datapoint-parallel) evaluator ----
// Keep your existing op_arity(), apply_op(), token enums, etc.
// __launch_bounds__(128, 3)  
// __global__ void eval_prefix_kernel_batch(const int* __restrict__ tokens,
//                                          const float* __restrict__ values,
//                                          const float* __restrict__ X, // [dataPoints, num_features]
//                                          int len,
//                                          int num_features,
//                                          int dataPoints,
//                                          float* __restrict__ out)
// {
//     const int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= dataPoints) return;

//     const float* x = X + (size_t)idx * (size_t)num_features;

//     RegStack60 st; 
//     st.clear();

//     // Evaluate one prefix expression for datapoint idx
//     // (optional) unroll modestly if len has a small fixed upper bound:
//     for (int i = len - 1; i >= 0; --i) {
//         const int t = tokens[i];
//         if (t == TOK_CONST) {
//             st.push(values[i]);
//         } else if (t == TOK_VAR) {
//             int fidx = (int)values[i];
//             float v = (fidx >= 0 && fidx < num_features) ? x[fidx] : 0.0f;
//             st.push(v);
//         } else {
//             const int ar = op_arity(t);
//             if (ar == 1) {
//                 float a = st.pop();
//                 float r = apply_op(t, a, 0.0f);
//                 st.push(r);
//             } else if (ar == 2) {
//                 float a_v = st.pop();
//                 float b_v = st.pop();
//                 float r;
//                 if (t == OP_ADD)      r = a_v + b_v;
//                 else if (t == OP_SUB) r = a_v - b_v;
//                 else if (t == OP_MUL) r = a_v * b_v;
//                 else if (t == OP_DIV) r = a_v / b_v;
//                 else                  r = apply_op(t, a_v, b_v);
//                 st.push(r);
//             } else { // ternary IF
//                 float c = st.pop();
//                 float b = st.pop();
//                 float a = st.pop();
//                 float r = (a > 0.0f) ? b : c;
//                 st.push(r);
//             }
//         }
//     }

//     float r = (st.sp > 0) ? st.pop() : 0.0f;
//     out[idx] = r;
// }

extern "C" void eval_tree_gpu_batch(const int* tokens,
                                    const float* values,
                                    const float* X,
                                    int len,
                                    int num_features,
                                    int dataPoints,
                                    float* out_dev,
                                    int blocks,
                                    int threads,
                                    float* s_val) {
    // if (threads <= 0) threads = 256;
    // if (blocks  <= 0) blocks  = (dataPoints + threads - 1) / threads;
    // datapoints 
    // int shmem = 16000; 
    eval_prefix_kernel_batch<<<blocks, threads>>>(tokens, values, X, len, num_features, dataPoints, s_val, out_dev);
    // eval_prefix_kernel_batch<<<blocks, threads>>>(tokens, values, X, len, num_features, dataPoints, out_dev);
}

// staged tokens/values and per-thread stacks.
// ---- Batched (datapoint-parallel) evaluator with shared tokens/values ----

// __global__ void eval_prefix_kernel_batch_shmem(const int* __restrict__ tokens,
//                                                const float* __restrict__ values,
//                                                const float* __restrict__ X, // [dataPoints, num_features]
//                                                int len,
//                                                int num_features,
//                                                int dataPoints,
//                                                float* __restrict__ out) {
//     const int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= dataPoints) return;

//     // Dynamic shared memory layout (16B aligned segments):
//     // [ s_val_sh: blockDim.x * (MAX_EVAL_STACK+1) * float ]
//     extern __shared__ unsigned char smem[];

//     // Per-thread value stacks in shared memory with padded stride to avoid bank conflicts
//     const int STACK_STRIDE = MAX_EVAL_STACK + 1;
//     float* s_val_sh = reinterpret_cast<float*>(smem);
//     float* s_val    = s_val_sh + (size_t)threadIdx.x * (size_t)STACK_STRIDE;

//     const float* x = X + (size_t)idx * (size_t)num_features;

//     int sp = 0;

//     // Evaluate one prefix expression for datapoint idx
//     for (int i = len - 1; i >= 0; --i) {
//         const int t = tokens[i];
//         if (t == TOK_CONST) {
//             s_val[sp++] = values[i];
//         } else if (t == TOK_VAR) {
//             int fidx = (int)values[i];
//             float v = (fidx >= 0 && fidx < num_features) ? x[fidx] : 0.0f;
//             s_val[sp++] = v;
//         } else {
//             const int ar = op_arity(t);
//             if (ar == 1) {
//                 float a = s_val[--sp];
//                 s_val[sp++] = apply_op(t, a, 0.0f);
//             } else if (ar == 2) {
//                 float a_v = s_val[--sp];
//                 float b_v = s_val[--sp];
//                 float r;
//                 if      (t == OP_ADD) r = a_v + b_v;
//                 else if (t == OP_SUB) r = a_v - b_v;
//                 else if (t == OP_MUL) r = a_v * b_v;
//                 else if (t == OP_DIV) r = a_v / b_v;
//                 else                  r = apply_op(t, a_v, b_v);
//                 s_val[sp++] = r;
//             } else { // ternary IF
//                 float c = s_val[--sp];
//                 float b = s_val[--sp];
//                 float a = s_val[--sp];
//                 float r = (a > 0.0f) ? b : c;
//                 s_val[sp++] = r;
//             }
//         }
//     }
//     out[idx] = (sp > 0) ? s_val[sp - 1] : 0.0f;
// }


__global__ void eval_prefix_kernel_batch_shmem(const int* __restrict__ tokens,
                                               const float* __restrict__ values,
                                               const float* __restrict__ X, // [dataPoints, num_features]
                                               int len,
                                               int num_features,
                                               int dataPoints,
                                               float* __restrict__ out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // This branch is UNIFORM across the block (len is the same for all threads),
    // so it's safe to return before any __syncthreads().
    // if (len > MAX_EVAL_STACK) {
    //     if (idx == 0) out[0] = NAN;
    //     return;
    // }

    // Dynamic shared memory layout (16B aligned segments):
    // [ s_X: rows_this_block * num_features * float ]  (allocated for max threads)
    extern __shared__ unsigned char smem[];
    auto align16 = [](size_t x) { return (x + 15) & ~((size_t)15); };

    size_t off = 0;
    // X tile for this block
    const size_t x_tile_elems = (size_t)blockDim.x * (size_t)num_features;
    float* s_X = reinterpret_cast<float*>(smem + off);
    off = align16(off + x_tile_elems * sizeof(float));

    // Compute rows covered by this block and tile X into shared for coalesced loads
    const int block_start = blockIdx.x * blockDim.x;
    const int rows_this_block = max(0, min(blockDim.x, dataPoints - block_start));
    const int total_tile_elems = rows_this_block * num_features;
    for (int linear = threadIdx.x; linear < total_tile_elems; linear += blockDim.x) {
        const int r = linear / num_features;   // row within this block tile
        const int f = linear % num_features;   // feature index
        s_X[(size_t)r * (size_t)num_features + f] =
            X[(size_t)(block_start + r) * (size_t)num_features + f];
    }
    __syncthreads();

    // Call inactive threads for tail block
    if (threadIdx.x >= rows_this_block) return;

    const float* x = s_X + (size_t)threadIdx.x * (size_t)num_features;

    int sp = 0;
    float stack[MAX_EVAL_STACK];

    // Evaluate one prefix expression for datapoint idx
    for (int i = len - 1; i >= 0; --i) {
        const int t = tokens[i];
        if (t == TOK_CONST) {
            stack[sp++] = values[i];
        } else if (t == TOK_VAR) {
            int fidx = (int)values[i];
            float v = (fidx >= 0 && fidx < num_features) ? x[fidx] : 0.0f;
            stack[sp++] = v;
        } else {
            const int ar = op_arity(t);
            if (ar == 1) {
                float a = stack[--sp];
                stack[sp++] = apply_op(t, a, 0.0f);
            } else if (ar == 2) {
                float a_v = stack[--sp];
                float b_v = stack[--sp];
                float r;
                // if      (t == OP_ADD) r = a_v + b_v;
                // else if (t == OP_SUB) r = a_v - b_v;
                // else if (t == OP_MUL) r = a_v * b_v;
                // else if (t == OP_DIV) r = a_v / b_v;
                r = apply_op(t, a_v, b_v);
                stack[sp++] = r;
            } else { // ternary IF
                float c = stack[--sp];
                float b = stack[--sp];
                float a = stack[--sp];
                float r = (a > 0.0f) ? b : c;
                stack[sp++] = r;
            }
        }
    }
    out[block_start + threadIdx.x] = (sp > 0) ? stack[sp - 1] : 0.0f;
}

// extern "C" void eval_tree_gpu_batch(const int* tokens,
//                                     const float* values,
//                                     const float* X,
//                                     int len,
//                                     int num_features,
//                                     int dataPoints,
//                                     float* out_dev,
//                                     int blocks,
//                                     int threads) {
//     if (dataPoints <= 0) {
//         return;
//     }

//     // Match the style of your original launcher, but clamp to a sane max.
//     // Fix threads/blocks (you can tune threads later)
//     threads = 128;
//     blocks  = (dataPoints + threads - 1) / threads;

//     const int T = threads;

//     // Dynamic shared memory (16B alignment) must match kernel layout:
//     // [len * int] [len * float] [T * num_features * float] [T * (MAX_EVAL_STACK+1) * float]
//     auto align16 = [](size_t x) { return (x + 15) & ~((size_t)15); };
//     size_t shmem_bytes = 0;
//     // shmem_bytes = align16(shmem_bytes + (size_t)len * sizeof(int));
//     // shmem_bytes = align16(shmem_bytes + (size_t)len * sizeof(float));
//     shmem_bytes = align16(shmem_bytes + (size_t)T * (size_t)num_features * sizeof(float));
//     // shmem_bytes = align16(shmem_bytes + (size_t)T * (size_t)(MAX_EVAL_STACK + 1) * sizeof(float));

//     // printf("num_features: %d, shmem_bytes: %zu\n", num_features, shmem_bytes);

//     // Launch shared-memory kernel with dynamic shared allocation
//     eval_prefix_kernel_batch_shmem<<<blocks, T, shmem_bytes>>>(
//         tokens, values, X, len, num_features, dataPoints, out_dev);

//     // eval_prefix_kernel_batch<<<blocks, T>>>(
//     //     tokens, values, X, len, num_features, dataPoints, out_dev);

//     // print num features, shmem_bytes
//     // printf("num_features: %d, shmem_bytes: %zu\n", num_features, shmem_bytes);

//     // cudaError_t err = cudaGetLastError();
//     // if (err != cudaSuccess) {
//     //     fprintf(stderr, "eval_tree_gpu_batch launch error: %s\n",
//     //             cudaGetErrorString(err));
//     // }
// }


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


// Compute SMEM usage from the *max* values (deterministic)
// struct EvalSharedConfig {
//     static constexpr int Threads      = EVAL_MAX_THREADS_PER_BLOCK;
//     static constexpr int MaxFeatures  = MAX_NUM_FEATURES;
//     static constexpr int MaxStack     = MAX_EVAL_STACK;

//     // 2 * Threads * MaxFeatures  (buf0 + buf1)
//     // 1 * Threads * MaxStack     (s_val only)
//     static constexpr int SmemFloats =
//         2 * Threads * MaxFeatures +
//         1 * Threads * MaxStack;

//     static constexpr int SmemBytes  = SmemFloats * sizeof(float);
// };

// ---- Config (parameterized by K) ----
template<int Threads, int K>
struct EvalLaunchConfig {
    // 2 buffers * (K * Threads * MAX_NUM_FEATURES)  +  1 * (Threads * MAX_EVAL_STACK)
    static constexpr int SmemFloats =
        2 * K * Threads * MAX_NUM_FEATURES +
        1 * Threads * MAX_EVAL_STACK;
    static constexpr int SmemBytes  = SmemFloats * sizeof(float);
    static constexpr int MinBlocksPerSm = 1; // conservative
};


// using EvalCfg = EvalLaunchConfig<EVAL_MAX_THREADS_PER_BLOCK>;
// ---- compile-time config ----
// #ifndef EVAL_MAX_THREADS_PER_BLOCK
// #define EVAL_MAX_THREADS_PER_BLOCK 128   // safe now that only s_val is in SMEM
// #endif

// #ifndef MAX_EVAL_STACK
// #define MAX_EVAL_STACK 52
// #endif

// #ifndef MAX_NUM_FEATURES
// #define MAX_NUM_FEATURES 16
// #endif

// Choose how many tiles (of size THREADS) each block processes per stage.
// static constexpr int K_CT = 4;   // reduce grouping to raise occupancy

// -----------------------------------------------------------------------------
// K-grouped async kernel (unchanged math; only s_val is stored in SMEM)
// -----------------------------------------------------------------------------
// ---- K-grouped async kernel (values-only stack in SMEM) ----
template<int THREADS, int K>
__global__ void eval_async_kernel(
    const int*   __restrict__ tokens,
    const float* __restrict__ values,
    const float* __restrict__ X,       // [dataPoints, num_features]
    int len,
    int num_features,
    int dataPoints,
    float* __restrict__ out)
{
    if (len > MAX_EVAL_STACK) { if (blockIdx.x == 0 && threadIdx.x == 0) out[0] = NAN; return; }
    if (len <= 0 || num_features <= 0 || num_features > MAX_NUM_FEATURES || dataPoints <= 0) return;

    const int tid  = threadIdx.x;
    const int tile = blockDim.x;            // == THREADS
    const int rows_per_block = K * tile;    // K subtiles per stage

    // Per-block contiguous span
    const int t0    = blockIdx.x * rows_per_block;
    const int t_end = min(dataPoints, t0 + rows_per_block);
    if (t0 >= t_end) return;

    // Shared memory layout:
    // [buf0: K*THREADS * num_features]
    // [buf1: K*THREADS * num_features]
    // [s_val: THREADS    * MAX_EVAL_STACK]
    extern __shared__ float shmem[];
    const size_t tile_elems  = (size_t)THREADS * (size_t)num_features;
    const size_t group_elems = (size_t)K * tile_elems;

    float* buf0     = shmem;
    float* buf1     = buf0 + group_elems;
    float* s_val_sh = buf1 + group_elems;

    float* s_val = s_val_sh + (size_t)tid * (size_t)MAX_EVAL_STACK;

    // Copy rows [group_start, group_end) into dst_buf as K back-to-back subtiles
    auto copy_group = [&](int group_start, int group_end, float* dst_buf) {
        const int rows = max(0, group_end - group_start);
        const int total_elems = rows * num_features;

    #if __CUDA_ARCH__ >= 800
        const int total_elems4 = (total_elems / 4) * 4;
        for (int linear4 = tid * 4; linear4 < total_elems4; linear4 += THREADS * 4) {
            const int row_off = linear4 / num_features;
            const int fidx    = linear4 % num_features;
            const float* gptr = X + (size_t)(group_start + row_off) * (size_t)num_features + fidx;
            float*       sptr = dst_buf + (size_t)row_off * (size_t)num_features + fidx;
            if ((((uintptr_t)gptr | (uintptr_t)sptr) & 0xF) == 0 && (num_features - fidx) >= 4) {
                cp_async4(sptr, gptr);
            } else {
            #pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    const int linear = linear4 + kk;
                    if (linear >= total_elems) break;
                    const int row_k = linear / num_features;
                    const int col_k = linear % num_features;
                    cp_async<4>(
                        dst_buf + (size_t)row_k * (size_t)num_features + col_k,
                        X + (size_t)(group_start + row_k) * (size_t)num_features + col_k
                    );
                }
            }
        }
        for (int linear = total_elems4 + tid; linear < total_elems; linear += THREADS) {
            const int row_off = linear / num_features;
            const int fidx    = linear % num_features;
            cp_async<4>(
                dst_buf + (size_t)row_off * (size_t)num_features + fidx,
                X + (size_t)(group_start + row_off) * (size_t)num_features + fidx
            );
        }
        async_commit_group();
    #else
        for (int linear = tid; linear < total_elems; linear += THREADS) {
            const int row_off = linear / num_features;
            const int fidx    = linear % num_features;
            dst_buf[(size_t)row_off * (size_t)num_features + fidx] =
                X[(size_t)(group_start + row_off) * (size_t)num_features + fidx];
        }
    #endif

        // Zero-pad remaining rows up to K*tile so every subtile sees valid memory
        for (int row = rows + tid; row < K * tile; row += THREADS) {
            float* dst_row = dst_buf + (size_t)row * (size_t)num_features;
            for (int f = 0; f < num_features; ++f) dst_row[f] = 0.0f;
        }
    };

    // Preload first group
    int t = t0;
    int group_rows = min(rows_per_block, t_end - t);
    copy_group(t, t + group_rows, buf0);
#if __CUDA_ARCH__ >= 800
    async_wait_pending<0>();
#endif
    __syncthreads();

    float* cur = buf0;
    float* nxt = buf1;

    while (t < t_end) {
        group_rows = min(rows_per_block, t_end - t);

        const int t_next = t + rows_per_block;
        const bool has_next = (t_next < t_end);
        if (has_next) {
            const int next_rows = min(rows_per_block, t_end - t_next);
            copy_group(t_next, t_next + next_rows, nxt);
        }

        // Evaluate K subtiles of size `tile`
        for (int s = 0; s < K; ++s) {
            const int base_dp = t + s * tile;
            const int rows_this_sub = max(0, min(tile, t_end - base_dp));
            if (tid < rows_this_sub) {
                int sp = 0;
                float* feat_row = cur + ((size_t)s * (size_t)tile + (size_t)tid) * (size_t)num_features;

                // Same math as your working prefix batch kernel (values-only)
                for (int i = len - 1; i >= 0; --i) {
                    const int tcode = tokens[i];
                    if (tcode == TOK_CONST) {
                        s_val[sp++] = values[i];
                    } else if (tcode == TOK_VAR) {
                        int fidx = (int)values[i];
                        float v = (fidx >= 0 && fidx < num_features) ? feat_row[fidx] : 0.0f;
                        s_val[sp++] = v;
                    } else {
                        const int ar = op_arity(tcode);
                        if (ar == 1) {
                            float a = s_val[--sp];
                            s_val[sp++] = apply_op(tcode, a, 0.0f);
                        } else {
                            float a_v = s_val[--sp];
                            float b_v = s_val[--sp];
                            float r;
                            if      (tcode == OP_ADD) r = a_v + b_v;
                            else if (tcode == OP_SUB) r = a_v - b_v;
                            else if (tcode == OP_MUL) r = a_v * b_v;
                            else if (tcode == OP_DIV) r = a_v / b_v;
                            else                      r = apply_op(tcode, a_v, b_v);
                            s_val[sp++] = r;
                        }
                    }
                }
                out[base_dp + tid] = (sp > 0) ? s_val[sp - 1] : 0.0f;
            }
            __syncthreads(); // don’t let subtiles overlap on the same SMEM
        }

        if (!has_next) break;
    #if __CUDA_ARCH__ >= 800
        async_wait_pending<0>();
    #endif
        __syncthreads();
        float* tmp = cur; cur = nxt; nxt = tmp;
        t += rows_per_block;
    }
}

// ---- Host launcher (keeps signature and leaves everything else untouched) ----
extern "C" void eval_tree_gpu_async(
    const int*   tokens,
    const float* values,
    const float* X,          // [dataPoints, num_features]
    int len,
    int num_features,
    int dataPoints,
    float* out_dev,
    int blocks_hint,   // optional override; <=0 => auto
    int /*threads_hint*/ )  // ignored; THREADS is compile-time
{
    if (len <= 0 || len > MAX_EVAL_STACK ||
        num_features <= 0 || num_features > MAX_NUM_FEATURES ||
        dataPoints <= 0) {
        return;
    }

    // Compile-time choices (don’t alter other functions or macros)
    constexpr int THREADS_CT = EVAL_MAX_THREADS_PER_BLOCK;  // e.g., 128
    constexpr int K = 4;  // grouping used in kernel template launch

    // Must match the kernel's grouping K to size SMEM correctly
    const int rows_per_block = THREADS_CT * K;

    // Dynamic SMEM bytes needed:
    //   2 * (K * THREADS * num_features) + (THREADS * MAX_EVAL_STACK)
    const size_t smem_floats =
        2ULL * (size_t)rows_per_block * (size_t)num_features +
        1ULL * (size_t)THREADS_CT     * (size_t)MAX_EVAL_STACK;
    const int shmem_bytes = static_cast<int>(smem_floats * sizeof(float));

    // Grid size: each block covers rows_per_block datapoints
    int blocks = (blocks_hint > 0) ? blocks_hint
                                   : (dataPoints + rows_per_block - 1) / rows_per_block;

    dim3 grid(blocks, 1, 1);
    dim3 block(THREADS_CT, 1, 1);

    // Opt-in to large dynamic SMEM when needed (>48 KB)
    cudaError_t attr_err =
        cudaFuncSetAttribute(
            eval_async_kernel<THREADS_CT, K>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_bytes);
    if (attr_err != cudaSuccess) {
        // Not fatal if shmem_bytes <= default; just warn.
        fprintf(stderr, "cudaFuncSetAttribute failed: %s\n", cudaGetErrorString(attr_err));
    }

    eval_async_kernel<THREADS_CT, K>
        <<<grid, block, shmem_bytes>>>(tokens, values, X, len, num_features, dataPoints, out_dev);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr,
                "eval_tree_gpu_async launch error: %s (blocks=%d, threads=%d, shmem=%d)\n",
                cudaGetErrorString(err), blocks, THREADS_CT, shmem_bytes);
    }
}

// ---- Multi-expression batch evaluator (GPU) ----

// __device__ inline float eval_tree_device(const int* tokens,
//                                          const float* values,
//                                          const float* x,
//                                          int len,
//                                          int num_features)
// {
//     float stk[MAX_EVAL_STACK];
//     int   sp = 0;

//     for (int i = len - 1; i >= 0; --i) {
//         int tok = tokens[i];
//         if (tok > 0) { // operator
//             int arity = op_arity(tok);
//             if (arity == 1) {
//                 float a = static_stack_pop(stk, sp);
//                 float r = apply_op(tok, a, 0.0f);
//                 static_stack_push(stk, r, sp);
//             } else if (arity == 2) {
//                 float a = static_stack_pop(stk, sp);
//                 float b = static_stack_pop(stk, sp);
//                 float r = apply_op(tok, a, b);
//                 static_stack_push(stk, r, sp);
//             } else {
//                 // Ternary IF (kept for completeness if re-enabled)
//                 float c = static_stack_pop(stk, sp);
//                 float b = static_stack_pop(stk, sp);
//                 float a = static_stack_pop(stk, sp);
//                 float r = (a > 0.0f) ? b : c;
//                 static_stack_push(stk, r, sp);
//             }
//         } else if (tok == 0) { // constant
//             static_stack_push(stk, values[i], sp);
//         } else if (tok == -1) { // variable
//             int idx = (int)values[i];
//             float v = (idx >= 0 && idx < num_features) ? x[idx] : 0.0f;
//             static_stack_push(stk, v, sp);
//         }
//     }

//     return (sp > 0) ? static_stack_pop(stk, sp) : 0.0f;
// }

__device__ inline float eval_tree_device(const int* tokens,
                                         const float* values,
                                         const float* d_vars_flat,
                                         int len,
                                         int num_features,
                                         int num_dps,
                                         int dp_idx) {
    float stk[MAX_EVAL_STACK];
    int sp = 0;
    int output = 0; 
    int tok; 

    
    for (int i = len - 1; i >= 0; i--) {
        tok = tokens[i];
        if (tok > 0) { // operator
            int arity = op_arity(tok);
            if (arity == 2) {
                float a = stk[--sp];
                // --sp; 
                // float a = 0.0; 
                float b = stk[--sp];
                // --sp; 
                // float b = 0.0; 
                stk[sp++] = apply_op(tok, a, b);
                // stk[sp++] = (a, b);
                // sp++; 
                // stk[sp++] = 0; 
            } else { // unary
                float a = stk[--sp];
                // --sp; 
                // float a = 0.0; 
                stk[sp++] = apply_op(tok, a, 0.0f);
                // sp++; 
                // stk[sp++] = 0;
                // sp++; 
            }
        } else if (tok == 0) { // constant
            stk[sp++] = values[i];
            // tok = values[i];
            // sp++;
        } else if (tok == -1) { // variable
            int var_idx = (int)values[i];
            if (var_idx >= 0 && var_idx <= num_features) {
                stk[sp++] = d_vars_flat[var_idx * num_dps + dp_idx];
            } else {
                stk[sp++] = 0.0f;
            }
        }
    }
    return stk[--sp];
    // return 0.0f; 
    // return sp + tok; 
}

// __device__ inline float eval_tree_device(const int* tokens, const float* values, const float* x, int len, int num_features) {
//     float stk[MAX_EVAL_STACK];
//     int sp = 0;
//     int output = 0; 
//     float tmp = 1000; 

    
//     for (int i = len - 1; i >= 0; i--) {
//         int tok = tokens[i];
//         if (tok > 0) { // operator

//             int arity = op_arity(tok);
//             if (arity == 2) {
//                 float a = stk[--sp];
//                 // sp--;
//                 // float a = 0.0; 
//                 float b = stk[--sp]; 
//                 // sp--; 
//                 // if (a > b) { 
//                 //     stk[sp++] = 1.0f;
//                 // } else { 
//                 //     stk[sp++] = 0.0f;
//                 // }
//                 // float b = 0.0; 
//                 // tmp = apply_op(tok, tmp, tmp);
//                 tmp = a + b;
//                 stk[sp++] = tmp; 
//                 // sp++; 
//             } else { // unary
//                 float a = stk[--sp]; 
//                 // sp--;
//                 // if (a > 3) { 
//                 //     stk[sp++] = 1.0f;
//                 // } else {
//                 //     stk[sp++] = 0.0f;
//                 // }
//                 // tmp = apply_op(tok, a, a);
//                 tmp = tmp + a;
//                 stk[sp++] = tmp;
//                 // sp++; 
//             }
//         } else if (tok == 0) { // constant
//             // stk[sp++] = values[i];
//             tmp = values[i]; 
//             stk[sp++] = tmp;
//             // sp++;
//         } else if (tok == -1) { // variable
//             // stk[sp++] = x[(int)values[i]];
//             tmp = x[(int)values[i]];
//             stk[sp++] = tmp;
//             // sp++;
//         }
//     }
//     // return stk[--sp];
//     return  sp + tmp + stk[--sp]; 
// }

// GPU kernel: Each threadblock handles multiple expressions and subset of datapoints
__global__ void eval_prefix_kernel_multi_expression_batch(
    int *d_tokens_batch, float *d_values_batch, int *d_token_offsets, int *d_num_tokens,
    float *d_vars_flat, float *d_pred_batch, int num_vars, int num_dps, int num_exprs, int exprs_per_block)
{
    int block_expr_start = blockIdx.x * exprs_per_block;
    int block_expr_end = min(block_expr_start + exprs_per_block, num_exprs);
    int dp_start = blockIdx.y * blockDim.x;
    int dp_idx = dp_start + threadIdx.x; // which datapoint within this threadblock
    
    if (dp_idx >= num_dps) return;
    
    // Prepare input variables for this datapoint (shared by all expressions in this block)
    // float x[MAX_NUM_FEATURES];
    // #pragma unroll
    // for (int i = 0; i <= num_vars; i++) {
    //     // vars_flat is laid out as [var0_dp0, var0_dp1, ..., var1_dp0, var1_dp1, ...]
    //     x[i] = d_vars_flat[i * num_dps + dp_idx];
    // }
    
    // Process all expressions assigned to this threadblock
    for (int expr_idx = block_expr_start; expr_idx < block_expr_end; expr_idx++) {
        // Get tokens and values for this expression
        int token_offset = d_token_offsets[expr_idx];
        int *d_tokens = d_tokens_batch + token_offset;
        float *d_values = d_values_batch + token_offset;
        int num_tokens = d_num_tokens[expr_idx];
        
        // Get prediction pointer for this expression
        float *d_pred = d_pred_batch + (expr_idx * num_dps);
        
        // Evaluate expression for this datapoint
        d_pred[dp_idx] = eval_tree_device(d_tokens,
                                          d_values,
                                          d_vars_flat,
                                          num_tokens,
                                          num_vars,
                                          num_dps,
                                          dp_idx);
    }
}

// Host wrapper function to launch the kernel
extern "C" void launch_eval_prefix_kernel_multi_expression_batch(
    int *d_tokens_batch, float *d_values_batch, int *d_token_offsets, int *d_num_tokens,
    float *d_vars_flat, float *d_pred_batch, int num_vars, int num_dps, int num_exprs, int exprs_per_block,
    int blocks_x, int blocks_y, int threads_per_block)
{
    dim3 grid(blocks_x, blocks_y);
    eval_prefix_kernel_multi_expression_batch<<<grid, threads_per_block>>>(
        d_tokens_batch, d_values_batch, d_token_offsets, d_num_tokens,
        d_vars_flat, d_pred_batch, num_vars, num_dps, num_exprs, exprs_per_block);
}

__global__ void regression_mse_kernel(
    const float *preds,
    const double *labels,
    double *mse,
    unsigned int num_exprs,
    unsigned int total_dps)
{
    unsigned int expr = blockIdx.x;
    unsigned int block_dp = blockIdx.y;
    unsigned int tid = threadIdx.x;
    unsigned int dp = block_dp * blockDim.x + tid;

    __shared__ double shared_err[EVAL_MAX_THREADS_PER_BLOCK];

    double err = 0.0;
    if (expr < num_exprs && dp < total_dps) {
        size_t idx = (size_t)expr * (size_t)total_dps + (size_t)dp;
        double diff = (double)preds[idx] - labels[idx];
        err = (diff * diff) / (double)total_dps;
    }

    if (tid < blockDim.x) {
        shared_err[tid] = err;
    }

    __syncthreads();

    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride && (tid + stride) < blockDim.x) {
            shared_err[tid] += shared_err[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0 && expr < num_exprs) {
        atomicAdd(&mse[expr], shared_err[0]);
    }
}