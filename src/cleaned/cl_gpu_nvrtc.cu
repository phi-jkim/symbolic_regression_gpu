#include <cuda_runtime.h>
#include <nvrtc.h>
#include <cuda.h>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <algorithm>
#include <thread>
#include <cmath>
#include <chrono>

#include "../utils/utils.h"
#include "../utils/defs.h"

#ifndef MAX_EVAL_STACK
#define MAX_EVAL_STACK 60
#endif

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define NVRTC_CHECK(call) \
    do { \
        nvrtcResult _status = call; \
        if (_status != NVRTC_SUCCESS) { \
            fprintf(stderr, "NVRTC Error at %s:%d - %s\n", __FILE__, __LINE__, nvrtcGetErrorString(_status)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CU_CHECK(call) \
    do { \
        CUresult _status = call; \
        if (_status != CUDA_SUCCESS) { \
            const char* errStr = nullptr; \
            cuGetErrorString(_status, &errStr); \
            fprintf(stderr, "CU Error at %s:%d - %s\n", __FILE__, __LINE__, errStr ? errStr : "unknown"); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Helper C++ code for NVRTC
static const char* DEVICE_HELPERS_SRC = R"(
static const int OP_ADD        = 1;
static const int OP_SUB        = 2;
static const int OP_MUL        = 3;
static const int OP_DIV        = 4;
static const int OP_POW        = 5;
static const int OP_MIN        = 6;
static const int OP_MAX        = 7;
static const int OP_LOOSE_DIV  = 8;
static const int OP_LOOSE_POW  = 9;
static const int OP_SIN        = 10;
static const int OP_COS        = 11;
static const int OP_TAN        = 12;
static const int OP_SINH       = 13;
static const int OP_COSH       = 14;
static const int OP_TANH       = 15;
static const int OP_EXP        = 16;
static const int OP_LOG        = 17;
static const int OP_INV        = 18;
static const int OP_ASIN       = 19;
static const int OP_ACOS       = 20;
static const int OP_ATAN       = 21;
static const int OP_LOOSE_LOG  = 22;
static const int OP_LOOSE_INV  = 23;
static const int OP_ABS        = 24;
static const int OP_NEG        = 25;
static const int OP_SQRT       = 26;
static const int OP_LOOSE_SQRT = 27;

static const float DELTA   = 1e-9f;
static const float MAX_VAL = 1e9f;

__device__ inline float my_nan() { return __int_as_float(0x7fffffff); }
__device__ inline float clamp_val(float r) { return r; }
__device__ inline float safe_div(float a, float b) {
    const float eps = 1e-12f;
    float denom = fabsf(b) < eps ? (b < 0 ? -eps : eps) : b;
    return a / denom;
}
__device__ inline float loose_div(float a, float b) {
    float denom = fabsf(b) <= DELTA ? (b < 0 ? -DELTA : DELTA) : b;
    return a / denom;
}
__device__ inline float safe_log(float a) {
    const float eps = 1e-12f;
    return logf(fabsf(a) + eps);
}
__device__ inline float loose_log(float a) {
    if (a == 0.0f) return -MAX_VAL;
    return logf(fabsf(a));
}
__device__ inline float safe_inv(float a) {
    if (a == 0.0f) return my_nan();
    return 1.0f / a;
}
__device__ inline float loose_inv(float a) {
    float denom = fabsf(a) <= DELTA ? (a < 0 ? -DELTA : DELTA) : a;
    return 1.0f / denom;
}

__device__ inline float apply_op(int op, float a, float b) {
    switch (op) {
        case OP_ADD:       return clamp_val(a + b);
        case OP_SUB:       return clamp_val(a - b);
        case OP_MUL:       return clamp_val(a * b);
        case OP_DIV:       return (b == 0.0f) ? my_nan() : clamp_val(a / b);
        case OP_POW:       return clamp_val(powf(a, b));
        case OP_MIN:       return clamp_val(a <= b ? a : b);
        case OP_MAX:       return clamp_val(a >= b ? a : b);
        case OP_LOOSE_DIV: return clamp_val(loose_div(a, b));
        case OP_LOOSE_POW: return (a == 0.0f && b == 0.0f) ? 0.0f : clamp_val(powf(fabsf(a), b));
        case OP_SIN:        return clamp_val(sinf(a));
        case OP_COS:        return clamp_val(cosf(a));
        case OP_TAN:        return clamp_val(tanf(a));
        case OP_SINH:       return clamp_val(sinhf(a));
        case OP_COSH:       return clamp_val(coshf(a));
        case OP_TANH:       return clamp_val(tanhf(a));
        case OP_EXP:        return clamp_val(expf(a));
        case OP_LOG:        return clamp_val(logf(a));
        case OP_INV:        return clamp_val(safe_inv(a));
        case OP_ASIN:       return clamp_val(asinf(a));
        case OP_ACOS:       return clamp_val(acosf(a));
        case OP_ATAN:       return clamp_val(atanf(a));
        case OP_LOOSE_LOG:  return clamp_val(loose_log(a));
        case OP_LOOSE_INV:  return clamp_val(loose_inv(a));
        case OP_ABS:        return clamp_val(fabsf(a));
        case OP_NEG:        return clamp_val(-a);
        case OP_SQRT:       return clamp_val(sqrtf(a));
        case OP_LOOSE_SQRT: return clamp_val(sqrtf(fabsf(a)));
        default:            return 0.0f;
    }
}
)";

struct ExprDesc {
    const int*   tokens;
    const float* values;
    int          len;
};

static std::string build_multi_expr_kernel_src(
    const std::vector<ExprDesc>& exprs,
    int                          num_features)
{
    std::ostringstream oss;
    oss.setf(std::ios::scientific);

    // Device helpers
    oss << DEVICE_HELPERS_SRC << "\n";

    // Kernel header
    oss <<
R"(extern "C" __global__
void eval_multi_expr_kernel(const float* __restrict__ X,
                            int   num_features,
                            int   num_dps,
                            int   num_exprs,
                            float* __restrict__ out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_dps) return;
    const float* x = X + idx * num_features;
)";

    const size_t num_exprs = exprs.size();
    std::vector<std::string> expr_blocks(num_exprs);

    // Lambda to process chunk of expressions
    auto worker_fn = [&](size_t start_e, size_t end_e) {
        for (size_t e = start_e; e < end_e; ++e) {
            const ExprDesc& ed = exprs[e];

            std::ostringstream expr_oss;
            expr_oss.setf(std::ios::scientific);

            expr_oss << "    // ----- Expression " << e << " -----\n";
            expr_oss << "    {\n";
            expr_oss << "        float t[" << MAX_EVAL_STACK << "];\n";

            struct Temp { int id; };
            std::vector<Temp> stack;
            int nextTemp = 0;

            auto alloc_temp = [&]() -> int {
                if (nextTemp >= MAX_EVAL_STACK) {
                   return MAX_EVAL_STACK - 1; 
                }
                return nextTemp++;
            };

            auto emit_const = [&](float c) {
                int id = alloc_temp();
                expr_oss << "        t[" << id << "] = " << c << "f;\n";
                stack.push_back({id});
            };

            auto emit_var = [&](int fidx) {
                int id = alloc_temp();
                expr_oss << "        t[" << id << "]" 
                         << " = ((" << fidx << " >= 0 && " << fidx
                         << " < num_features) ? x[" << fidx << "] : 0.0f);\n";
                stack.push_back({id});
            };

            auto emit_unary = [&](int op) {
                if (stack.empty()) return;
                Temp a = stack.back(); stack.pop_back();
                int id = alloc_temp();
                expr_oss << "        t[" << id << "]" 
                         << " = apply_op(" << op << ", t[" << a.id << "], 0.0f);\n";
                stack.push_back({id});
            };

            auto emit_binary = [&](int op) {
                if (stack.size() < 2) return;
                Temp a = stack.back(); stack.pop_back();
                Temp b = stack.back(); stack.pop_back();
                int id = alloc_temp();
                expr_oss << "        t[" << id << "]" 
                         << " = apply_op(" << op << ", t[" << a.id << "], t[" << b.id << "]);\n";
                stack.push_back({id});
            };

            auto emit_ternary_if = [&]() {
                if (stack.size() < 3) return;
                Temp c = stack.back(); stack.pop_back();
                Temp b = stack.back(); stack.pop_back();
                Temp a = stack.back(); stack.pop_back();
                int id = alloc_temp();
                expr_oss << "        t[" << id << "]" 
                         << " = (t[" << a.id << "] > 0.0f) ? t[" << b.id << "]" 
                         << " : t[" << c.id << "];\n";
                stack.push_back({id});
            };

            for (int i = ed.len - 1; i >= 0; --i) {
                int t = ed.tokens[i];
                if (t == TOK_CONST) {
                    emit_const(ed.values[i]);
                } else if (t == TOK_VAR) {
                    int fidx = static_cast<int>(ed.values[i]);
                    emit_var(fidx);
                } else if (t == Function::IF) {
                    emit_ternary_if();
                } else {
                    int ar = op_arity(t);
                    if (ar == 1) {
                        emit_unary(t);
                    } else if (ar == 2) {
                        emit_binary(t);
                    } else {
                        emit_ternary_if();
                    }
                }
            }

            int result_id = 0;
            if (!stack.empty()) {
                result_id = stack.back().id;
            } else {
                expr_oss << "        t[0] = 0.0f;\n";
                result_id = 0;
            }

            expr_oss << "        out[" << e
                     << " * num_dps + idx] = t[" << result_id << "];\n";
            expr_oss << "    }\n\n";

            expr_blocks[e] = expr_oss.str();
        }
    };

    // Parallelize generation
    /*
    unsigned int hc = std::thread::hardware_concurrency();
    size_t max_threads = hc ? (size_t)hc : 1;
    if (max_threads > num_exprs) max_threads = num_exprs;
    std::vector<std::thread> workers;
    workers.reserve(max_threads);
    size_t chunk = (num_exprs + max_threads - 1) / max_threads;
    for (size_t t = 0; t < max_threads; ++t) {
        size_t start_e = t * chunk;
        if (start_e >= num_exprs) break;
        size_t end_e = start_e + chunk;
        if (end_e > num_exprs) end_e = num_exprs;
        workers.emplace_back(worker_fn, start_e, end_e);
    }
    for (auto &th : workers) th.join();
    */
    // Sequential generation for simplicity/safety first, can parallelize if needed
    worker_fn(0, num_exprs);

    for (size_t e = 0; e < num_exprs; ++e) {
        oss << expr_blocks[e];
    }

    oss << "}\n";
    return oss.str();
}

static double eval_multi_expr_straightline_gpu_nvrtc(
    const std::vector<ExprDesc>& exprs,
    const float*                 d_X,
    int                          num_features,
    int                          num_dps,
    float*                       d_out,
    double*                      compile_ms_p,
    double*                      jit_ms_p)
{
    if (exprs.empty()) return 0.0;
    using clock_t = std::chrono::high_resolution_clock;

    // 1. Build Source
    auto t1 = clock_t::now();
    std::string src = build_multi_expr_kernel_src(exprs, num_features);
    auto t2 = clock_t::now();
    double build_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    // 2. NVRTC Compile (C++ -> PTX)
    nvrtcProgram prog;
    NVRTC_CHECK(nvrtcCreateProgram(&prog, src.c_str(), "multi_expr.cu", 0, nullptr, nullptr));
    const char* opts[] = { "--std=c++17", "--gpu-architecture=compute_89", "-I../utils" }; // Assuming build logic
    nvrtcResult compile_res = nvrtcCompileProgram(prog, 3, opts);

    if (compile_res != NVRTC_SUCCESS) {
        size_t logSize;
        nvrtcGetProgramLogSize(prog, &logSize);
        std::string log(logSize, '\0');
        nvrtcGetProgramLog(prog, &log[0]);
        fprintf(stderr, "NVRTC Fail: %s\n", log.c_str());
        nvrtcDestroyProgram(&prog);
        exit(EXIT_FAILURE);
    }

    size_t ptxSize;
    NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptxSize));
    std::string ptx(ptxSize, '\0');
    NVRTC_CHECK(nvrtcGetPTX(prog, &ptx[0]));
    NVRTC_CHECK(nvrtcDestroyProgram(&prog));
    
    auto t3 = clock_t::now();
    double nvrtc_time_ms = std::chrono::duration<double, std::milli>(t3 - t2).count(); // Compile time

    // 3. JIT Load (PTX -> SASS)
    CUdevice cuDevice; CUcontext cuContext;
    CU_CHECK(cuInit(0));
    CU_CHECK(cuDeviceGet(&cuDevice, 0));
    CU_CHECK(cuCtxGetCurrent(&cuContext));
    if(!cuContext) CU_CHECK(cuCtxCreate(&cuContext, 0, cuDevice));

    CUmodule module; CUfunction func;
    CU_CHECK(cuModuleLoadData(&module, ptx.c_str()));
    CU_CHECK(cuModuleGetFunction(&func, module, "eval_multi_expr_kernel"));
    
    auto t4 = clock_t::now();
    double jit_ms = std::chrono::duration<double, std::milli>(t4 - t3).count();

    // 4. Launch
    int num_exprs_i = (int)exprs.size();
    void* args[] = { (void*)&d_X, (void*)&num_features, (void*)&num_dps, (void*)&num_exprs_i, (void*)&d_out };
    int threads = 128;
    int blocks = (num_dps + threads - 1) / threads;

    CU_CHECK(cuLaunchKernel(func, blocks, 1, 1, threads, 1, 1, 0, 0, args, nullptr));
    CU_CHECK(cuCtxSynchronize());
    CU_CHECK(cuModuleUnload(module));
    
    auto t5 = clock_t::now();
    double launch_ms = std::chrono::duration<double, std::milli>(t5 - t4).count();

    if(compile_ms_p) *compile_ms_p = build_ms + nvrtc_time_ms;
    if(jit_ms_p) *jit_ms_p = jit_ms;

    return launch_ms;
}

// Reduction kernel (Same as PTX)
__global__ void reduce_mse_from_preds_kernel_nvrtc(const float* d_preds, const float* d_targets, float* d_mses, int num_exprs, int total_dps) {
    int expr_idx = blockIdx.x;
    if (expr_idx >= num_exprs) return;

    float fsum = 0.0f;
    int valid_count = 0;

    for (int i = threadIdx.x; i < total_dps; i += blockDim.x) {
        float pred = d_preds[expr_idx * total_dps + i];
        float target = d_targets[i];
        if (!isnan(pred) && !isnan(target)) {
            float diff = pred - target;
            fsum += diff * diff;
            valid_count++;
        }
    }

    extern __shared__ float sdata_f[];
    int* scount = (int*)&sdata_f[blockDim.x];
    int tid = threadIdx.x;
    sdata_f[tid] = fsum;
    scount[tid] = valid_count;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata_f[tid] += sdata_f[tid + s];
            scount[tid] += scount[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_mses[expr_idx] = (scount[0] > 0) ? sdata_f[0] / scount[0] : NAN;
    }
}

static float* flatten_X(float** vars, int num_vars, int total_dps) {
    float* flat = new float[(num_vars+1)*total_dps];
    for(int v=0; v<=num_vars; ++v)
        for(int i=0; i<total_dps; ++i)
            flat[v*total_dps+i] = vars[v][i];
    return flat;
}

struct GPUNVRTCContext {
    float* d_X = nullptr;
    float* d_targets = nullptr;
    float* d_out = nullptr;
    float* d_mses = nullptr;
    size_t X_size = 0;
    size_t out_size = 0;
    size_t targets_size = 0;
    ~GPUNVRTCContext() {
        if(d_X) cudaFree(d_X);
        if(d_targets) cudaFree(d_targets);
        if(d_out) cudaFree(d_out);
        if(d_mses) cudaFree(d_mses);
    }
};

void* create_gpu_nvrtc_context() { return new GPUNVRTCContext(); }
void destroy_gpu_nvrtc_context(void* ctx) { delete (GPUNVRTCContext*)ctx; }

void evaluate_gpu_nvrtc_wrapper(
    InputInfo& input_info, 
    float*** all_vars, 
    std::vector<float>& mses, 
    void* ctx_ptr, 
    bool upload_X,
    RunStats& stats) 
{
    GPUNVRTCContext* ctx = (GPUNVRTCContext*)ctx_ptr;
    int num_exprs = input_info.num_exprs;
    if (num_exprs == 0) return;
    int num_vars = input_info.num_vars[0];
    int num_dps = input_info.num_dps[0];
    int num_features = num_vars + 1;

    // Buffer management
    size_t req_X_size = (size_t)num_features * num_dps * sizeof(float);
    if(ctx->X_size < req_X_size) {
        if(ctx->d_X) cudaFree(ctx->d_X);
        cudaMalloc(&ctx->d_X, req_X_size);
        ctx->X_size = req_X_size;
    }
    size_t req_out_size = (size_t)num_exprs * num_dps * sizeof(float);
    if(ctx->out_size < req_out_size) {
        if(ctx->d_out) cudaFree(ctx->d_out);
        cudaMalloc(&ctx->d_out, req_out_size);
        ctx->out_size = req_out_size;
    }
    size_t req_tgt_size = num_dps * sizeof(float);
    if(ctx->targets_size < req_tgt_size) {
        if(ctx->d_targets) cudaFree(ctx->d_targets);
        cudaMalloc(&ctx->d_targets, req_tgt_size);
        ctx->targets_size = req_tgt_size;
    }
    if(!ctx->d_mses) cudaMalloc(&ctx->d_mses, num_exprs * sizeof(float));

    // Upload X and Targets
    float t_h2d = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    if(upload_X) {
        float* h_X = flatten_X(all_vars[0], num_vars, num_dps);
        cudaMemcpy(ctx->d_X, h_X, req_X_size, cudaMemcpyHostToDevice);
        
        float* h_targets = new float[num_dps];
        for(int i=0; i<num_dps; ++i) h_targets[i] = all_vars[0][num_vars][i];
        cudaMemcpy(ctx->d_targets, h_targets, req_tgt_size, cudaMemcpyHostToDevice);

        delete[] h_X;
        delete[] h_targets;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    t_h2d = std::chrono::duration<double, std::milli>(t1-t0).count();

    // Prepare ExprDesc
    std::vector<ExprDesc> exprs(num_exprs);
    std::vector<int> all_tokens;
    std::vector<float> all_values;
    all_tokens.reserve(num_exprs * 30);
    all_values.reserve(num_exprs * 30);
    
    for(int i=0; i<num_exprs; ++i) {
        int len = input_info.num_tokens[i];
        int start = all_tokens.size();
        for(int k=0; k<len; ++k) {
            all_tokens.push_back(input_info.tokens[i][k]);
            all_values.push_back((float)input_info.values[i][k]);
        }
        exprs[i] = { &all_tokens[start], &all_values[start], len };
    }

    // Launch NVRTC
    double compile_ms=0, jit_ms=0;
    double launch_ms = eval_multi_expr_straightline_gpu_nvrtc(exprs, ctx->d_X, num_features, num_dps, ctx->d_out, &compile_ms, &jit_ms);

    // Reduce
    int threads=256;
    size_t smem = threads*(sizeof(float)+sizeof(int));
    reduce_mse_from_preds_kernel_nvrtc<<<num_exprs, threads, smem>>>(ctx->d_out, ctx->d_targets, ctx->d_mses, num_exprs, num_dps);
    cudaDeviceSynchronize();
    
    // D2H
    mses.resize(num_exprs);
    auto t_d2h_start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(mses.data(), ctx->d_mses, num_exprs*sizeof(float), cudaMemcpyDeviceToHost);
    auto t_d2h_end = std::chrono::high_resolution_clock::now();
    double d2h_ms = std::chrono::duration<double, std::milli>(t_d2h_end - t_d2h_start).count();

    // Stats
    // Compile Time maps to build_src + nvrtc
    // JIT Time maps to load_ptx (JIT)
    stats.total_eval_time_ms += (t_h2d + compile_ms + jit_ms + launch_ms + d2h_ms);
    stats.data_transfer_time_ms = t_h2d + d2h_ms;
    stats.jit_compile_time_ms = compile_ms + jit_ms; // Or distinguish? user just asked for "jit time". 
    // Usually JIT includes NVRTC. But here we have two phases.
    // Let's sum them up for "Compile/JIT" overhead.
    stats.gpu_kernel_time_ms = launch_ms; 
}
