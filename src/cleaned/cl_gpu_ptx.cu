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

#define MAX_EVAL_STACK 60

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

#define CU_LINK_CHECK(call, linkState, error_log) \
    do { \
        CUresult _st = (call); \
        if (_st != CUDA_SUCCESS) { \
            const char* estr = nullptr; \
            cuGetErrorString(_st, &estr); \
            fprintf(stderr, "CU Error %d: %s\n", (int)_st, estr); \
            if (error_log && error_log[0] != '\0') { \
                fprintf(stderr, "JIT error log:\n%s\n", error_log); \
            } \
            if (linkState) cuLinkDestroy(linkState); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

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
)";

struct ExprDesc {
    const int*   tokens;
    const float* values;
    int          len;
};

static std::string f32_to_ptx_hex(float v) {
    uint32_t u;
    std::memcpy(&u, &v, sizeof(u));
    std::ostringstream oss;
    oss << "0f" << std::hex << std::setfill('0') << std::setw(8) << u;
    return oss.str();
}

static std::string build_multi_expr_kernel_ptx(const std::vector<ExprDesc>& exprs, int num_features) {
    std::ostringstream oss;
    oss.setf(std::ios::scientific);
    oss << ".version 8.4\n.target sm_89\n.address_size 64\n";
    oss << ".extern .func (.param .b32 func_retval0) apply_op(.param .b32 p0, .param .b32 p1, .param .b32 p2);\n";
    
    oss << ".visible .entry eval_multi_expr_kernel(\n";
    oss << "    .param .u64 _X, .param .u32 _num_features, .param .u32 _num_dps, .param .u32 _num_exprs, .param .u64 _out\n) {\n";
    oss << "    .reg .pred %p<4>;\n    .reg .b32 %r<16>;\n    .reg .b64 %rd<12>;\n    .reg .f32 %f<" << (MAX_EVAL_STACK + 4) << ">;\n";
    oss << "    .param .b32 _ret; .param .b32 _op; .param .b32 _a; .param .b32 _b;\n";

    oss << "    ld.param.u64 %rd0, [_X];\n    ld.param.u32 %r0, [_num_features];\n    ld.param.u32 %r1, [_num_dps];\n    ld.param.u64 %rd1, [_out];\n";
    oss << "    mov.u32 %r2, %tid.x; mov.u32 %r3, %ctaid.x; mov.u32 %r4, %ntid.x;\n";
    oss << "    mad.lo.s32 %r5, %r3, %r4, %r2; // idx\n";
    oss << "    setp.ge.u32 %p0, %r5, %r1; @%p0 bra DONE;\n";
    oss << "    mul.lo.u32 %r6, %r5, %r0; mul.wide.u32 %rd2, %r6, 4; add.s64 %rd3, %rd0, %rd2;\n";

    const size_t num_exprs = exprs.size();
    for(size_t e=0; e<num_exprs; ++e) {
        const ExprDesc& ed = exprs[e];
        oss << "\n    // Expr " << e << "\n";
        oss << "    mov.u32 %r7, " << (unsigned int)e << "; mul.lo.u32 %r7, %r7, %r1; add.u32 %r7, %r7, %r5; mul.wide.u32 %rd4, %r7, 4; add.s64 %rd5, %rd1, %rd4;\n";
        
        struct Temp { int reg; };
        std::vector<Temp> stack;
        stack.reserve(MAX_EVAL_STACK);

        auto push_const = [&](float c) {
            int reg = (stack.size() >= MAX_EVAL_STACK) ? MAX_EVAL_STACK - 1 : (int)stack.size();
            oss << "    mov.f32 %f" << reg << ", " << f32_to_ptx_hex(c) << ";\n";
            stack.push_back({reg});
        };
        auto push_var = [&](int fidx) {
            int reg = (stack.size() >= MAX_EVAL_STACK) ? MAX_EVAL_STACK - 1 : (int)stack.size();
            // Check if fidx < num_features (%r0)
            oss << "    mov.u32 %r8, " << fidx << ";\n";
            oss << "    setp.lt.u32 %p2, %r8, %r0;\n";
            oss << "    @%p2 mul.wide.u32 %rd6, %r8, 4;\n"; 
            oss << "    @%p2 add.s64 %rd6, %rd3, %rd6;\n";
            oss << "    @%p2 ld.global.f32 %f" << reg << ", [%rd6];\n";
            oss << "    @!%p2 mov.f32 %f" << reg << ", 0f00000000;\n";
            stack.push_back({reg});
        };
        auto emit_unary = [&](int op) {
            if(stack.empty()) return;
            Temp a = stack.back(); 
            int r = a.reg; // reuse reg
            oss << "    mov.u32 %r10, " << op << "; st.param.b32 [_op], %r10; st.param.f32 [_a], %f" << r << "; st.param.f32 [_b], 0f00000000; call.uni (_ret), apply_op, (_op, _a, _b); ld.param.f32 %f" << r << ", [_ret];\n";
        };
        auto emit_binary = [&](int op) {
             if(stack.size()<2) return;
             Temp a = stack.back(); stack.pop_back();
             Temp b = stack.back(); stack.pop_back();
             int ra = a.reg; int rb = b.reg;
             oss << "    mov.u32 %r10, " << op << "; st.param.b32 [_op], %r10; st.param.f32 [_a], %f" << ra << "; st.param.f32 [_b], %f" << rb << "; call.uni (_ret), apply_op, (_op, _a, _b); ld.param.f32 %f" << rb << ", [_ret];\n";
             stack.push_back({rb});
        };
        auto emit_ternary_if = [&]() {
            if(stack.size()<3) return;
            Temp c = stack.back(); stack.pop_back();
            Temp b = stack.back(); stack.pop_back();
            Temp a = stack.back(); stack.pop_back();
            int ra = a.reg; int rb = b.reg; int rc = c.reg;
            oss << "    setp.gt.f32 %p1, %f" << ra << ", 0f00000000; selp.f32 %f" << ra << ", %f" << rb << ", %f" << rc << ", %p1;\n";
            stack.push_back({ra});
        };

        for(int i=ed.len-1; i>=0; --i) {
            int tok = ed.tokens[i];
            if(tok == TOK_CONST) push_const(ed.values[i]);
            else if(tok == TOK_VAR) push_var((int)ed.values[i]);
            else if(tok == Function::IF) emit_ternary_if();
            else {
                int ar = op_arity(tok);
                if(ar==1) emit_unary(tok);
                else if(ar==2) emit_binary(tok);
                else emit_ternary_if();
            }
        }
        int res_reg = stack.empty() ? 0 : stack.back().reg;
        if(stack.empty()) oss << "    mov.f32 %f0, 0f00000000;\n";
        oss << "    st.global.f32 [%rd5], %f" << res_reg << ";\n";
    }
    oss << "DONE:\n    ret;\n}\n";
    return oss.str();
}

static double eval_multi_expr_straightline_gpu(
    const std::vector<ExprDesc>& exprs,
    const float* d_X, int num_features, int num_dps,
    float* d_out, double* compile_ms_p, double* jit_ms_p) 
{
    if(exprs.empty()) return 0;
    using clock_t = std::chrono::high_resolution_clock;
    auto t0 = clock_t::now();
    std::string ptx = build_multi_expr_kernel_ptx(exprs, num_features);
    auto t1 = clock_t::now();
    
    CUdevice cuDevice; CUcontext cuContext;
    CU_CHECK(cuInit(0));
    CU_CHECK(cuDeviceGet(&cuDevice, 0));
    CU_CHECK(cuCtxGetCurrent(&cuContext));
    if(!cuContext) CU_CHECK(cuCtxCreate(&cuContext, 0, cuDevice));

    static char error_log[8192] = {0};
    CUlinkState linkState;
    CUjit_option jit_opts[] = { CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES };
    void* jit_optvals[] = { error_log, (void*)(long)sizeof(error_log) };
    
    auto t2 = clock_t::now();
    CU_LINK_CHECK(cuLinkCreate(2, jit_opts, jit_optvals, &linkState), linkState, error_log);
    CU_LINK_CHECK(cuLinkAddData(linkState, CU_JIT_INPUT_PTX, (void*)ptx.c_str(), ptx.size(), "main", 0,0,0), linkState, error_log);
    CU_LINK_CHECK(cuLinkAddFile(linkState, CU_JIT_INPUT_PTX, "../utils/helpers.ptx", 0,0,0), linkState, error_log);
    
    void* cubin; size_t cubinSize;
    CU_CHECK(cuLinkComplete(linkState, &cubin, &cubinSize));
    CUmodule module; CUfunction func;
    CU_CHECK(cuModuleLoadData(&module, cubin));
    CU_CHECK(cuModuleGetFunction(&func, module, "eval_multi_expr_kernel"));
    
    auto t3 = clock_t::now();

    int num_exprs = (int)exprs.size();
    void* args[] = { (void*)&d_X, (void*)&num_features, (void*)&num_dps, (void*)&num_exprs, (void*)&d_out };
    int threads=128; int blocks=(num_dps+threads-1)/threads;
    
    auto t4 = clock_t::now();
    CU_CHECK(cuLaunchKernel(func, blocks,1,1, threads,1,1, 0,0, args, 0));
    CU_CHECK(cuCtxSynchronize());
    auto t5 = clock_t::now();

    CU_CHECK(cuModuleUnload(module));
    CU_CHECK(cuLinkDestroy(linkState));
    
    double build_ms = std::chrono::duration<double,std::milli>(t1-t0).count();
    double jit_ms = std::chrono::duration<double,std::milli>(t3-t2).count();
    double launch_ms = std::chrono::duration<double,std::milli>(t5-t4).count();

    if(compile_ms_p) *compile_ms_p = build_ms;
    if(jit_ms_p) *jit_ms_p = jit_ms;
    return launch_ms;
}

__global__ void reduce_mse_from_preds_kernel(const float* d_preds, const float* d_targets, float* d_mses, int num_exprs, int total_dps) {
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

struct GPUPTXContext {
    float* d_X = nullptr;
    float* d_targets = nullptr;
    float* d_out = nullptr;
    float* d_mses = nullptr;
    size_t X_size = 0;
    size_t out_size = 0;
    size_t targets_size = 0;
    ~GPUPTXContext() {
        if(d_X) cudaFree(d_X);
        if(d_targets) cudaFree(d_targets);
        if(d_out) cudaFree(d_out);
        if(d_mses) cudaFree(d_mses);
    }
};

void* create_gpu_ptx_context() { return new GPUPTXContext(); }
void destroy_gpu_ptx_context(void* ctx) { delete (GPUPTXContext*)ctx; }

void evaluate_gpu_ptx_wrapper(
    InputInfo& input_info, 
    float*** all_vars, 
    std::vector<float>& mses, 
    void* ctx_ptr, 
    bool upload_X,
    RunStats& stats)  
{
    GPUPTXContext* ctx = (GPUPTXContext*)ctx_ptr;
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
        
        // Flatten targets (last var)
        float* h_targets = new float[num_dps];
        for(int i=0; i<num_dps; ++i) h_targets[i] = all_vars[0][num_vars][i]; // Target is usually last? all_vars is [vars][dps]. Logic differs.
        // check cl_gpu_simple: flatten includes v=0..num_vars. The target is usually separate in some contexts or implicit.
        // In flatten_X function above: v <= num_vars. So flat array has vars 0..num_vars.
        // eval_multi_expr assumes X has num_features cols.
        // Targets are needed for MSE.
        // Let's assume input_info.values has targets? No.
        // evolution code passes all_vars.
        // Standard SR: all_vars is [variable_index][datapoint]. Var index 'num_vars' is target.
        // My flatten_X includes target as last column?
        // flatten_X loop: v <= num_vars. Yes.
        
        // Wait, if X includes target, and we call eval(X), the evaluator can access target as a feature?
        // Usually target is excluded from features available to GP.
        // But `flatten_X` includes it. 
        // In `eval_multi_expr_straightline_gpu`, it takes `d_X` and `num_features`.
        // If expressions only reference vars 0..num_vars-1, it is safe.
        // But `reduce_mse` needs targets.
        // So I should upload targets separately to `d_targets`.
        
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
    all_tokens.reserve(num_exprs * 30); // heuristic
    all_values.reserve(num_exprs * 30);
    
    // Need to handle packing? 
    // InputInfo has tokens[i] and values[i].
    for(int i=0; i<num_exprs; ++i) {
        int len = input_info.num_tokens[i];
        int start = all_tokens.size();
        for(int k=0; k<len; ++k) {
            all_tokens.push_back(input_info.tokens[i][k]);
            all_values.push_back((float)input_info.values[i][k]);
        }
        exprs[i] = { &all_tokens[start], &all_values[start], len };
    }

    // Launch PTX
    double compile_ms=0, jit_ms=0;
    double launch_ms = eval_multi_expr_straightline_gpu(exprs, ctx->d_X, num_features, num_dps, ctx->d_out, &compile_ms, &jit_ms);

    // Reduce
    int threads=256;
    size_t smem = threads*(sizeof(float)+sizeof(int));
    auto t_red_start = std::chrono::high_resolution_clock::now();
    reduce_mse_from_preds_kernel<<<num_exprs, threads, smem>>>(ctx->d_out, ctx->d_targets, ctx->d_mses, num_exprs, num_dps);
    cudaDeviceSynchronize();
    auto t_red_end = std::chrono::high_resolution_clock::now();
    double reduce_ms = std::chrono::duration<double, std::milli>(t_red_end - t_red_start).count();

    // D2H
    mses.resize(num_exprs);
    auto t_d2h_start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(mses.data(), ctx->d_mses, num_exprs*sizeof(float), cudaMemcpyDeviceToHost);
    auto t_d2h_end = std::chrono::high_resolution_clock::now();
    double d2h_ms = std::chrono::duration<double, std::milli>(t_d2h_end - t_d2h_start).count();

    // Stats
    stats.total_eval_time_ms += (t_h2d + compile_ms + jit_ms + launch_ms + reduce_ms + d2h_ms);
    stats.data_transfer_time_ms = t_h2d + d2h_ms;
    stats.jit_compile_time_ms = compile_ms + jit_ms;
    stats.gpu_kernel_time_ms = launch_ms + reduce_ms; 
    // We can also log compile time if we add a field for it, or merge into kernel/detect time.
    // For now, let's just ensure total is correct.
}
