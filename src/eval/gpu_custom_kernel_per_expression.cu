// gpu_custom_kernel.cu
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <limits>
#include <thread>
#include <chrono>

#include <nvrtc.h>
#include <cuda.h>
#include <cstdint>
#include <iomanip>
#include <cstring>

#include "../utils/utils.h"
#include "../utils/gpu_kernel.h"

#ifndef MAX_EVAL_STACK
#define MAX_EVAL_STACK 60
#endif

#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

#define CU_LINK_CHECK(call, linkState, error_log)                      \
    do {                                                               \
        CUresult _st = (call);                                         \
        if (_st != CUDA_SUCCESS) {                                     \
            const char* estr = nullptr;                                \
            cuGetErrorString(_st, &estr);                              \
            fprintf(stderr, "CU Error %d: %s\n", (int)_st, estr);      \
            if (error_log && error_log[0] != '\0') {                   \
                fprintf(stderr, "JIT error log:\n%s\n", error_log);    \
            }                                                          \
            if (linkState) cuLinkDestroy(linkState);                   \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

#define NVRTC_CHECK(call)                                           \
    do {                                                            \
        nvrtcResult _status = call;                                 \
        if (_status != NVRTC_SUCCESS) {                             \
            fprintf(stderr, "NVRTC Error at %s:%d - %s\n",          \
                    __FILE__, __LINE__, nvrtcGetErrorString(_status)); \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

#define CU_CHECK(call)                                              \
    do {                                                            \
        CUresult _status = call;                                    \
        if (_status != CUDA_SUCCESS) {                              \
            const char* errStr = nullptr;                           \
            cuGetErrorString(_status, &errStr);                     \
            fprintf(stderr, "CU Error at %s:%d - %s\n",             \
                    __FILE__, __LINE__, errStr ? errStr : "unknown"); \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

// ============================================================================
//  Device helpers & opcode mapping used by NVRTC-generated kernels
// ============================================================================


static const char* DEVICE_HELPERS_SRC = R"(
// Minimal opcode definitions (copy values from defs.h/opcodes.h)
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

// Simple device-side NaN helper to avoid relying on NAN from system headers.
__device__ inline float my_nan() {
    return __int_as_float(0x7fffffff);
}

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
        // Binary operators (1-9)
        case OP_ADD:       return clamp_val(a + b);
        case OP_SUB:       return clamp_val(a - b);
        case OP_MUL:       return clamp_val(a * b);
        case OP_DIV:       return (b == 0.0f) ? my_nan() : clamp_val(a / b);
        case OP_POW:       return clamp_val(powf(a, b));
        case OP_MIN:       return clamp_val(a <= b ? a : b);
        case OP_MAX:       return clamp_val(a >= b ? a : b);
        case OP_LOOSE_DIV: return clamp_val(loose_div(a, b));
        case OP_LOOSE_POW: return (a == 0.0f && b == 0.0f) ? 0.0f : clamp_val(powf(fabsf(a), b));
        // Unary operators (10-27)
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

// ============================================================================
//  Single-expression straight-line kernel (per expression)
// ============================================================================

// Build a CUDA kernel source that evaluates a single prefix expression.
// Signature:
//
// extern "C" __global__
// void eval_expr_kernel(const float* __restrict__ X,
//                       int num_features,
//                       int num_dps,
//                       float* __restrict__ out);
//
// X is [num_dps, num_features] row-major.
static std::string build_straightline_kernel_src(
    const int*   tokens,
    const float* values,
    int          len,
    int          num_features)
{
    std::ostringstream oss;
    oss.setf(std::ios::scientific);

    // 1. Device helpers
    oss << DEVICE_HELPERS_SRC << "\n";

    // 2. Kernel header
    oss <<
R"(extern "C" __global__
void eval_expr_kernel(const float* __restrict__ X,
                      int num_features,
                      int num_dps,
                      float* __restrict__ out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_dps) return;
    const float* x = X + idx * num_features;
)";

    struct Temp { int id; };
    std::vector<Temp> stack;
    int nextTemp = 0;

    auto alloc_temp = [&]() -> int {
        if (nextTemp >= MAX_EVAL_STACK) {
            throw std::runtime_error("MAX_EVAL_STACK exceeded in build_straightline_kernel_src");
        }
        return nextTemp++;
    };

    auto emit_const = [&](float c) {
        int id = alloc_temp();
        oss << "    float t" << id << " = " << c << "f;\n";
        stack.push_back({id});
    };

    auto emit_var = [&](int fidx) {
        int id = alloc_temp();
        oss << "    float t" << id
            << " = ((" << fidx << " >= 0 && " << fidx
            << " < num_features) ? x[" << fidx << "] : 0.0f);\n";
        stack.push_back({id});
    };

    auto emit_unary = [&](int op) {
        if (stack.empty()) return;
        Temp a = stack.back(); stack.pop_back();
        int id = alloc_temp();
        oss << "    float t" << id
            << " = apply_op(" << op << ", t" << a.id << ", 0.0f);\n";
        stack.push_back({id});
    };

    auto emit_binary = [&](int op) {
        if (stack.size() < 2) return;
        Temp a = stack.back(); stack.pop_back();
        Temp b = stack.back(); stack.pop_back();
        int id = alloc_temp();
        oss << "    float t" << id
            << " = apply_op(" << op << ", t" << a.id << ", t" << b.id << ");\n";
        stack.push_back({id});
    };

    auto emit_ternary_if = [&]() {
        if (stack.size() < 3) return;
        Temp c = stack.back(); stack.pop_back();
        Temp b = stack.back(); stack.pop_back();
        Temp a = stack.back(); stack.pop_back();
        int id = alloc_temp();
        oss << "    float t" << id
            << " = (t" << a.id << " > 0.0f) ? t" << b.id
            << " : t" << c.id << ";\n";
        stack.push_back({id});
    };

    // Simulate prefix evaluation from len-1 down to 0
    for (int i = len - 1; i >= 0; --i) {
        int t = tokens[i];
        if (t == TOK_CONST) {
            emit_const(values[i]);
        } else if (t == TOK_VAR) {
            int fidx = static_cast<int>(values[i]);
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
        oss << "    float t0 = 0.0f;\n";
        result_id = 0;
    }

    oss << "    out[idx] = t" << result_id << ";\n";
    oss << "}\n";

    return oss.str();
}

// Compile and run the generated single-expression kernel once.
static void eval_expr_straightline_gpu(
    const int*   tokens,
    const float* values,
    int          len,
    const float* d_X,
    int          num_features,
    int          num_dps,
    float*       d_out)
{
    using clock_t = std::chrono::high_resolution_clock;

    // 1. Build kernel source
    std::string src = build_straightline_kernel_src(tokens, values, len, num_features);

    // 2. Compile with NVRTC (C++ -> PTX)
    auto t_nvrtc_start = clock_t::now();

    nvrtcProgram prog;
    NVRTC_CHECK(nvrtcCreateProgram(
        &prog,
        src.c_str(),
        "expr.cu",
        0, nullptr, nullptr));

    const char* opts[] = {
        "--std=c++17",
        "--gpu-architecture=compute_89",
        "-Isrc/utils"
    };
    nvrtcResult compile_res = nvrtcCompileProgram(prog, 3, opts);

    // Optional: log
    size_t logSize = 0;
    nvrtcGetProgramLogSize(prog, &logSize);
    if (logSize > 1) {
        std::string log(logSize, '\0');
        nvrtcGetProgramLog(prog, &log[0]);
        fprintf(stderr, "[codegen] NVRTC log:\n%s\n", log.c_str());
    }
    if (compile_res != NVRTC_SUCCESS) {
        fprintf(stderr, "[codegen] NVRTC compilation failed.\n");
        nvrtcDestroyProgram(&prog);
        exit(EXIT_FAILURE);
    }

    // 3. Get PTX
    size_t ptxSize = 0;
    NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptxSize));
    std::string ptx(ptxSize, '\0');
    NVRTC_CHECK(nvrtcGetPTX(prog, &ptx[0]));
    NVRTC_CHECK(nvrtcDestroyProgram(&prog));

    auto t_nvrtc_end = clock_t::now();
    double nvrtc_ms = std::chrono::duration<double, std::milli>(t_nvrtc_end - t_nvrtc_start).count();
    fprintf(stderr, "[single-expr] C++->PTX NVRTC time = %g ms\n", nvrtc_ms);

    // 4. Load PTX with driver API using current context (PTX -> SASS JIT)
    CUdevice  cuDevice;
    CUcontext cuContext;
    CU_CHECK(cuInit(0));
    CU_CHECK(cuDeviceGet(&cuDevice, 0));
    CU_CHECK(cuCtxGetCurrent(&cuContext));
    if (!cuContext) {
        CU_CHECK(cuCtxCreate(&cuContext, 0, cuDevice));
    }

    CUmodule   module;
    CUfunction func;

    auto t_jit_start = clock_t::now();
    CU_CHECK(cuModuleLoadData(&module, ptx.c_str()));
    CU_CHECK(cuModuleGetFunction(&func, module, "eval_expr_kernel"));
    auto t_jit_end = clock_t::now();
    double jit_ms = std::chrono::duration<double, std::milli>(t_jit_end - t_jit_start).count();
    fprintf(stderr, "[single-expr] PTX->SASS JIT (cuModuleLoadData) time = %g ms\n", jit_ms);

    // 5. Launch kernel
    void* args[] = {
        (void*)&d_X,
        (void*)&num_features,
        (void*)&num_dps,
        (void*)&d_out
    };

    int threads = 128;
    int blocks  = (num_dps + threads - 1) / threads;

    CU_CHECK(cuLaunchKernel(
        func,
        blocks, 1, 1,
        threads, 1, 1,
        0,
        0,
        args,
        nullptr));

    CU_CHECK(cuCtxSynchronize());

    // 6. Clean up module
    CU_CHECK(cuModuleUnload(module));
}

// ============================================================================
//  PTX multi-expression kernel
// ============================================================================

// Predeclare ExprDesc so we can use it before the struct definition later.
struct ExprDesc {
    const int*   tokens;
    const float* values;
    int          len;
};

// Helper: convert float to PTX hex literal.
static std::string f32_to_ptx_hex(float v) {
    uint32_t u;
    std::memcpy(&u, &v, sizeof(u));
    std::ostringstream oss;
    oss << "0f" << std::hex << std::setfill('0') << std::setw(8) << u;
    return oss.str();
}

// PTX-based multi-expression kernel.
static std::string build_multi_expr_kernel_ptx(
    const std::vector<ExprDesc>& exprs,
    int                          num_features)
{
    (void)num_features; // not needed at codegen time for PTX

    std::ostringstream oss;
    oss.setf(std::ios::scientific);

    // ---------------------------------------------------------------------
    // PTX header (match helpers.ptx/.version/.target)
    // ---------------------------------------------------------------------
    oss <<
R"(.version 8.4
.target sm_89
.address_size 64

// Prototype for apply_op helper defined in helpers.ptx
.extern .func (.param .b32 func_retval0) apply_op(
    .param .b32 apply_op_param_0,
    .param .b32 apply_op_param_1,
    .param .b32 apply_op_param_2);

.visible .entry eval_multi_expr_kernel(
    .param .u64 _X,
    .param .u32 _num_features,
    .param .u32 _num_dps,
    .param .u32 _num_exprs,
    .param .u64 _out
)
{
    .reg .pred %p<4>;
    .reg .b32  %r<16>;
    .reg .b64  %rd<12>;
    .reg .f32  %f<)" << (MAX_EVAL_STACK + 4) << R"(>;

    // Parameters for calls to apply_op(op, a, b)
    .param .b32 _ret;
    .param .b32 _op;
    .param .b32 _a;
    .param .b32 _b;

    // Load parameters
    ld.param.u64 %rd0, [_X];            // X base
    ld.param.u32 %r0,  [_num_features]; // num_features
    ld.param.u32 %r1,  [_num_dps];      // num_dps
    ld.param.u64 %rd1, [_out];          // out base

    // Compute global thread index: idx = blockIdx.x * blockDim.x + threadIdx.x
    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mad.lo.s32 %r5, %r3, %r4, %r2;      // r5 = idx

    // if (idx >= num_dps) return;
    setp.ge.u32 %p0, %r5, %r1;
    @%p0 bra DONE;

    // Compute pointer to this datapoint's feature row:
    // row_elems = idx * num_features
    // row_bytes = row_elems * 4
    // rd3 = X + row_bytes
    mul.lo.u32   %r6,  %r5, %r0;
    mul.wide.u32 %rd2, %r6, 4;
    add.s64      %rd3, %rd0, %rd2;
)";

    const size_t num_exprs = exprs.size();

    // ---------------------------------------------------------------------
    // Per-expression straight-line PTX, stack-based register allocation
    // ---------------------------------------------------------------------
    for (size_t e = 0; e < num_exprs; ++e) {
        const ExprDesc& ed = exprs[e];

        oss << "\n    // ----- Expression " << e << " -----\n";
        oss << "    // len = " << ed.len << "\n";
        oss << "    // Compute out index for expr " << e << ": out[e * num_dps + idx]\n";
        oss << "    mov.u32 %r7, " << (unsigned int)e << ";\n";
        oss << "    mul.lo.u32 %r7, %r7, %r1;\n";
        oss << "    add.u32 %r7, %r7, %r5;\n";
        oss << "    mul.wide.u32 %rd4, %r7, 4;\n";
        oss << "    add.s64 %rd5, %rd1, %rd4;\n\n";

        struct Temp { int reg; };
        std::vector<Temp> stack;
        stack.reserve(MAX_EVAL_STACK);

        auto push_const = [&](float c) {
            if (stack.size() >= MAX_EVAL_STACK) {
                int reg = MAX_EVAL_STACK - 1;
                oss << "    // CONST (clamped) " << c << "\n";
                oss << "    mov.f32 %f" << reg << ", " << f32_to_ptx_hex(c) << ";\n";
                stack.push_back({reg});
                return;
            }
            int reg = static_cast<int>(stack.size());
            oss << "    // CONST " << c << "\n";
            oss << "    mov.f32 %f" << reg << ", " << f32_to_ptx_hex(c) << ";\n";
            stack.push_back({reg});
        };

        auto push_var = [&](int fidx) {
            if (stack.size() >= MAX_EVAL_STACK) {
                int reg = MAX_EVAL_STACK - 1;
                oss << "    // VAR (clamped) x[" << fidx << "]\n";
                oss << "    mov.u32 %r8, " << fidx << ";\n";
                oss << "    mul.wide.u32 %rd6, %r8, 4;\n";
                oss << "    add.s64 %rd6, %rd3, %rd6;\n";
                oss << "    ld.global.f32 %f" << reg << ", [%rd6];\n";
                stack.push_back({reg});
                return;
            }
            int reg = static_cast<int>(stack.size());
            oss << "    // VAR x[" << fidx << "]\n";
            oss << "    mov.u32 %r8, " << fidx << ";\n";
            oss << "    mul.wide.u32 %rd6, %r8, 4;\n";
            oss << "    add.s64 %rd6, %rd3, %rd6;\n";
            oss << "    ld.global.f32 %f" << reg << ", [%rd6];\n";
            stack.push_back({reg});
        };

        auto emit_unary = [&](int op) {
            if (stack.empty()) return;
            Temp a = stack.back();
            int  r = a.reg;

            // Call helpers.ptx: apply_op(op, a, 0.0f)
            oss << "    // UNARY op=" << op << " via helpers.ptx apply_op\n";
            oss << "    mov.u32 %r10, " << op << ";\n";
            oss << "    st.param.b32 [_op], %r10;\n";
            oss << "    st.param.f32 [_a], %f" << r << ";\n";
            oss << "    mov.f32 %f" << (MAX_EVAL_STACK + 3) << ", 0f00000000;\n";
            oss << "    st.param.f32 [_b], %f" << (MAX_EVAL_STACK + 3) << ";\n";
            oss << "    call.uni (_ret), apply_op, (_op, _a, _b);\n";
            oss << "    ld.param.f32 %f" << r << ", [_ret];\n";
        };

        auto emit_binary = [&](int op) {
            if (stack.size() < 2) return;
            Temp a = stack.back(); stack.pop_back();
            Temp b = stack.back(); stack.pop_back();
            int ra = a.reg;
            int rb = b.reg;

            // Call helpers.ptx: apply_op(op, a, b)
            oss << "    // BINARY op=" << op << " via helpers.ptx apply_op\n";
            oss << "    mov.u32 %r10, " << op << ";\n";
            oss << "    st.param.b32 [_op], %r10;\n";
            oss << "    st.param.f32 [_a], %f" << ra << ";\n";
            oss << "    st.param.f32 [_b], %f" << rb << ";\n";
            oss << "    call.uni (_ret), apply_op, (_op, _a, _b);\n";
            oss << "    ld.param.f32 %f" << rb << ", [_ret];\n";
            stack.push_back({rb});
        };

        auto emit_ternary_if = [&]() {
            if (stack.size() < 3) return;
            Temp c = stack.back(); stack.pop_back();
            Temp b = stack.back(); stack.pop_back();
            Temp a = stack.back(); stack.pop_back();
            int rc = c.reg;
            int rb = b.reg;
            int ra = a.reg;

            oss << "    // IF (a>0 ? b : c)\n";
            oss << "    setp.gt.f32 %p1, %f" << ra << ", 0f00000000;\n";
            oss << "    selp.f32 %f" << ra << ", %f" << rb
                << ", %f" << rc << ", %p1;\n";

            stack.push_back({ra});
        };

        // Simulate prefix evaluation from len-1 down to 0
        for (int i = ed.len - 1; i >= 0; --i) {
            int tok = ed.tokens[i];
            if (tok == TOK_CONST) {
                push_const(ed.values[i]);
            } else if (tok == TOK_VAR) {
                int fidx = static_cast<int>(ed.values[i]);
                push_var(fidx);
            } else if (tok == Function::IF) {
                emit_ternary_if();
            } else {
                int ar = op_arity(tok);
                if (ar == 1) {
                    emit_unary(tok);
                } else if (ar == 2) {
                    emit_binary(tok);
                } else {
                    emit_ternary_if();
                }
            }
        }

        int result_reg = 0;
        if (!stack.empty()) {
            result_reg = stack.back().reg;
        } else {
            oss << "    mov.f32 %f0, 0f00000000;\n";
            result_reg = 0;
        }

        oss << "    // Store result of expr " << e << "\n";
        oss << "    st.global.f32 [%rd5], %f" << result_reg << ";\n";
    }

    // Epilogue
    oss <<
R"(
DONE:
    ret;
}
)";

    return oss.str();
}


// ============================================================================
//  Multi-expression straight-line kernel (NVRTC C++ version)
// ============================================================================

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

    auto worker_fn = [&](size_t start_e, size_t end_e) {
        for (size_t e = start_e; e < end_e; ++e) {
            const ExprDesc& ed = exprs[e];

            std::ostringstream expr_oss;
            expr_oss.setf(std::ios::scientific);

            expr_oss << "    // ----- Expression " << e << " -----\n";
            expr_oss << "    {\n";

            struct Temp { int id; };
            std::vector<Temp> stack;
            int nextTemp = 0;

            auto alloc_temp = [&]() -> int {
                if (nextTemp >= MAX_EVAL_STACK) {
                    throw std::runtime_error("MAX_EVAL_STACK exceeded in build_multi_expr_kernel_src");
                }
                return nextTemp++;
            };

            auto emit_const = [&](float c) {
                int id = alloc_temp();
                expr_oss << "        float t" << id << " = " << c << "f;\n";
                stack.push_back({id});
            };

            auto emit_var = [&](int fidx) {
                int id = alloc_temp();
                expr_oss << "        float t" << id
                         << " = ((" << fidx << " >= 0 && " << fidx
                         << " < num_features) ? x[" << fidx << "] : 0.0f);\n";
                stack.push_back({id});
            };

            auto emit_unary = [&](int op) {
                if (stack.empty()) return;
                Temp a = stack.back(); stack.pop_back();
                int id = alloc_temp();
                expr_oss << "        float t" << id
                         << " = apply_op(" << op << ", t" << a.id << ", 0.0f);\n";
                stack.push_back({id});
            };

            auto emit_binary = [&](int op) {
                if (stack.size() < 2) return;
                Temp a = stack.back(); stack.pop_back();
                Temp b = stack.back(); stack.pop_back();
                int id = alloc_temp();
                expr_oss << "        float t" << id
                         << " = apply_op(" << op << ", t" << a.id << ", t" << b.id << ");\n";
                stack.push_back({id});
            };

            auto emit_ternary_if = [&]() {
                if (stack.size() < 3) return;
                Temp c = stack.back(); stack.pop_back();
                Temp b = stack.back(); stack.pop_back();
                Temp a = stack.back(); stack.pop_back();
                int id = alloc_temp();
                expr_oss << "        float t" << id
                         << " = (t" << a.id << " > 0.0f) ? t" << b.id
                         << " : t" << c.id << ";\n";
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
                expr_oss << "        float t0 = 0.0f;\n";
                result_id = 0;
            }

            expr_oss << "        out[" << e
                     << " * num_dps + idx] = t" << result_id << ";\n";
            expr_oss << "    }\n\n";

            expr_blocks[e] = expr_oss.str();
        }
    };

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

    for (auto &th : workers) {
        th.join();
    }

    for (size_t e = 0; e < num_exprs; ++e) {
        oss << expr_blocks[e];
    }

    oss << "}\n";
    return oss.str();
}

// Compile and run a multi-expression kernel once using PTX codegen.
static double eval_multi_expr_straightline_gpu(
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
    auto t_build_start = clock_t::now();

    // 1. Build PTX source (CPU codegen)
    std::string ptx = build_multi_expr_kernel_ptx(exprs, num_features);

    auto t_build_end = clock_t::now();
    double build_ms = std::chrono::duration<double, std::milli>(
        t_build_end - t_build_start).count();

    // 2. Load PTX in current context and JIT (PTX -> SASS)
    CUdevice  cuDevice;
    CUcontext cuContext;
    CU_CHECK(cuInit(0));
    CU_CHECK(cuDeviceGet(&cuDevice, 0));
    CU_CHECK(cuCtxGetCurrent(&cuContext));
    if (!cuContext) {
        CU_CHECK(cuCtxCreate(&cuContext, 0, cuDevice));
    }

    CUmodule   module;
    CUfunction func;

    float wall_time_ms = 0.0f;

    static char error_log[8192] = {0};
    static char info_log[8192]  = {0};

    CUjit_option jit_opts[] = {
        CU_JIT_WALL_TIME,
        CU_JIT_ERROR_LOG_BUFFER,
        CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_INFO_LOG_BUFFER,
        CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES
    };

    void* jit_optvals[] = {
        &wall_time_ms,
        error_log,
        (void*)(long)sizeof(error_log),
        info_log,
        (void*)(long)sizeof(info_log)
    };

    auto t_jit_start = clock_t::now();

    CUlinkState linkState;
    
    CU_LINK_CHECK(cuLinkCreate((unsigned int)(sizeof(jit_opts) / sizeof(jit_opts[0])),
                          jit_opts,
                          jit_optvals,
                          &linkState), linkState, error_log);

    CU_LINK_CHECK(
        cuLinkAddData(linkState,
                      CU_JIT_INPUT_PTX,
                      (void*)ptx.c_str(),
                      ptx.size(),
                      "eval_multi_expr_ptx",
                      0,          // numOptions
                      nullptr,    // options
                      nullptr),   // optionValues
        linkState,
        error_log);

    // Link in precompiled helpers PTX (asin_f32/acos_f32/atan_f32/exp_f32/etc.).
    CU_LINK_CHECK(
        cuLinkAddFile(linkState,
                      CU_JIT_INPUT_PTX,
                      "src/utils/helpers.ptx",
                      0,          // numOptions
                      nullptr,    // options
                      nullptr),   // optionValues
        linkState,
        error_log);

    void*  cubin_data  = nullptr;
    size_t cubin_size  = 0;
    CU_CHECK(cuLinkComplete(linkState, &cubin_data, &cubin_size));

    CU_CHECK(cuModuleLoadData(&module, cubin_data));

    fprintf(stderr, "[PTX JIT] info log:\n%s\n", info_log);
    fprintf(stderr, "[PTX JIT] error log:\n%s\n", error_log);

    CU_CHECK(cuLinkDestroy(linkState));

    CU_CHECK(cuModuleGetFunction(&func, module, "eval_multi_expr_kernel"));

    auto t_jit_end = clock_t::now();
    double jit_ms = std::chrono::duration<double, std::milli>(
        t_jit_end - t_jit_start).count();

    // 3. Launch kernel
    int num_exprs_i = static_cast<int>(exprs.size());
    void* args[] = {
        (void*)&d_X,
        (void*)&num_features,
        (void*)&num_dps,
        (void*)&num_exprs_i,
        (void*)&d_out
    };

    int threads = 128;
    int blocks  = (num_dps + threads - 1) / threads;

    auto t_launch_start = clock_t::now();

    CU_CHECK(cuLaunchKernel(
        func,
        blocks, 1, 1,
        threads, 1, 1,
        0,
        0,
        args,
        nullptr));

    CU_CHECK(cuCtxSynchronize());
    CU_CHECK(cuModuleUnload(module));

    auto t_launch_end = clock_t::now();
    double launch_ms = std::chrono::duration<double, std::milli>(
        t_launch_end - t_launch_start).count();

    fprintf(stderr,
            "[multi-expr-ptx] timings: build_ptx=%g ms, PTX->SASS JIT=%g ms (reported %g ms), launch+sync=%g ms (num_exprs=%d, num_dps=%d)\n",
            build_ms, jit_ms, (double)wall_time_ms, launch_ms, num_exprs_i, num_dps);

    if (compile_ms_p) *compile_ms_p = build_ms;
    if (jit_ms_p)     *jit_ms_p     = (double)wall_time_ms;

    return launch_ms;
}

// ============================================================================
//  Evolution path using single-expression codegen (unchanged below)
// ============================================================================

void evolve(InputInfo &input_info,
            double ***all_vars,
            double **all_predictions,
            int expr_idx,
            int pop_size,
            int num_generations,
            int maxGPLen,
            unsigned int constSamplesLen,
            float outProb,
            float constProb,
            const unsigned int* h_keys,
            const float* h_depth2leaf,
            const float* h_roulette,
            const float* h_consts)
{
    if (expr_idx < 0 || expr_idx >= input_info.num_exprs) {
        fprintf(stderr, "evolve: invalid expr_idx %d (num_exprs=%d)\n",
                expr_idx, input_info.num_exprs);
        return;
    }

    int num_vars = input_info.num_vars[expr_idx];
    int num_dps  = input_info.num_dps[expr_idx];

    if (num_vars <= 0 || num_dps <= 0 || maxGPLen <= 0 ||
        pop_size <= 0 || num_generations <= 0) {
        fprintf(stderr,
                "evolve: invalid parameters (num_vars=%d, num_dps=%d, maxGPLen=%d, pop=%d, gens=%d)\n",
                num_vars, num_dps, maxGPLen, pop_size, num_generations);
        return;
    }

    int varLen = num_vars;
    int outLen = 1;

    int dev_count = 0;
    cudaError_t cerr = cudaGetDeviceCount(&dev_count);
    fprintf(stderr, "[debug] cudaGetDeviceCount: err=%d (%s), count=%d\n",
            (int)cerr, cudaGetErrorString(cerr), dev_count);

    cerr = cudaSetDevice(0);
    fprintf(stderr, "[debug] cudaSetDevice(0): err=%d (%s)\n",
            (int)cerr, cudaGetErrorString(cerr));
    CUDA_CHECK(cerr);

    // Build input matrix X_inputs [num_dps, varLen] and labels [num_dps]
    std::vector<float> h_X_inputs((size_t)num_dps * (size_t)varLen);
    std::vector<float> h_labels(num_dps);
    for (int dp = 0; dp < num_dps; ++dp) {
        for (int v = 0; v < varLen; ++v) {
            h_X_inputs[(size_t)dp * (size_t)varLen + v] =
                (float)all_vars[expr_idx][v][dp];
        }
        h_labels[dp] = (float)all_vars[expr_idx][varLen][dp];
    }

    float   *d_val  = nullptr;
    int16_t *d_type = nullptr;
    int16_t *d_size = nullptr;
    CUDA_CHECK(cudaMalloc(&d_val,  (size_t)pop_size * (size_t)maxGPLen * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_type, (size_t)pop_size * (size_t)maxGPLen * sizeof(int16_t)));
    CUDA_CHECK(cudaMalloc(&d_size, (size_t)pop_size * (size_t)maxGPLen * sizeof(int16_t)));

    unsigned int *d_keys       = nullptr;
    float        *d_depth2leaf = nullptr;
    float        *d_roulette   = nullptr;
    float        *d_consts     = nullptr;

    CUDA_CHECK(cudaMalloc(&d_keys,       2 * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_depth2leaf, MAX_FULL_DEPTH * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_roulette,   Function::END * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_consts,     constSamplesLen * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_keys, h_keys,
                          2 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_depth2leaf, h_depth2leaf,
                          MAX_FULL_DEPTH * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_roulette, h_roulette,
                          Function::END * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_consts, h_consts,
                          constSamplesLen * sizeof(float), cudaMemcpyHostToDevice));

    generate((unsigned int)pop_size,
             (unsigned int)maxGPLen,
             (unsigned int)varLen,
             (unsigned int)outLen,
             constSamplesLen,
             outProb,
             constProb,
             d_keys,
             d_depth2leaf,
             d_roulette,
             d_consts,
             d_val,
             d_type,
             d_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    float *d_X = nullptr;
    float *d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_X, (size_t)num_dps * (size_t)varLen * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, (size_t)num_dps * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_X, h_X_inputs.data(),
                          h_X_inputs.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_labels.data(),
                          h_labels.size() * sizeof(float), cudaMemcpyHostToDevice));

    std::vector<int>    h_tokens(maxGPLen);
    std::vector<float>  h_values(maxGPLen);
    std::vector<double> h_values_d(maxGPLen);

    std::vector<float>   h_pop_val((size_t)pop_size * (size_t)maxGPLen);
    std::vector<int16_t> h_pop_type((size_t)pop_size * (size_t)maxGPLen);
    std::vector<int16_t> h_pop_size((size_t)pop_size * (size_t)maxGPLen);

    float *d_out_multi = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out_multi,
                          (size_t)pop_size * (size_t)num_dps * sizeof(float)));
    std::vector<float> h_out_multi((size_t)pop_size * (size_t)num_dps);

    float best_fitness = std::numeric_limits<float>::infinity();
    int   best_idx     = -1;

    std::vector<float> kernel_times_ms;
    std::vector<float> compile_times_ms;
    std::vector<float> jit_times_ms;
    kernel_times_ms.reserve((size_t)num_generations);
    compile_times_ms.reserve((size_t)num_generations);
    jit_times_ms.reserve((size_t)num_generations);

    for (int gen = 0; gen < num_generations; ++gen) {
        CUDA_CHECK(cudaMemcpy(h_pop_val.data(), d_val,
                              h_pop_val.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_pop_type.data(), d_type,
                              h_pop_type.size() * sizeof(int16_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_pop_size.data(), d_size,
                              h_pop_size.size() * sizeof(int16_t), cudaMemcpyDeviceToHost));

        std::vector<ExprDesc> exprs;
        exprs.reserve(pop_size);
        std::vector<int>   all_tokens;
        std::vector<float> all_values;
        all_tokens.reserve((size_t)pop_size * (size_t)maxGPLen);
        all_values.reserve((size_t)pop_size * (size_t)maxGPLen);
        std::vector<int> expr_to_pop;
        expr_to_pop.reserve(pop_size);

        for (int i = 0; i < pop_size; ++i) {
            const float   *val_i  = &h_pop_val[(size_t)i * (size_t)maxGPLen];
            const int16_t *type_i = &h_pop_type[(size_t)i * (size_t)maxGPLen];
            const int16_t *sub_i  = &h_pop_size[(size_t)i * (size_t)maxGPLen];

            int len = (int)sub_i[0];
            if (len <= 0 || len > maxGPLen) continue;

            size_t base = all_tokens.size();
            for (int t = 0; t < len; ++t) {
                int16_t node_type = (int16_t)(type_i[t] & NodeType::TYPE_MASK);
                float   node_val  = val_i[t];
                int     tok;
                float   val;
                if (node_type == NodeType::CONST) {
                    tok = TOK_CONST;
                    val = node_val;
                } else if (node_type == NodeType::VAR) {
                    tok = TOK_VAR;
                    val = node_val;
                } else {
                    tok = static_cast<int>(node_val);
                    val = 0.0f;
                }
                all_tokens.push_back(tok);
                all_values.push_back(val);
                h_tokens[t]   = tok;
                h_values[t]   = val;
                h_values_d[t] = static_cast<double>(val);
            }

            std::string formula = format_formula(h_tokens.data(), h_values_d.data(), len);
            // fprintf(stderr, "[evolve] gen=%d ind=%d len=%d formula=%s\n",
            //         gen, i, len, formula.c_str());

            ExprDesc ed;
            ed.tokens = all_tokens.data() + base;
            ed.values = all_values.data() + base;
            ed.len    = len;
            exprs.push_back(ed);
            expr_to_pop.push_back(i);
        }

        if (!exprs.empty()) {
            double compile_ms = 0.0;
            double jit_ms     = 0.0;
            double launch_ms  = eval_multi_expr_straightline_gpu(
                exprs,
                d_X,
                varLen,
                num_dps,
                d_out_multi,
                &compile_ms,
                &jit_ms);
            kernel_times_ms.push_back((float)launch_ms);
            compile_times_ms.push_back((float)compile_ms);
            jit_times_ms.push_back((float)jit_ms);

            CUDA_CHECK(cudaMemcpy(h_out_multi.data(), d_out_multi,
                                  (size_t)exprs.size() * (size_t)num_dps * sizeof(float),
                                  cudaMemcpyDeviceToHost));

            for (size_t e = 0; e < exprs.size(); ++e) {
                int pop_i = expr_to_pop[e];
                const float* preds = &h_out_multi[e * (size_t)num_dps];
                double mse = 0.0;
                for (int dp = 0; dp < num_dps; ++dp) {
                    double diff = (double)preds[dp] - (double)h_labels[dp];
                    mse += diff * diff;
                }
                mse /= (double)num_dps;

                if ((float)mse < best_fitness) {
                    best_fitness = (float)mse;
                    best_idx     = pop_i;
                }
            }
        }

        fprintf(stderr, "[evolve] generation %d best MSE=%g (idx=%d)\n",
                gen, best_fitness, best_idx);
    }

    auto summarize_times = [](const std::vector<float>& times,
                              const char* label) {
        if (times.empty()) return;
        float total_ms = 0.0f;
        for (float v : times) total_ms += v;
        float first_ms = times.front();

        std::vector<float> sorted_all = times;
        std::sort(sorted_all.begin(), sorted_all.end());
        float  median_ms = 0.0f;
        size_t n         = sorted_all.size();
        if (n % 2 == 1)
            median_ms = sorted_all[n / 2];
        else
            median_ms = 0.5f * (sorted_all[n / 2 - 1] + sorted_all[n / 2]);

        float  mean_excl_first_ms = 0.0f;
        size_t n_excl             = 0;
        if (times.size() > 1) {
            for (size_t i = 1; i < times.size(); ++i) {
                mean_excl_first_ms += times[i];
            }
            n_excl = times.size() - 1;
            mean_excl_first_ms /= (float)n_excl;
        }

        fprintf(stderr,
                "%s: total=%g ms, first=%g ms, median_all=%g ms, "
                "mean_excl_first=%g ms over %zu evals (%zu excl first)\n",
                label,
                (double)total_ms, (double)first_ms, (double)median_ms,
                (double)mean_excl_first_ms, n, n_excl);
    };

    summarize_times(kernel_times_ms,  "[evolve] kernel time (launch+sync)");
    summarize_times(compile_times_ms, "[evolve] compile time (C++->PTX NVRTC)");

    if (best_idx >= 0) {
        const float   *val_i  = &h_pop_val[(size_t)best_idx * (size_t)maxGPLen];
        const int16_t *type_i = &h_pop_type[(size_t)best_idx * (size_t)maxGPLen];
        const int16_t *sub_i  = &h_pop_size[(size_t)best_idx * (size_t)maxGPLen];

        int len = (int)sub_i[0];
        if (len > 0 && len <= maxGPLen) {
            for (int t = 0; t < len; ++t) {
                int16_t node_type = (int16_t)(type_i[t] & NodeType::TYPE_MASK);
                float   node_val  = val_i[t];
                if (node_type == NodeType::CONST) {
                    h_tokens[t] = TOK_CONST;
                    h_values[t] = node_val;
                } else if (node_type == NodeType::VAR) {
                    h_tokens[t] = TOK_VAR;
                    h_values[t] = node_val;
                } else {
                    h_tokens[t] = (int)node_val;
                    h_values[t] = 0.0f;
                }
            }
            std::vector<float> h_best_out(num_dps);
            eval_expr_straightline_gpu(
                h_tokens.data(),
                h_values.data(),
                len,
                d_X,
                varLen,
                num_dps,
                d_out_multi);

            CUDA_CHECK(cudaMemcpy(h_best_out.data(), d_out_multi,
                                  (size_t)num_dps * sizeof(float), cudaMemcpyDeviceToHost));
            for (int dp = 0; dp < num_dps; ++dp) {
                all_predictions[expr_idx][dp] = (double)h_best_out[dp];
            }
        }
    }

    cudaFree(d_out_multi);
    cudaFree(d_X);
    cudaFree(d_y);
    cudaFree(d_val);
    cudaFree(d_type);
    cudaFree(d_size);
    cudaFree(d_keys);
    cudaFree(d_depth2leaf);
    cudaFree(d_roulette);
    cudaFree(d_consts);
}

// Evolution-based batch entry point.
void eval_evolve_jinha_batch(InputInfo &input_info,
                             double ***all_vars,
                             double **all_predictions)
{
    const int pop_size        = 256;
    const int num_generations = 5;
    const int maxGPLen        = 60;
    const unsigned int constSamplesLen = 8;
    const float outProb   = 0.0f;
    const float constProb = 0.01f;

    unsigned int h_keys[2] = {42u, 1337u};
    float h_depth[MAX_FULL_DEPTH];
    for (int i = 0; i < MAX_FULL_DEPTH; ++i) h_depth[i] = 0.0001f;

    std::vector<float> h_roulette(Function::LOOSE_SQRT + 1);
    h_roulette[0] = 0.0f;
    for (int i = 1; i <= Function::LOOSE_SQRT; ++i)
        h_roulette[i] =
            static_cast<float>(i) / static_cast<float>(Function::LOOSE_SQRT);

    float h_consts[constSamplesLen] = {-1.f, -0.5f, 0.f, 0.5f, 1.f, 2.f, 3.f, 4.f};

    evolve(input_info,
           all_vars,
           all_predictions,
           0,
           pop_size,
           num_generations,
           maxGPLen,
           constSamplesLen,
           outProb,
           constProb,
           h_keys,
           h_depth,
           h_roulette.data(),
           h_consts);
}

// multi-expr PTX batch path
void eval_multi_expr_ptx_batch(InputInfo &input_info,
                               double ***all_vars,
                               double **all_predictions)
{
    double alloc_ms = 0.0;
    double memcpy_h2d_ms_total = 0.0;
    double memcpy_d2h_ms_total = 0.0;

    const int num_exprs = input_info.num_exprs;
    if (num_exprs <= 0) {
        return;
    }

    // Evaluate each expression independently on its own X dataset
    for (int expr_id = 0; expr_id < num_exprs; ++expr_id) {
        const int num_tokens   = input_info.num_tokens[expr_id];
        const int num_vars     = input_info.num_vars[expr_id];
        const int num_dps      = input_info.num_dps[expr_id];
        const int num_features = num_vars + 1;  // vars + label

        if (num_tokens <= 0 || num_dps <= 0 || num_features <= 0) {
            continue;
        }

        // Tokens buffer (prefer packed)
        const int* tokens_host_ptr = nullptr;
        std::vector<int> tokens_temp;
        if (input_info.tokens_packed && input_info.tokens_packed[expr_id]) {
            tokens_host_ptr = input_info.tokens_packed[expr_id];
        } else {
            tokens_temp.resize(num_tokens);
            for (int i = 0; i < num_tokens; ++i) {
                tokens_temp[i] = input_info.tokens[expr_id][i];
            }
            tokens_host_ptr = tokens_temp.data();
        }

        // Values buffer (prefer packed float)
        const float* values_host_ptr = nullptr;
        std::vector<float> values_temp;
        if (input_info.values_packed_f32 && input_info.values_packed_f32[expr_id]) {
            values_host_ptr = input_info.values_packed_f32[expr_id];
        } else {
            values_temp.resize(num_tokens);
            for (int i = 0; i < num_tokens; ++i) {
                values_temp[i] = (float)input_info.values[expr_id][i];
            }
            values_host_ptr = values_temp.data();
        }

        // X buffer for this expression (prefer pre-packed)
        const float* X_src = nullptr;
        std::vector<float> X_host;
        if (input_info.X_packed_f32 && input_info.X_packed_f32[expr_id]) {
            X_src = input_info.X_packed_f32[expr_id];
        } else {
            X_host.resize((size_t)num_dps * (size_t)num_features);
            for (int dp = 0; dp < num_dps; ++dp) {
                for (int v = 0; v < num_features; ++v) {
                    X_host[(size_t)dp * (size_t)num_features + v] =
                        (float)all_vars[expr_id][v][dp];
                }
            }
            X_src = X_host.data();
        }

        // Device buffers for this expression
        float *d_X   = nullptr;
        float *d_out = nullptr;

        TimePoint t_alloc = measure_clock();
        CUDA_CHECK(cudaMalloc(&d_X,   (size_t)num_dps * (size_t)num_features * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, (size_t)num_dps * sizeof(float)));
        CUDA_CHECK(cudaDeviceSynchronize());
        alloc_ms += clock_to_ms(t_alloc, measure_clock());

        // Host -> device copy for X
        TimePoint t_h2d = measure_clock();
        CUDA_CHECK(cudaMemcpy(d_X,
                              X_src,
                              (size_t)num_dps * (size_t)num_features * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
        memcpy_h2d_ms_total += clock_to_ms(t_h2d, measure_clock());

        // Evaluate this expression via single-expression straight-line kernel
        eval_expr_straightline_gpu(tokens_host_ptr,
                                   values_host_ptr,
                                   num_tokens,
                                   d_X,
                                   num_features,
                                   num_dps,
                                   d_out);

        // Copy predictions back
        std::vector<float> out_host((size_t)num_dps);
        TimePoint t_d2h = measure_clock();
        CUDA_CHECK(cudaMemcpy(out_host.data(),
                              d_out,
                              (size_t)num_dps * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());
        memcpy_d2h_ms_total += clock_to_ms(t_d2h, measure_clock());

        for (int dp = 0; dp < num_dps; ++dp) {
            all_predictions[expr_id][dp] = (double)out_host[dp];
        }

        CUDA_CHECK(cudaFree(d_X));
        CUDA_CHECK(cudaFree(d_out));
    }

    std::cout << "Alloc time (host wall): " << alloc_ms << " ms" << std::endl;
    std::cout << "Total H2D memcpy time (host wall): " << memcpy_h2d_ms_total << " ms" << std::endl;
    std::cout << "Total D2H memcpy time (host wall): " << memcpy_d2h_ms_total << " ms" << std::endl;
    std::cout << "(Per-expression NVRTC/driver PTX build & JIT times are logged by eval_expr_straightline_gpu)" << std::endl;
}
