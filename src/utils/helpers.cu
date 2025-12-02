// Device math helpers and opcode dispatcher shared across kernels.
// nvcc -arch=sm_89 -dc -ptx src/utils/helpers.cu -o src/utils/helpers.ptx

#include <math.h>

// Match values used in gpu_custom_kernel_per_expression.cu
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

// Opcode constants must match defs.h / opcodes.h
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

// Main opcode dispatcher, exported with C linkage for PTX calls if needed.
extern "C" __device__ float apply_op(int op, float a, float b) {
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

// Thin C-linkage wrappers for trig/exp, usable from PTX or other code.
// extern "C" __device__ float asin_f32(float x) { return asinf(x); }
// extern "C" __device__ float acos_f32(float x) { return acosf(x); }
// extern "C" __device__ float atan_f32(float x) { return atanf(x); }
// extern "C" __device__ float exp_f32 (float x) { return expf(x); }
