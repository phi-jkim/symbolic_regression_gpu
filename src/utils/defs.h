#pragma once
// defs.h for symbolic_regression_gpu – GPU evolution helpers
// Adapted from evogp project but re-mapped to symbolic_regression op-codes
// Provides NodeType / Function enums and common constants used by GPU
// generate / mutate / crossover kernels.

#include <stdint.h>
#include "opcodes.h"

// -----------------------------------------------------------------------------
// Global constants (stack sizes, safety numbers)
// -----------------------------------------------------------------------------
#ifndef MAX_FULL_DEPTH
#define MAX_FULL_DEPTH 10
#endif

#ifndef DELTA
#define DELTA 1e-9f
#endif

#ifndef MAX_VAL
#define MAX_VAL 1e9f
#endif

constexpr int MAX_STACK      = 1024;   // per-thread scratch memory (matches evogp)

// -----------------------------------------------------------------------------
// Node type encoding (7 low bits store the arity category, high bit marks OUT)
// -----------------------------------------------------------------------------
enum NodeType : int16_t {
    VAR        = 0,      // variable leaf
    CONST      = 1,      // constant leaf
    UFUNC      = 2,      // unary   function
    BFUNC      = 3,      // binary  function
    TFUNC      = 4,      // ternary function (currently unused in SR)
    TYPE_MASK  = 0x7F,   // mask to drop OUT_NODE flag
    OUT_NODE   = 1 << 7, // flag for multi-output representation
    UFUNC_OUT  = UFUNC + OUT_NODE,
    BFUNC_OUT  = BFUNC + OUT_NODE,
    TFUNC_OUT  = TFUNC + OUT_NODE
};

// -----------------------------------------------------------------------------
// Function opcode enumeration used by GPU kernels.
// The numeric values are mapped to the symbolic_regression op-codes declared
// in opcodes.h so that the evaluator can re-use the same encoding.
// IF / comparison ops are kept to preserve indices expected by the original
// kernels but they are not generated (roulette weight should be zero).
// -----------------------------------------------------------------------------
enum Function : int16_t {
    IF          = 0,                // ternary (not used in SR)
    ADD         = OP_ADD,           // 1
    SUB         = OP_SUB,           // 2
    MUL         = OP_MUL,           // 3
    DIV         = OP_DIV,           // 4
    POW         = OP_POW,           // 5
    MIN         = OP_MIN,           // 6
    MAX         = OP_MAX,           // 7
    LOOSE_DIV   = OP_LOOSE_DIV,     // 8
    LOOSE_POW   = OP_LOOSE_POW,     // 9
    // Unary block ------------------------------------------------------------
    SIN         = OP_SIN,           // 10
    COS         = OP_COS,           // 11
    TAN         = OP_TAN,           // 12
    SINH        = OP_SINH,          // 13
    COSH        = OP_COSH,          // 14
    TANH        = OP_TANH,          // 15
    EXP         = OP_EXP,           // 16
    LOG         = OP_LOG,           // 17
    INV         = OP_INV,           // 18
    ASIN        = OP_ASIN,          // 19
    ACOS        = OP_ACOS,          // 20
    ATAN        = OP_ATAN,          // 21
    LOOSE_LOG   = OP_LOOSE_LOG,     // 22
    LOOSE_INV   = OP_LOOSE_INV,     // 23
    ABS         = OP_ABS,           // 24
    NEG         = OP_NEG,           // 25
    SQRT        = OP_SQRT,          // 26
    LOOSE_SQRT  = OP_LOOSE_SQRT,    // 27
    END         = 28,               // marker
    // LT          = 28, (not used)
    // GT          = 29,
    // LE          = 30,
    // GE          = 31,
};

static_assert(ADD == 1 && SUB == 2 && MUL == 3, "Opcode remap failed – check defs.h");

// -----------------------------------------------------------------------------
// Helper structs matching evogp for tree storage
// -----------------------------------------------------------------------------
struct GPNode {
    float   value;        // constant, variable index, or opcode packed as float
    int16_t nodeType;     // NodeType enum with OUT flag if any
    int16_t subtreeSize;  // #nodes in subtree rooted at this node (including self)
};

struct OutNodeValue {
    int16_t function;   // lower 16 bits
    int16_t outIndex;   // upper 16 bits
    __host__ __device__ inline operator float() const { return *(const float*)this; }
};

struct NchildDepth {
    int16_t childs;  // #children left to generate
    int16_t depth;   // current depth
};

// simple FNV-1a style 3-int hash (same as evogp::hash)
constexpr size_t _FNV_offset_basis = 14695981039346656037ULL;
constexpr size_t _FNV_prime        = 1099511628211ULL;
__host__ __device__ inline unsigned int hash(const unsigned int n, const unsigned int k1, const unsigned int k2) {
    const unsigned int a[3]{n,k1,k2};
    auto h = _FNV_offset_basis;
    auto b = &reinterpret_cast<const unsigned char&>(a);
    constexpr auto C = sizeof(unsigned int)*3;
    for(size_t i=0;i<C;++i){ h ^= static_cast<size_t>(b[i]); h*= _FNV_prime; }
    return (unsigned int)h;
}

// #define RandomEngine thrust::random::taus88
