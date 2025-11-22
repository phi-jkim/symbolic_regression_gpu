#pragma once

#ifdef __CUDACC__
#define SR_HD __host__ __device__ inline
#else
#define SR_HD inline
#endif

// Special tokens
constexpr int TOK_CONST = 0;
constexpr int TOK_VAR   = -1;

// Operator codes (prefix evaluation order)
constexpr int OP_ADD        = 1;
constexpr int OP_SUB        = 2;
constexpr int OP_MUL        = 3;
constexpr int OP_DIV        = 4;
constexpr int OP_POW        = 5;
constexpr int OP_MIN        = 6;
constexpr int OP_MAX        = 7;
constexpr int OP_LOOSE_DIV  = 8;
constexpr int OP_LOOSE_POW  = 9;
constexpr int OP_SIN        = 10;
constexpr int OP_COS        = 11;
constexpr int OP_TAN        = 12;
constexpr int OP_SINH       = 13;
constexpr int OP_COSH       = 14;
constexpr int OP_TANH       = 15;
constexpr int OP_EXP        = 16;
constexpr int OP_LOG        = 17;
constexpr int OP_INV        = 18;
constexpr int OP_ASIN       = 19;
constexpr int OP_ACOS       = 20;
constexpr int OP_ATAN       = 21;
constexpr int OP_LOOSE_LOG  = 22;
constexpr int OP_LOOSE_INV  = 23;
constexpr int OP_ABS        = 24;
constexpr int OP_NEG        = 25;
constexpr int OP_SQRT       = 26;
constexpr int OP_LOOSE_SQRT = 27;

SR_HD int op_arity(int op)
{
    switch (op)
    {
        case OP_SIN:
        case OP_COS:
        case OP_TAN:
        case OP_SINH:
        case OP_COSH:
        case OP_TANH:
        case OP_EXP:
        case OP_LOG:
        case OP_INV:
        case OP_ASIN:
        case OP_ACOS:
        case OP_ATAN:
        case OP_LOOSE_LOG:
        case OP_LOOSE_INV:
        case OP_ABS:
        case OP_NEG:
        case OP_SQRT:
        case OP_LOOSE_SQRT:
            return 1;
        case OP_ADD:
        case OP_SUB:
        case OP_MUL:
        case OP_DIV:
        case OP_POW:
        case OP_MIN:
        case OP_MAX:
        case OP_LOOSE_DIV:
        case OP_LOOSE_POW:
            return 2;
        default:
            return 2; // treat unknown operators as binary for safety
    }
}

SR_HD bool is_unary_op(int op)
{
    return op_arity(op) == 1;
}

SR_HD bool is_binary_op(int op)
{
    return op_arity(op) == 2;
}

#undef SR_HD
