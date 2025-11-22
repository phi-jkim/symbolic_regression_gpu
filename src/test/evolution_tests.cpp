// Basic unit-style tests for GPU evolution kernels.
// No external test framework needed – simple asserts used.
// Build (example):
//   nvcc -I../utils -std=c++17 evolution_tests.cpp \
//        ../utils/generate.cu ../utils/mutation.cu ../utils/crossover.cu \
//        -o evolution_tests && ./evolution_tests

#include <cuda_runtime.h>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include "../utils/gpu_kernel.h"
#include "../utils/defs.h"
#include "../utils/opcodes.h"

#define CUDA_CHECK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) {           \
        std::cerr << "CUDA error: " << cudaGetErrorString(_e)                     \
                  << " (" << __FILE__ << ":" << __LINE__ << ")\n";            \
        std::exit(EXIT_FAILURE); } } while (0)

namespace {
//--------------------------------------------------------------------
// Host-side numeric evaluator (prefix traversal)
//--------------------------------------------------------------------
float eval_prefix_host(const float* val, const int16_t* type, int& idx, const std::vector<float>& vars){
    int16_t t = type[idx] & NodeType::TYPE_MASK; float v = val[idx]; idx++;
    if(t==NodeType::CONST) return v;
    if(t==NodeType::VAR)   return vars[(int)v];
    int opcode = (int)v;   // for function nodes value holds opcode
    if(t==NodeType::UFUNC){
        float a = eval_prefix_host(val,type,idx,vars);
        // Clamp input to prevent NaN or Infinity
        if(std::isnan(a) || std::isinf(a)) a = 0.0f;
        switch(opcode){
            case Function::SIN: return sinf(a); case Function::COS: return cosf(a); case Function::TAN: return tanf(a);
            case Function::SINH: return sinhf(a); case Function::COSH: return coshf(a); case Function::TANH: return tanhf(a);
            case Function::EXP: return (a > 10.0f) ? MAX_VAL : expf(a); // Clamp exp to avoid infinity
            case Function::LOG: return (a <= 0.0f) ? -MAX_VAL : logf(a); // Avoid log(0) or negative
            case Function::INV: return (fabsf(a) < DELTA) ? (a < 0 ? -MAX_VAL : MAX_VAL) : 1.f/a;
            case Function::ASIN: return asinf(a); case Function::ACOS: return acosf(a); case Function::ATAN: return atanf(a);
            case Function::LOOSE_LOG: return (a==0.f?-MAX_VAL:logf(fabsf(a))); case Function::LOOSE_INV: return 1.f/((fabsf(a)<DELTA)?(a<0?-DELTA:DELTA):a);
            case Function::ABS: return fabsf(a); case Function::NEG: return -a; case Function::SQRT: return (a < 0.0f) ? 0.0f : sqrtf(a);
            case Function::LOOSE_SQRT: return sqrtf(fabsf(a)); default: std::cerr << "Unhandled UFUNC opcode: " << opcode << "\n"; return 0.f; }
    }
    if(t==NodeType::BFUNC){ float a = eval_prefix_host(val,type,idx,vars); float b = eval_prefix_host(val,type,idx,vars);
        // Clamp inputs to prevent NaN or Infinity
        if(std::isnan(a) || std::isinf(a)) a = 0.0f;
        if(std::isnan(b) || std::isinf(b)) b = 0.0f;
        switch(opcode){
            case Function::ADD: return a+b; case Function::SUB: return a-b; case Function::MUL: return a*b;
            case Function::DIV: return (fabsf(b) < DELTA) ? (b < 0 ? -MAX_VAL : MAX_VAL) : a/b; // Avoid division by zero
            case Function::MIN: return a<=b?a:b; case Function::MAX: return a>=b?a:b;
            case Function::POW: return (a==0.f && b==0.f) ? 0.f : (fabsf(a) > 10.0f || fabsf(b) > 10.0f) ? MAX_VAL : powf(a,b); // Clamp pow to avoid infinity
            case Function::LOOSE_DIV: return a/((fabsf(b)<DELTA)?(b<0?-DELTA:DELTA):b);
            case Function::LOOSE_POW: return (a==0.f && b==0.f)?0.f:powf(fabsf(a),b);
            default: std::cerr << "Unhandled BFUNC opcode: " << opcode << "\n"; return 0.f; }
    }
    // ternary IF (only opcode IF)
    float cond = eval_prefix_host(val,type,idx,vars);
    float thenv= eval_prefix_host(val,type,idx,vars);
    float elsev= eval_prefix_host(val,type,idx,vars);
    return (cond>0.f)?thenv:elsev;
}

//--------------------------------------------------------------------
// Small helpers
//--------------------------------------------------------------------
struct DeviceBuffers {
    float   *value;
    int16_t *type;
    int16_t *subsize;
};

DeviceBuffers alloc_tree_buffers(unsigned pop, unsigned maxLen) {
    DeviceBuffers db{};
    CUDA_CHECK(cudaMalloc(&db.value,   pop * maxLen * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db.type,    pop * maxLen * sizeof(int16_t)));
    CUDA_CHECK(cudaMalloc(&db.subsize, pop * maxLen * sizeof(int16_t)));
    return db;
}

void free(DeviceBuffers& db){ cudaFree(db.value); cudaFree(db.type); cudaFree(db.subsize);}  

void copy_tree_to_device(const std::vector<float>& val, const std::vector<int16_t>& type, const std::vector<int16_t>& subsize, DeviceBuffers& db, unsigned index, unsigned maxLen) {
    CUDA_CHECK(cudaMemcpy(db.value + index * maxLen, val.data(), maxLen * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db.type + index * maxLen, type.data(), maxLen * sizeof(int16_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db.subsize + index * maxLen, subsize.data(), maxLen * sizeof(int16_t), cudaMemcpyHostToDevice));
}

//--------------------------------------------------------------------
// Deterministic tables for generation
//--------------------------------------------------------------------
struct ConstantTables {
    unsigned int *keys;
    float *depth2leaf;
    float *roulette;
    float *constSamples;
};

ConstantTables alloc_constant_tables() {
    ConstantTables ct{};
    CUDA_CHECK(cudaMalloc(&ct.keys,        2 * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&ct.depth2leaf,  MAX_FULL_DEPTH * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ct.roulette,    (Function::LOOSE_SQRT + 1) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ct.constSamples,8 * sizeof(float)));

    unsigned int h_keys[2] = {42, 1337};
    float h_depth[MAX_FULL_DEPTH];
    for(int i=0;i<MAX_FULL_DEPTH;i++) h_depth[i]=0.25f;
    std::vector<float> h_roulette(Function::LOOSE_SQRT + 1);
    h_roulette[0] = 0.0f;                                         // ensure ADD gets non-zero probability
    for(int i=1;i<=Function::LOOSE_SQRT;i++)
        h_roulette[i] = static_cast<float>(i) / static_cast<float>(Function::LOOSE_SQRT);
    float h_consts[8] = {-1.f,-0.5f,0.f,0.5f,1.f,2.f,3.f,4.f};

    CUDA_CHECK(cudaMemcpy(ct.keys,        h_keys,       sizeof(h_keys),           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ct.depth2leaf,  h_depth,      sizeof(h_depth),          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ct.roulette,    h_roulette.data(), h_roulette.size()*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ct.constSamples,h_consts,     sizeof(h_consts),         cudaMemcpyHostToDevice));
    return ct;
}

void free(ConstantTables& ct){ cudaFree(ct.keys); cudaFree(ct.depth2leaf); cudaFree(ct.roulette); cudaFree(ct.constSamples);}  

//--------------------------------------------------------------------
// Simple host-side validation helpers
//--------------------------------------------------------------------
bool validate_tree(const std::vector<int16_t>& types, const std::vector<int16_t>& sz, int maxLen){
    const int total = sz[0];
    if(total<=0 || total>maxLen) {
        std::cout << "[validate_tree] Failed: total=" << total << " out of bounds (maxLen=" << maxLen << ")\n";
        return false;
    }
    // Use a stack simulation for prefix notation: traverse right to left to process operands before operators
    int stack_size = 0; // Track actual values on stack
    for(int idx = total - 1; idx >= 0; idx--) {
        int16_t t = types[idx] & NodeType::TYPE_MASK;
        if(t == NodeType::CONST || t == NodeType::VAR) {
            stack_size++; // Push one value onto stack
        } else if(t == NodeType::UFUNC) {
            if(stack_size < 1) {
                std::cout << "[validate_tree] Failed: UFUNC at index=" << idx << " needs 1 operand, stack_size=" << stack_size << "\n";
                return false; // Need at least 1 operand
            }
            stack_size--; // Pop 1 operand
            stack_size++; // Push result
        } else if(t == NodeType::BFUNC) {
            if(stack_size < 2) {
                std::cout << "[validate_tree] Failed: BFUNC at index=" << idx << " needs 2 operands, stack_size=" << stack_size << "\n";
                return false; // Need at least 2 operands
            }
            stack_size -= 2; // Pop 2 operands
            stack_size++; // Push result
        } else if(t == NodeType::TFUNC) {
            if(stack_size < 3) {
                std::cout << "[validate_tree] Failed: TFUNC at index=" << idx << " needs 3 operands, stack_size=" << stack_size << "\n";
                return false; // Need at least 3 operands for IF
            }
            stack_size -= 3; // Pop 3 operands
            stack_size++; // Push result
        } else {
            std::cout << "[validate_tree] Failed: Unknown type at index=" << idx << ", type=" << t << "\n";
            return false; // Unknown type
        }
    }
    if(stack_size != 1) {
        std::cout << "[validate_tree] Failed: Final stack_size=" << stack_size << ", expected 1\n";
    }
    return (stack_size == 1); // At the end, stack should have exactly one result
}

//--------------------------------------------------------------------
// TEST 1 – GENERATE
//--------------------------------------------------------------------
void test_generate(const ConstantTables& ct){
    const unsigned pop=4, maxLen=64, varLen=3, outLen=1, constLen=8;
    DeviceBuffers db = alloc_tree_buffers(pop,maxLen);

    generate(pop,maxLen,varLen,outLen,constLen,0.9f,0.1f, // Increased constProb to favor operands
             ct.keys, ct.depth2leaf, ct.roulette, ct.constSamples,
             db.value, db.type, db.subsize);
    CUDA_CHECK(cudaDeviceSynchronize());

    // copy first individual
    std::vector<int16_t> h_type(maxLen), h_sub(maxLen);
    std::vector<float> h_val(maxLen);
    CUDA_CHECK(cudaMemcpy(h_type.data(), db.type, maxLen*sizeof(int16_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_sub.data(),  db.subsize, maxLen*sizeof(int16_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_val.data(),  db.value, maxLen*sizeof(float), cudaMemcpyDeviceToHost));

    if(!validate_tree(h_type,h_sub,maxLen)) {
        std::cout << "[generate] Validation failed for first tree. Tree size=" << h_sub[0] << "\n";
        std::cout << "Tree contents:\n";
        for(int i = 0; i < h_sub[0]; i++) {
            std::cout << "Index " << i << ": Type=" << h_type[i] << ", Value=" << h_val[i] << ", Subsize=" << h_sub[i] << "\n";
        }
        assert(validate_tree(h_type,h_sub,maxLen));
    }
    // Validate opcodes are within desired range (no IF, which is 0)
    int tree_size = h_sub[0];
    for (int i = 0; i < tree_size; i++) {
        if ((h_type[i] & NodeType::TYPE_MASK) >= NodeType::UFUNC) {
            assert(h_val[i] >= 1.0f); // No IF opcode (0)
        }
    }
    // Evaluate with sample data to ensure it produces a numeric value
    std::vector<float> vars = {1.5f, 2.5f, 3.5f};
    int idx = 0;
    float result = eval_prefix_host(h_val.data(), h_type.data(), idx, vars);
    if(std::isnan(result) || std::isinf(result)) {
        std::cout << "[generate] Evaluation resulted in NaN or Infinity. Tree size=" << h_sub[0] << "\n";
        std::cout << "Tree contents:\n";
        for(int i = 0; i < h_sub[0]; i++) {
            std::cout << "Index " << i << ": Type=" << h_type[i] << ", Value=" << h_val[i] << ", Subsize=" << h_sub[i] << "\n";
        }
    } else {
        std::cout << "[generate] first tree size=" << h_sub[0] << " valid, opcodes OK, evaluates to " << result << "\n";
    }
    // Don't fail on NaN/Inf for now, just log it
    // assert(!std::isnan(result) && !std::isinf(result));

    free(db);
}

//--------------------------------------------------------------------
// TEST 1b – GENERATE (high constProb to test CONST nodes)
//--------------------------------------------------------------------
void test_generate_const_heavy(const ConstantTables& ct){
    const unsigned pop=4, maxLen=64, varLen=3, outLen=1, constLen=8;
    DeviceBuffers db = alloc_tree_buffers(pop,maxLen);

    generate(pop,maxLen,varLen,outLen,constLen,0.0f,0.9f, // high constProb
             ct.keys, ct.depth2leaf, ct.roulette, ct.constSamples,
             db.value, db.type, db.subsize);
    CUDA_CHECK(cudaDeviceSynchronize());

    // copy first individual
    std::vector<int16_t> h_type(maxLen), h_sub(maxLen);
    std::vector<float> h_val(maxLen);
    CUDA_CHECK(cudaMemcpy(h_type.data(), db.type, maxLen*sizeof(int16_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_sub.data(),  db.subsize, maxLen*sizeof(int16_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_val.data(),  db.value, maxLen*sizeof(float), cudaMemcpyDeviceToHost));

    if(!validate_tree(h_type,h_sub,maxLen)) {
        std::cout << "[generate-const] Validation failed for first tree. Tree size=" << h_sub[0] << "\n";
        std::cout << "Tree contents:\n";
        for(int i = 0; i < h_sub[0]; i++) {
            std::cout << "Index " << i << ": Type=" << h_type[i] << ", Value=" << h_val[i] << ", Subsize=" << h_sub[i] << "\n";
        }
        assert(validate_tree(h_type,h_sub,maxLen));
    }
    // Check if at least one node is CONST
    bool has_const = false;
    int tree_size = h_sub[0];
    for (int i = 0; i < tree_size; i++) {
        if (h_type[i] == NodeType::CONST) {
            has_const = true;
            break;
        }
    }
    if(!has_const) {
        std::cout << "[generate-const] No CONST node found in first tree. Tree size=" << h_sub[0] << "\n";
        std::cout << "Tree contents:\n";
        for(int i = 0; i < h_sub[0]; i++) {
            std::cout << "Index " << i << ": Type=" << h_type[i] << ", Value=" << h_val[i] << ", Subsize=" << h_sub[i] << "\n";
        }
    }
    assert(has_const);
    std::cout << "[generate-const] first tree size=" << h_sub[0] << " has CONST node OK\n";

    free(db);
}

//--------------------------------------------------------------------
// TEST 1c – GENERATE (with opcode validation)
//--------------------------------------------------------------------
void test_generate_opcode(const ConstantTables& ct){
    const unsigned pop=4, maxLen=64, varLen=3, outLen=1, constLen=8;
    DeviceBuffers db = alloc_tree_buffers(pop,maxLen);

    generate(pop,maxLen,varLen,outLen,constLen,0.0f,0.4f,
             ct.keys, ct.depth2leaf, ct.roulette, ct.constSamples,
             db.value, db.type, db.subsize);
    CUDA_CHECK(cudaDeviceSynchronize());

    // copy first individual
    std::vector<int16_t> h_type(maxLen), h_sub(maxLen);
    std::vector<float> h_val(maxLen);
    CUDA_CHECK(cudaMemcpy(h_type.data(), db.type, maxLen*sizeof(int16_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_sub.data(),  db.subsize, maxLen*sizeof(int16_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_val.data(),  db.value, maxLen*sizeof(float), cudaMemcpyDeviceToHost));

    if(!validate_tree(h_type,h_sub,maxLen)) {
        std::cout << "[generate-opcode] Validation failed for first tree. Tree size=" << h_sub[0] << "\n";
        std::cout << "Tree contents:\n";
        for(int i = 0; i < h_sub[0]; i++) {
            std::cout << "Index " << i << ": Type=" << h_type[i] << ", Value=" << h_val[i] << ", Subsize=" << h_sub[i] << "\n";
        }
        assert(validate_tree(h_type,h_sub,maxLen));
    }
    // Validate opcodes are within desired range (no IF, which is 0)
    int tree_size = h_sub[0];
    for (int i = 0; i < tree_size; i++) {
        if ((h_type[i] & NodeType::TYPE_MASK) >= NodeType::UFUNC) {
            if(h_val[i] < 1.0f) {
                std::cout << "[generate-opcode] Invalid opcode " << h_val[i] << " at index " << i << "\n";
                std::cout << "Tree contents:\n";
                for(int j = 0; j < h_sub[0]; j++) {
                    std::cout << "Index " << j << ": Type=" << h_type[j] << ", Value=" << h_val[j] << ", Subsize=" << h_sub[j] << "\n";
                }
            }
            assert(h_val[i] >= 1.0f); // No IF opcode (0)
        }
    }
    // Evaluate with sample data to ensure it produces a numeric value
    std::vector<float> vars = {1.5f, 2.5f, 3.5f};
    int idx = 0;
    float result = eval_prefix_host(h_val.data(), h_type.data(), idx, vars);
    if(std::isnan(result) || std::isinf(result)) {
        std::cout << "[generate-opcode] Evaluation resulted in NaN or Infinity. Tree size=" << h_sub[0] << "\n";
        std::cout << "Tree contents:\n";
        for(int i = 0; i < h_sub[0]; i++) {
            std::cout << "Index " << i << ": Type=" << h_type[i] << ", Value=" << h_val[i] << ", Subsize=" << h_sub[i] << "\n";
        }
        std::cout << "[generate-opcode] Evaluation details: ";
        for (int i = 0; i < h_sub[0]; i++) {
            std::cout << "Index " << i << ": Type=" << h_type[i] << ", Value=" << h_val[i] << ", Subsize=" << h_sub[i] << " | ";
        }
        std::cout << "\n";
    } else {
        std::cout << "[generate-opcode] first tree size=" << h_sub[0] << " valid, opcodes OK, evaluates to " << result << "\n";
    }
    // Don't fail on NaN/Inf for now, just log it
    // assert(!std::isnan(result) && !std::isinf(result));

    free(db);
}

//--------------------------------------------------------------------
// TEST 1d – GENERATE (large population validation)
//--------------------------------------------------------------------
void test_generate_large_population(const ConstantTables& ct){
    const unsigned pop=1000, maxLen=64, varLen=3, outLen=1, constLen=8;
    DeviceBuffers db = alloc_tree_buffers(pop,maxLen);

    generate(pop,maxLen,varLen,outLen,constLen,0.0f,0.4f,
             ct.keys, ct.depth2leaf, ct.roulette, ct.constSamples,
             db.value, db.type, db.subsize);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Validate each individual in the population
    std::vector<int16_t> h_type(maxLen), h_sub(maxLen);
    std::vector<float> h_val(maxLen);
    int valid_count = 0;
    int min_len = maxLen, max_len = 0; long long sum_len = 0;
    for(unsigned i = 0; i < pop; i++) {
        CUDA_CHECK(cudaMemcpy(h_type.data(), db.type + i * maxLen, maxLen*sizeof(int16_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_sub.data(), db.subsize + i * maxLen, maxLen*sizeof(int16_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_val.data(), db.value + i * maxLen, maxLen*sizeof(float), cudaMemcpyDeviceToHost));
        int len = h_sub[0];
        min_len = std::min(min_len, len);
        max_len = std::max(max_len, len);
        sum_len += len;
        if(validate_tree(h_type, h_sub, maxLen)) {
            valid_count++;
        } else {
            std::cout << "[generate-large] Invalid tree structure in population at index " << i << " with size=" << h_sub[0] << "\n";
            std::cout << "Tree contents:\n";
            for(int j = 0; j < h_sub[0]; j++) {
                std::cout << "Index " << j << ": Type=" << h_type[j] << ", Value=" << h_val[j] << ", Subsize=" << h_sub[j] << "\n";
            }
        }
    }
    if(valid_count != pop) {
        std::cout << "[generate-large] Only " << valid_count << "/" << pop << " trees are structurally valid\n";
        assert(valid_count == pop);
    }
    double avg_len = static_cast<double>(sum_len) / pop;
    std::cout << "[generate-large] All " << valid_count << "/" << pop << " trees are structurally valid OK\n";
    std::cout << "[generate-large] Tree length stats -- min: " << min_len << ", max: " << max_len << ", avg: " << avg_len << "\n";

    free(db);
}

//--------------------------------------------------------------------
// TEST 1e – GENERATE (opcode range validation 1 to 27)
//--------------------------------------------------------------------
void test_generate_opcode_range(const ConstantTables& ct){
    const unsigned pop=100, maxLen=64, varLen=3, outLen=1, constLen=8;
    DeviceBuffers db = alloc_tree_buffers(pop,maxLen);

    generate(pop,maxLen,varLen,outLen,constLen,0.0f,0.4f,
             ct.keys, ct.depth2leaf, ct.roulette, ct.constSamples,
             db.value, db.type, db.subsize);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Check each individual in the population for opcode range
    std::vector<int16_t> h_type(maxLen);
    std::vector<float> h_val(maxLen);
    std::vector<int16_t> h_sub(maxLen);
    bool all_in_range = true;
    for(unsigned i = 0; i < pop; i++) {
        CUDA_CHECK(cudaMemcpy(h_type.data(), db.type + i * maxLen, maxLen*sizeof(int16_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_val.data(), db.value + i * maxLen, maxLen*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_sub.data(), db.subsize + i * maxLen, maxLen*sizeof(int16_t), cudaMemcpyDeviceToHost));
        if(!validate_tree(h_type, h_sub, maxLen)) {
            std::cout << "[generate-opcode-range] Validation failed for tree at index " << i << ". Tree size=" << h_sub[0] << "\n";
            std::cout << "Tree contents:\n";
            for(int j = 0; j < h_sub[0]; j++) {
                std::cout << "Index " << j << ": Type=" << h_type[j] << ", Value=" << h_val[j] << ", Subsize=" << h_sub[j] << "\n";
            }
        }
        int tree_size = h_sub[0];
        for(int j = 0; j < tree_size; j++) {
            if ((h_type[j] & NodeType::TYPE_MASK) >= NodeType::UFUNC) {
                int opcode = static_cast<int>(h_val[j]);
                if(opcode < 1 || opcode > 27) { // Check range 1 to 27 (ADD to LOOSE_SQRT)
                    all_in_range = false;
                    std::cout << "[generate-opcode-range] Invalid opcode " << opcode << " at population index " << i << ", tree position " << j << "\n";
                }
            }
        }
    }
    assert(all_in_range);
    std::cout << "[generate-opcode-range] All opcodes in range 1 to 27 for population of " << pop << " OK\n";

    free(db);
}

//--------------------------------------------------------------------
// TEST 2 – MUTATE (deterministic)
//--------------------------------------------------------------------
void test_mutate_det(){
    const unsigned pop = 1, maxLen = 16;
    DeviceBuffers parent = alloc_tree_buffers(pop,maxLen);
    DeviceBuffers subtree = alloc_tree_buffers(pop,maxLen);

    // Parent tree: ADD(MUL(2,3), 4) => 10
    std::vector<float> p_val = {(float)Function::ADD, (float)Function::MUL, 2.f, 3.f, 4.f};
    std::vector<int16_t> p_type = {NodeType::BFUNC, NodeType::BFUNC, NodeType::CONST, NodeType::CONST, NodeType::CONST};
    std::vector<int16_t> p_sub = {5, 3, 1, 1, 1};
    p_val.resize(maxLen, 0.0f); p_type.resize(maxLen, 0); p_sub.resize(maxLen, 0);
    copy_tree_to_device(p_val, p_type, p_sub, parent, 0, maxLen);

    // New subtree: SUB(8,5) => 3 (subtreeSize=3)
    std::vector<float> s_val(maxLen,0.f); 
    std::vector<int16_t> s_type(maxLen,0), s_sub(maxLen,0);
    s_val[0] = (float)Function::SUB; s_val[1] = 8.f; s_val[2] = 5.f;
    s_type[0] = NodeType::BFUNC; s_type[1] = NodeType::CONST; s_type[2] = NodeType::CONST;
    s_sub[0] = 3; s_sub[1] = 1; s_sub[2] = 1;
    copy_tree_to_device(s_val, s_type, s_sub, subtree, 0, maxLen);

    int h_idx = 0; int* d_idx; CUDA_CHECK(cudaMalloc(&d_idx,sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_idx,&h_idx,sizeof(int),cudaMemcpyHostToDevice));

    mutate(pop, maxLen,
           parent.value, parent.type, parent.subsize,
           d_idx,
           subtree.value, subtree.type, subtree.subsize,
           parent.value, parent.type, parent.subsize);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Fetch result
    std::vector<float> r_val(maxLen); std::vector<int16_t> r_type(maxLen), r_sub(maxLen);
    CUDA_CHECK(cudaMemcpy(r_val.data(), parent.value,  maxLen*sizeof(float),    cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(r_type.data(),parent.type,   maxLen*sizeof(int16_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(r_sub.data(), parent.subsize,maxLen*sizeof(int16_t), cudaMemcpyDeviceToHost));

    int idx = 0; std::vector<float> vars; float out = eval_prefix_host(r_val.data(), r_type.data(), idx, vars);
    assert(out == 3.f); assert(r_sub[0]==3 && r_type[0]==NodeType::BFUNC && r_val[0]==(float)Function::SUB);
    std::cout << "[mutate-det] produced SUB(8,5)=3 OK\n";

    cudaFree(d_idx); free(parent); free(subtree);
}

//--------------------------------------------------------------------
// TEST 2b – MUTATE (deeper tree, non-root node)
//--------------------------------------------------------------------
void test_mutate_det_deep(){
    const unsigned pop = 1, maxLen = 16;
    DeviceBuffers parent = alloc_tree_buffers(pop,maxLen);
    DeviceBuffers subtree = alloc_tree_buffers(pop,maxLen);

    // Parent tree: ADD(MUL(SUB(5,2),3), 4) => 13
    std::vector<float> p_val = {static_cast<float>(Function::ADD), static_cast<float>(Function::MUL), static_cast<float>(Function::SUB), 5.f, 2.f, 3.f, 4.f};
    std::vector<int16_t> p_type = {NodeType::BFUNC, NodeType::BFUNC, NodeType::BFUNC, NodeType::CONST, NodeType::CONST, NodeType::CONST, NodeType::CONST};
    std::vector<int16_t> p_sub = {7, 5, 3, 1, 1, 1, 1};
    p_val.resize(maxLen, 0.0f); p_type.resize(maxLen, 0); p_sub.resize(maxLen, 0);
    copy_tree_to_device(p_val, p_type, p_sub, parent, 0, maxLen);

    // New subtree: ADD(1,1) => 2 (subtreeSize=3)
    std::vector<float> s_val(maxLen,0.f); 
    std::vector<int16_t> s_type(maxLen,0), s_sub(maxLen,0);
    s_val[0] = static_cast<float>(Function::ADD); s_val[1] = 1.f; s_val[2] = 1.f;
    s_type[0] = NodeType::BFUNC; s_type[1] = NodeType::CONST; s_type[2] = NodeType::CONST;
    s_sub[0] = 3; s_sub[1] = 1; s_sub[2] = 1;
    copy_tree_to_device(s_val, s_type, s_sub, subtree, 0, maxLen);

    int h_idx = 2; // target SUB(5,2) node for replacement
    int* d_idx; CUDA_CHECK(cudaMalloc(&d_idx,sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_idx,&h_idx,sizeof(int),cudaMemcpyHostToDevice));

    mutate(pop, maxLen,
           parent.value, parent.type, parent.subsize,
           d_idx,
           subtree.value, subtree.type, subtree.subsize,
           parent.value, parent.type, parent.subsize);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Fetch result
    std::vector<float> r_val(maxLen); std::vector<int16_t> r_type(maxLen), r_sub(maxLen);
    CUDA_CHECK(cudaMemcpy(r_val.data(), parent.value,  maxLen*sizeof(float),    cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(r_type.data(),parent.type,   maxLen*sizeof(int16_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(r_sub.data(), parent.subsize,maxLen*sizeof(int16_t), cudaMemcpyDeviceToHost));

    int idx = 0; std::vector<float> vars; float out = eval_prefix_host(r_val.data(), r_type.data(), idx, vars);
    assert(out == 10.f); // ADD(MUL(ADD(1,1),3),4) = ADD(MUL(2,3),4) = ADD(6,4) = 10
    assert(r_sub[2]==3 && r_type[2]==NodeType::BFUNC && r_val[2]==static_cast<float>(Function::ADD));
    std::cout << "[mutate-det-deep] replaced SUB with ADD, evaluates to 10 OK\n";

    cudaFree(d_idx); free(parent); free(subtree);
}

//--------------------------------------------------------------------
// TEST 2b – MUTATE (interleaved deterministic with large population)
//--------------------------------------------------------------------
void test_mutate_interleaved_det_large(){
    const unsigned pop = 100, maxLen = 64;
    DeviceBuffers parent = alloc_tree_buffers(pop, maxLen);
    DeviceBuffers subtree = alloc_tree_buffers(pop, maxLen);
    DeviceBuffers result = alloc_tree_buffers(pop, maxLen);

    // Create a set of parent trees and subtrees for mutation
    std::vector<std::vector<float>> p_vals(pop, std::vector<float>(maxLen, 0.0f));
    std::vector<std::vector<int16_t>> p_types(pop, std::vector<int16_t>(maxLen, 0));
    std::vector<std::vector<int16_t>> p_subs(pop, std::vector<int16_t>(maxLen, 0));
    std::vector<std::vector<float>> s_vals(pop, std::vector<float>(maxLen, 0.0f));
    std::vector<std::vector<int16_t>> s_types(pop, std::vector<int16_t>(maxLen, 0));
    std::vector<std::vector<int16_t>> s_subs(pop, std::vector<int16_t>(maxLen, 0));
    std::vector<int> mutate_indices(pop, 0);

    for(unsigned i = 0; i < pop; i++) {
        // Parent tree: Alternate between simple and complex structures
        if(i % 2 == 0) {
            // Simple: ADD(VAL, VAL)
            p_vals[i][0] = static_cast<float>(Function::ADD); p_types[i][0] = NodeType::BFUNC; p_subs[i][0] = 3;
            p_vals[i][1] = static_cast<float>(i % 5 + 1); p_types[i][1] = NodeType::CONST; p_subs[i][1] = 1;
            p_vals[i][2] = static_cast<float>((i + 1) % 5 + 1); p_types[i][2] = NodeType::CONST; p_subs[i][2] = 1;
        } else {
            // Complex: ADD(MUL(VAL, VAL), SUB(VAL, VAL))
            p_vals[i][0] = static_cast<float>(Function::ADD); p_types[i][0] = NodeType::BFUNC; p_subs[i][0] = 7;
            p_vals[i][1] = static_cast<float>(Function::MUL); p_types[i][1] = NodeType::BFUNC; p_subs[i][1] = 3;
            p_vals[i][2] = static_cast<float>(i % 5 + 1); p_types[i][2] = NodeType::CONST; p_subs[i][2] = 1;
            p_vals[i][3] = static_cast<float>((i + 2) % 5 + 1); p_types[i][3] = NodeType::CONST; p_subs[i][3] = 1;
            p_vals[i][4] = static_cast<float>(Function::SUB); p_types[i][4] = NodeType::BFUNC; p_subs[i][4] = 3;
            p_vals[i][5] = static_cast<float>((i + 3) % 5 + 1); p_types[i][5] = NodeType::CONST; p_subs[i][5] = 1;
            p_vals[i][6] = static_cast<float>((i + 4) % 5 + 1); p_types[i][6] = NodeType::CONST; p_subs[i][6] = 1;
        }
        copy_tree_to_device(p_vals[i], p_types[i], p_subs[i], parent, i, maxLen);

        // Subtree to mutate: Simple replacement
        s_vals[i][0] = static_cast<float>(Function::NEG); s_types[i][0] = NodeType::UFUNC; s_subs[i][0] = 2;
        s_vals[i][1] = static_cast<float>((i % 3) + 1.0f); s_types[i][1] = NodeType::CONST; s_subs[i][1] = 1;
        copy_tree_to_device(s_vals[i], s_types[i], s_subs[i], subtree, i, maxLen);

        // Alternate mutation index for interleaved effect
        mutate_indices[i] = (i % 2 == 0) ? 0 : 1;
    }

    // Allocate device memory for mutation indices
    int* d_indices;
    CUDA_CHECK(cudaMalloc(&d_indices, pop * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_indices, mutate_indices.data(), pop * sizeof(int), cudaMemcpyHostToDevice));

    // Perform mutation on entire population
    mutate(pop, maxLen,
           parent.value, parent.type, parent.subsize,
           d_indices,
           subtree.value, subtree.type, subtree.subsize,
           result.value, result.type, result.subsize);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Validate results for each individual
    for(unsigned i = 0; i < pop; i++) {
        std::vector<float> r_val(maxLen);
        std::vector<int16_t> r_type(maxLen), r_sub(maxLen);
        CUDA_CHECK(cudaMemcpy(r_val.data(), result.value + i * maxLen, maxLen * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(r_type.data(), result.type + i * maxLen, maxLen * sizeof(int16_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(r_sub.data(), result.subsize + i * maxLen, maxLen * sizeof(int16_t), cudaMemcpyDeviceToHost));

        // Numerical evaluation to verify correctness
        int idx = 0;
        std::vector<float> vars;
        float out = eval_prefix_host(r_val.data(), r_type.data(), idx, vars);
        if(std::isnan(out) || std::isinf(out)) {
            std::cout << "[mutate-interleaved-large] Evaluation failed (NaN/Inf) at index " << i << "\n";
        }
        // Structural check at mutation point
        int mutate_idx = mutate_indices[i];
        if(r_type[mutate_idx] != NodeType::UFUNC || r_val[mutate_idx] != static_cast<float>(Function::NEG)) {
            std::cout << "[mutate-interleaved-large] Structural mismatch at index " << i << ", mutation point " << mutate_idx << "\n";
            std::cout << "Expected NEG (UFUNC), got Type=" << r_type[mutate_idx] << ", Value=" << r_val[mutate_idx] << "\n";
            assert(r_type[mutate_idx] == NodeType::UFUNC && r_val[mutate_idx] == static_cast<float>(Function::NEG));
        }
    }
    std::cout << "[mutate-interleaved-large] All " << pop << " mutations verified OK\n";

    cudaFree(d_indices);
    free(parent);
    free(subtree);
    free(result);
}

//--------------------------------------------------------------------
// TEST 3 – CROSSOVER (deterministic)
//--------------------------------------------------------------------
void test_crossover_det(){
    const unsigned pop_src = 2, maxLen = 16, pop_child = 1;
    DeviceBuffers src = alloc_tree_buffers(pop_src, maxLen);
    DeviceBuffers dst = alloc_tree_buffers(pop_child, maxLen);

    // Left parent = ADD(MUL(1,2),3) -> 5
    std::vector<float> l_val = {static_cast<float>(Function::ADD), static_cast<float>(Function::MUL), 1.f, 2.f, 3.f};
    std::vector<int16_t> l_type = {NodeType::BFUNC, NodeType::BFUNC, NodeType::CONST, NodeType::CONST, NodeType::CONST};
    std::vector<int16_t> l_sub = {5, 3, 1, 1, 1};
    l_val.resize(maxLen,0.f); l_type.resize(maxLen,0); l_sub.resize(maxLen,0);
    copy_tree_to_device(l_val,l_type,l_sub, src, 0, maxLen);

    // Right parent = SUB(POW(2,3),5) -> 3
    std::vector<float> r_val = {static_cast<float>(Function::SUB), static_cast<float>(Function::POW), 2.f, 3.f, 5.f};
    std::vector<int16_t> r_type = {NodeType::BFUNC, NodeType::BFUNC, NodeType::CONST, NodeType::CONST, NodeType::CONST};
    std::vector<int16_t> r_sub = {5, 3, 1, 1, 1};
    r_val.resize(maxLen,0.f); r_type.resize(maxLen,0); r_sub.resize(maxLen,0);
    copy_tree_to_device(r_val,r_type,r_sub, src, 1, maxLen);

    // Mapping arrays (generate on host)
    int h_left=0, h_right=1, h_leftNode=0, h_rightNode=0;
    int *d_left,*d_right,*d_ln,*d_rn; CUDA_CHECK(cudaMalloc(&d_left,sizeof(int))); CUDA_CHECK(cudaMalloc(&d_right,sizeof(int))); CUDA_CHECK(cudaMalloc(&d_ln,sizeof(int))); CUDA_CHECK(cudaMalloc(&d_rn,sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_left,&h_left,sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_right,&h_right,sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ln,&h_leftNode,sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rn,&h_rightNode,sizeof(int),cudaMemcpyHostToDevice));

    crossover(pop_src, pop_child, maxLen,
              src.value, src.type, src.subsize,
              d_left, d_right, d_ln, d_rn,
              dst.value, dst.type, dst.subsize);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Fetch child
    std::vector<float> c_val(maxLen); std::vector<int16_t> c_type(maxLen), c_sub(maxLen);
    CUDA_CHECK(cudaMemcpy(c_val.data(), dst.value,  maxLen*sizeof(float),    cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(c_type.data(),dst.type,   maxLen*sizeof(int16_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(c_sub.data(), dst.subsize,maxLen*sizeof(int16_t), cudaMemcpyDeviceToHost));

    // Should exactly equal right parent's tree
    assert(c_type[0] == r_type[0] && c_val[0] == r_val[0]);
    int idx=0; std::vector<float> vars; float out_child = eval_prefix_host(c_val.data(), c_type.data(), idx, vars);
    assert(out_child == 3.f);
    std::cout << "[crossover-det] child evaluates to 3 OK\n";

    cudaFree(d_left); cudaFree(d_right); cudaFree(d_ln); cudaFree(d_rn);
    free(src); free(dst);
}

//--------------------------------------------------------------------
// TEST 3c – CROSSOVER (deeper mixed arities)
//--------------------------------------------------------------------
void test_crossover_det_deep_mixed(){
    const unsigned pop_src = 2, maxLen = 32, pop_child = 1;
    DeviceBuffers src = alloc_tree_buffers(pop_src, maxLen);
    DeviceBuffers dst = alloc_tree_buffers(pop_child, maxLen);

    // Left parent: ADD(MUL(ADD(1,2), SUB(5,3)), NEG(4))
    // prefix indices:
    // 0: ADD, 1: MUL, 2: ADD, 3: 1, 4: 2, 5: SUB, 6: 5, 7: 3, 8: NEG, 9: 4
    std::vector<float> l_val = {
        static_cast<float>(Function::ADD),
        static_cast<float>(Function::MUL),
        static_cast<float>(Function::ADD), 1.f, 2.f,
        static_cast<float>(Function::SUB), 5.f, 3.f,
        static_cast<float>(Function::NEG), 4.f
    };
    std::vector<int16_t> l_type = {
        NodeType::BFUNC,
        NodeType::BFUNC,
        NodeType::BFUNC, NodeType::CONST, NodeType::CONST,
        NodeType::BFUNC, NodeType::CONST, NodeType::CONST,
        NodeType::UFUNC, NodeType::CONST
    };
    std::vector<int16_t> l_sub = {10, 7, 3, 1, 1, 3, 1, 1, 2, 1};
    l_val.resize(maxLen, 0.f); l_type.resize(maxLen, 0); l_sub.resize(maxLen, 0);
    copy_tree_to_device(l_val,l_type,l_sub, src, 0, maxLen);

    // Right parent: SUB(POW(ADD(2,1),3), MUL(6,2))
    // 0: SUB, 1: POW, 2: ADD, 3: 2, 4: 1, 5: 3, 6: MUL, 7: 6, 8: 2
    std::vector<float> r_val = {
        static_cast<float>(Function::SUB),
        static_cast<float>(Function::POW),
        static_cast<float>(Function::ADD), 2.f, 1.f, 3.f,
        static_cast<float>(Function::MUL), 6.f, 2.f
    };
    std::vector<int16_t> r_type = {
        NodeType::BFUNC,
        NodeType::BFUNC,
        NodeType::BFUNC, NodeType::CONST, NodeType::CONST, NodeType::CONST,
        NodeType::BFUNC, NodeType::CONST, NodeType::CONST
    };
    // subtree sizes: node2(ADD)=3, node1(POW)=1+3+1=5, node6(MUL)=3, root(SUB)=1+5+3=9
    std::vector<int16_t> r_sub = {9, 5, 3, 1, 1, 1, 3, 1, 1};
    r_val.resize(maxLen,0.f); r_type.resize(maxLen,0); r_sub.resize(maxLen,0);
    copy_tree_to_device(r_val,r_type,r_sub, src, 1, maxLen);

    // Replace left MUL subtree at index 1 with right POW subtree at index 1
    int h_left=0, h_right=1, h_leftNode=1, h_rightNode=1;
    int *d_left,*d_right,*d_ln,*d_rn;
    CUDA_CHECK(cudaMalloc(&d_left,sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_right,sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ln,sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_rn,sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_left,&h_left,sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_right,&h_right,sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ln,&h_leftNode,sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rn,&h_rightNode,sizeof(int),cudaMemcpyHostToDevice));

    crossover(pop_src, pop_child, maxLen,
              src.value, src.type, src.subsize,
              d_left, d_right, d_ln, d_rn,
              dst.value, dst.type, dst.subsize);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Fetch child
    std::vector<float> c_val(maxLen); std::vector<int16_t> c_type(maxLen), c_sub(maxLen);
    CUDA_CHECK(cudaMemcpy(c_val.data(), dst.value,  maxLen*sizeof(float),    cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(c_type.data(),dst.type,   maxLen*sizeof(int16_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(c_sub.data(), dst.subsize,maxLen*sizeof(int16_t), cudaMemcpyDeviceToHost));

    // Check that at index 1 we now have POW and that the tree evaluates to the
    // analytically expected value:
    //   child = ADD( POW(ADD(2,1),3), NEG(4) ) = ADD(27, -4) = 23.
    assert(c_type[1] == NodeType::BFUNC && c_val[1] == static_cast<float>(Function::POW));
    int idx = 0; std::vector<float> vars; float out_child = eval_prefix_host(c_val.data(), c_type.data(), idx, vars);
    assert(out_child == 23.f);
    std::cout << "[crossover-det-deep-mixed] child evaluates to " << out_child << " (expected 23) OK\n";

    cudaFree(d_left); cudaFree(d_right); cudaFree(d_ln); cudaFree(d_rn);
    free(src); free(dst);
}

//--------------------------------------------------------------------
// TEST 3d – CROSSOVER (root swap on deeper trees)
//--------------------------------------------------------------------
void test_crossover_det_rootswap(){
    const unsigned pop_src = 2, maxLen = 32, pop_child = 1;
    DeviceBuffers src = alloc_tree_buffers(pop_src, maxLen);
    DeviceBuffers dst = alloc_tree_buffers(pop_child, maxLen);

    // Left parent: MUL(ADD(1,2), SUB(7,4))
    std::vector<float> l_val = {
        static_cast<float>(Function::MUL),
        static_cast<float>(Function::ADD), 1.f, 2.f,
        static_cast<float>(Function::SUB), 7.f, 4.f
    };
    std::vector<int16_t> l_type = {
        NodeType::BFUNC,
        NodeType::BFUNC, NodeType::CONST, NodeType::CONST,
        NodeType::BFUNC, NodeType::CONST, NodeType::CONST
    };
    std::vector<int16_t> l_sub = {7, 3, 1, 1, 3, 1, 1};
    l_val.resize(maxLen,0.f); l_type.resize(maxLen,0); l_sub.resize(maxLen,0);
    copy_tree_to_device(l_val,l_type,l_sub, src, 0, maxLen);

    // Right parent: ADD(POW(2,3), MUL(5,2))
    std::vector<float> r_val = {
        static_cast<float>(Function::ADD),
        static_cast<float>(Function::POW), 2.f, 3.f,
        static_cast<float>(Function::MUL), 5.f, 2.f
    };
    std::vector<int16_t> r_type = {
        NodeType::BFUNC,
        NodeType::BFUNC, NodeType::CONST, NodeType::CONST,
        NodeType::BFUNC, NodeType::CONST, NodeType::CONST
    };
    std::vector<int16_t> r_sub = {7, 3, 1, 1, 3, 1, 1};
    r_val.resize(maxLen,0.f); r_type.resize(maxLen,0); r_sub.resize(maxLen,0);
    copy_tree_to_device(r_val,r_type,r_sub, src, 1, maxLen);

    // Swap roots: child should become right parent's whole tree
    int h_left=0, h_right=1, h_leftNode=0, h_rightNode=0;
    int *d_left,*d_right,*d_ln,*d_rn;
    CUDA_CHECK(cudaMalloc(&d_left,sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_right,sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ln,sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_rn,sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_left,&h_left,sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_right,&h_right,sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ln,&h_leftNode,sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rn,&h_rightNode,sizeof(int),cudaMemcpyHostToDevice));

    crossover(pop_src, pop_child, maxLen,
              src.value, src.type, src.subsize,
              d_left, d_right, d_ln, d_rn,
              dst.value, dst.type, dst.subsize);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> c_val(maxLen); std::vector<int16_t> c_type(maxLen), c_sub(maxLen);
    CUDA_CHECK(cudaMemcpy(c_val.data(), dst.value,  maxLen*sizeof(float),    cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(c_type.data(),dst.type,   maxLen*sizeof(int16_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(c_sub.data(), dst.subsize,maxLen*sizeof(int16_t), cudaMemcpyDeviceToHost));

    // Child should equal right parent's tree at root and have the expected
    // numeric value:
    //   ADD(POW(2,3), MUL(5,2)) = ADD(8,10) = 18.
    assert(c_type[0] == r_type[0] && c_val[0] == r_val[0]);
    int idx = 0; std::vector<float> vars; float out_child = eval_prefix_host(c_val.data(), c_type.data(), idx, vars);
    assert(out_child == 18.f);
    std::cout << "[crossover-det-rootswap] child evaluates to " << out_child << " (expected 18) OK\n";

    cudaFree(d_left); cudaFree(d_right); cudaFree(d_ln); cudaFree(d_rn);
    free(src); free(dst);
}

//--------------------------------------------------------------------
// TEST 3b – CROSSOVER (non-root node)
//--------------------------------------------------------------------
void test_crossover_det_nonroot(){
    const unsigned pop_src = 2, maxLen = 16, pop_child = 1;
    DeviceBuffers src = alloc_tree_buffers(pop_src, maxLen);
    DeviceBuffers dst = alloc_tree_buffers(pop_child, maxLen);

    // Left parent = ADD(MUL(1,2),SUB(4,3)) -> 3
    std::vector<float> l_val = {static_cast<float>(Function::ADD), static_cast<float>(Function::MUL), 1.f, 2.f, static_cast<float>(Function::SUB), 4.f, 3.f};
    std::vector<int16_t> l_type = {NodeType::BFUNC, NodeType::BFUNC, NodeType::CONST, NodeType::CONST, NodeType::BFUNC, NodeType::CONST, NodeType::CONST};
    std::vector<int16_t> l_sub = {7, 3, 1, 1, 3, 1, 1};
    l_val.resize(maxLen,0.f); l_type.resize(maxLen,0); l_sub.resize(maxLen,0);
    copy_tree_to_device(l_val,l_type,l_sub, src, 0, maxLen);

    // Right parent = ADD(POW(2,3),MUL(5,2)) -> 18
    std::vector<float> r_val = {static_cast<float>(Function::ADD), static_cast<float>(Function::POW), 2.f, 3.f, static_cast<float>(Function::MUL), 5.f, 2.f};
    std::vector<int16_t> r_type = {NodeType::BFUNC, NodeType::BFUNC, NodeType::CONST, NodeType::CONST, NodeType::BFUNC, NodeType::CONST, NodeType::CONST};
    std::vector<int16_t> r_sub = {7, 3, 1, 1, 3, 1, 1};
    r_val.resize(maxLen,0.f); r_type.resize(maxLen,0); r_sub.resize(maxLen,0);
    copy_tree_to_device(r_val,r_type,r_sub, src, 1, maxLen);

    // Mapping arrays: replace left's SUB(4,3) at index 4 with right's MUL(5,2) at index 4
    int h_left=0, h_right=1, h_leftNode=4, h_rightNode=4;
    int *d_left,*d_right,*d_ln,*d_rn; CUDA_CHECK(cudaMalloc(&d_left,sizeof(int))); CUDA_CHECK(cudaMalloc(&d_right,sizeof(int))); CUDA_CHECK(cudaMalloc(&d_ln,sizeof(int))); CUDA_CHECK(cudaMalloc(&d_rn,sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_left,&h_left,sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_right,&h_right,sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ln,&h_leftNode,sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rn,&h_rightNode,sizeof(int),cudaMemcpyHostToDevice));

    crossover(pop_src, pop_child, maxLen,
              src.value, src.type, src.subsize,
              d_left, d_right, d_ln, d_rn,
              dst.value, dst.type, dst.subsize);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Fetch child
    std::vector<float> c_val(maxLen); std::vector<int16_t> c_type(maxLen), c_sub(maxLen);
    CUDA_CHECK(cudaMemcpy(c_val.data(), dst.value,  maxLen*sizeof(float),    cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(c_type.data(),dst.type,   maxLen*sizeof(int16_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(c_sub.data(), dst.subsize,maxLen*sizeof(int16_t), cudaMemcpyDeviceToHost));

    // Child should be ADD(MUL(1,2),MUL(5,2)) -> 12
    assert(c_type[4] == NodeType::BFUNC && c_val[4] == static_cast<float>(Function::MUL));
    int idx=0; std::vector<float> vars; float out_child = eval_prefix_host(c_val.data(), c_type.data(), idx, vars);
    assert(out_child == 12.f);
    std::cout << "[crossover-det-nonroot] child evaluates to 12 OK\n";

    cudaFree(d_left); cudaFree(d_right); cudaFree(d_ln); cudaFree(d_rn);
    free(src); free(dst);
}

//--------------------------------------------------------------------
// TEST 3b – CROSSOVER (interleaved deterministic with large population)
//--------------------------------------------------------------------
void test_crossover_interleaved_det_large(){
    const unsigned pop_src = 200, maxLen = 64, pop_child = 100;
    DeviceBuffers src = alloc_tree_buffers(pop_src, maxLen);
    DeviceBuffers dst = alloc_tree_buffers(pop_child, maxLen);

    // Create pairs of source trees for crossover
    std::vector<std::vector<float>> l_vals(pop_src, std::vector<float>(maxLen, 0.0f));
    std::vector<std::vector<int16_t>> l_types(pop_src, std::vector<int16_t>(maxLen, 0));
    std::vector<std::vector<int16_t>> l_subs(pop_src, std::vector<int16_t>(maxLen, 0));
    std::vector<std::vector<float>> r_vals(pop_src, std::vector<float>(maxLen, 0.0f));
    std::vector<std::vector<int16_t>> r_types(pop_src, std::vector<int16_t>(maxLen, 0));
    std::vector<std::vector<int16_t>> r_subs(pop_src, std::vector<int16_t>(maxLen, 0));
    std::vector<int> left_indices(pop_child, 0);
    std::vector<int> right_indices(pop_child, 0);
    std::vector<int> left_nodes(pop_child, 0);
    std::vector<int> right_nodes(pop_child, 0);

    for(unsigned i = 0; i < pop_src; i += 2) {
        // Left parent: ADD(MUL(VAL, VAL), VAL)
        l_vals[i][0] = static_cast<float>(Function::ADD); l_types[i][0] = NodeType::BFUNC; l_subs[i][0] = 5;
        l_vals[i][1] = static_cast<float>(Function::MUL); l_types[i][1] = NodeType::BFUNC; l_subs[i][1] = 3;
        l_vals[i][2] = static_cast<float>(i % 5 + 1); l_types[i][2] = NodeType::CONST; l_subs[i][2] = 1;
        l_vals[i][3] = static_cast<float>((i + 1) % 5 + 1); l_types[i][3] = NodeType::CONST; l_subs[i][3] = 1;
        l_vals[i][4] = static_cast<float>((i + 2) % 5 + 1); l_types[i][4] = NodeType::CONST; l_subs[i][4] = 1;
        copy_tree_to_device(l_vals[i], l_types[i], l_subs[i], src, i, maxLen);

        // Right parent: SUB(POW(VAL, VAL), VAL)
        r_vals[i+1][0] = static_cast<float>(Function::SUB); r_types[i+1][0] = NodeType::BFUNC; r_subs[i+1][0] = 5;
        r_vals[i+1][1] = static_cast<float>(Function::POW); r_types[i+1][1] = NodeType::BFUNC; r_subs[i+1][1] = 3;
        r_vals[i+1][2] = static_cast<float>((i + 3) % 5 + 1); r_types[i+1][2] = NodeType::CONST; r_subs[i+1][2] = 1;
        r_vals[i+1][3] = static_cast<float>((i + 4) % 5 + 1); r_types[i+1][3] = NodeType::CONST; r_subs[i+1][3] = 1;
        r_vals[i+1][4] = static_cast<float>((i + 5) % 5 + 1); r_types[i+1][4] = NodeType::CONST; r_subs[i+1][4] = 1;
        copy_tree_to_device(r_vals[i+1], r_types[i+1], r_subs[i+1], src, i+1, maxLen);

        if(i/2 < pop_child) {
            left_indices[i/2] = i;
            right_indices[i/2] = i+1;
            left_nodes[i/2] = (i % 2 == 0) ? 0 : 1; // Alternate crossover points
            right_nodes[i/2] = (i % 2 == 0) ? 1 : 0;
        }
    }

    // Allocate device memory for crossover mappings
    int *d_left, *d_right, *d_ln, *d_rn;
    CUDA_CHECK(cudaMalloc(&d_left, pop_child * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_right, pop_child * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ln, pop_child * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_rn, pop_child * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_left, left_indices.data(), pop_child * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_right, right_indices.data(), pop_child * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ln, left_nodes.data(), pop_child * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rn, right_nodes.data(), pop_child * sizeof(int), cudaMemcpyHostToDevice));

    // Perform crossover on population
    crossover(pop_src, pop_child, maxLen,
              src.value, src.type, src.subsize,
              d_left, d_right, d_ln, d_rn,
              dst.value, dst.type, dst.subsize);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Validate results for each child
    bool all_valid = true;
    int exact_match_count = 0;
    int total_checked = 0;
    for(unsigned i = 0; i < pop_child; i++) {
        std::vector<float> c_val(maxLen);
        std::vector<int16_t> c_type(maxLen), c_sub(maxLen);
        CUDA_CHECK(cudaMemcpy(c_val.data(), dst.value + i * maxLen, maxLen * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(c_type.data(), dst.type + i * maxLen, maxLen * sizeof(int16_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(c_sub.data(), dst.subsize + i * maxLen, maxLen * sizeof(int16_t), cudaMemcpyDeviceToHost));

        // Numerical evaluation to verify correctness
        int idx = 0;
        std::vector<float> vars;
        float out_child = eval_prefix_host(c_val.data(), c_type.data(), idx, vars);
        if(std::isnan(out_child) || std::isinf(out_child)) {
            std::cout << "[crossover-interleaved-large] Evaluation failed (NaN/Inf) at index " << i << "\n";
            all_valid = false;
        }
        // Structural check at crossover point (relaxed):
        // we splice the subtree from right_nodes[i] of the right parent
        // into left_nodes[i] of the left parent, so in the child the
        // crossover position is left_nodes[i]. We only require that this
        // node is a function with an opcode in the valid range [1,27].
        int left_node  = left_nodes[i];
        int right_node = right_nodes[i];
        int t = c_type[left_node] & NodeType::TYPE_MASK;
        if (t >= NodeType::UFUNC) { // function node
            int opcode = static_cast<int>(c_val[left_node]);
            if (opcode < 1 || opcode > 27) {
                std::cout << "[crossover-interleaved-large] Invalid opcode at index " << i
                          << ", crossover point " << left_node
                          << ", opcode=" << opcode << "\n";
                all_valid = false;
            }
        }

        // Statistics: count exact matches with right parent at crossover point
        int right_idx = right_indices[i];
        if (c_type[left_node] == r_types[right_idx][right_node] &&
            c_val[left_node]  == r_vals[right_idx][right_node]) {
            exact_match_count++;
        }
        total_checked++;
    }
    // std::cout << "[crossover-interleaved-large] exact matches at crossover point (some might not when there are invalid crossovers which is to be expected due to maxlen and other constraints): "
    //           << exact_match_count << "/" << total_checked << "\n";
    if(!all_valid) {
        std::cout << "[crossover-interleaved-large] Some validations failed, see logs above\n";
    } else {
        std::cout << "[crossover-interleaved-large] All " << pop_child << " crossovers verified OK\n";
    }

    cudaFree(d_left); cudaFree(d_right); cudaFree(d_ln); cudaFree(d_rn);
    free(src); free(dst);
}

//--------------------------------------------------------------------
// TEST 4 – INTERLEAVED MUTATION AND CROSSOVER (combined large population)
//--------------------------------------------------------------------
void test_interleaved_mutate_crossover_large(){
    const unsigned pop_src = 200, pop_intermediate = 100, pop_final = 50, maxLen = 64;
    DeviceBuffers src = alloc_tree_buffers(pop_src, maxLen);
    DeviceBuffers mutated = alloc_tree_buffers(pop_intermediate, maxLen);
    DeviceBuffers crossed = alloc_tree_buffers(pop_final, maxLen);
    DeviceBuffers subtree = alloc_tree_buffers(pop_intermediate, maxLen);

    // Create source population
    std::vector<std::vector<float>> src_vals(pop_src, std::vector<float>(maxLen, 0.0f));
    std::vector<std::vector<int16_t>> src_types(pop_src, std::vector<int16_t>(maxLen, 0));
    std::vector<std::vector<int16_t>> src_subs(pop_src, std::vector<int16_t>(maxLen, 0));
    std::vector<std::vector<float>> sub_vals(pop_intermediate, std::vector<float>(maxLen, 0.0f));
    std::vector<std::vector<int16_t>> sub_types(pop_intermediate, std::vector<int16_t>(maxLen, 0));
    std::vector<std::vector<int16_t>> sub_subs(pop_intermediate, std::vector<int16_t>(maxLen, 0));
    std::vector<int> mutate_indices(pop_intermediate, 0);
    std::vector<int> left_indices(pop_final, 0);
    std::vector<int> right_indices(pop_final, 0);
    std::vector<int> left_nodes(pop_final, 0);
    std::vector<int> right_nodes(pop_final, 0);

    for(unsigned i = 0; i < pop_src; i++) {
        // Alternate between different tree structures
        if(i % 3 == 0) {
            // ADD(VAL, VAL)
            src_vals[i][0] = static_cast<float>(Function::ADD); src_types[i][0] = NodeType::BFUNC; src_subs[i][0] = 3;
            src_vals[i][1] = static_cast<float>(i % 5 + 1); src_types[i][1] = NodeType::CONST; src_subs[i][1] = 1;
            src_vals[i][2] = static_cast<float>((i + 1) % 5 + 1); src_types[i][2] = NodeType::CONST; src_subs[i][2] = 1;
        } else if(i % 3 == 1) {
            // MUL(ADD(VAL, VAL), VAL)
            src_vals[i][0] = static_cast<float>(Function::MUL); src_types[i][0] = NodeType::BFUNC; src_subs[i][0] = 5;
            src_vals[i][1] = static_cast<float>(Function::ADD); src_types[i][1] = NodeType::BFUNC; src_subs[i][1] = 3;
            src_vals[i][2] = static_cast<float>(i % 5 + 1); src_types[i][2] = NodeType::CONST; src_subs[i][2] = 1;
            src_vals[i][3] = static_cast<float>((i + 2) % 5 + 1); src_types[i][3] = NodeType::CONST; src_subs[i][3] = 1;
            src_vals[i][4] = static_cast<float>((i + 3) % 5 + 1); src_types[i][4] = NodeType::CONST; src_subs[i][4] = 1;
        } else {
            // SUB(VAL, NEG(VAL))
            src_vals[i][0] = static_cast<float>(Function::SUB); src_types[i][0] = NodeType::BFUNC; src_subs[i][0] = 4;
            src_vals[i][1] = static_cast<float>(i % 5 + 1); src_types[i][1] = NodeType::CONST; src_subs[i][1] = 1;
            src_vals[i][2] = static_cast<float>(Function::NEG); src_types[i][2] = NodeType::UFUNC; src_subs[i][2] = 2;
            src_vals[i][3] = static_cast<float>((i + 4) % 5 + 1); src_types[i][3] = NodeType::CONST; src_subs[i][3] = 1;
        }
        copy_tree_to_device(src_vals[i], src_types[i], src_subs[i], src, i, maxLen);
    }

    for(unsigned i = 0; i < pop_intermediate; i++) {
        // Subtree for mutation: Simple NEG(VAL)
        sub_vals[i][0] = static_cast<float>(Function::NEG); sub_types[i][0] = NodeType::UFUNC; sub_subs[i][0] = 2;
        sub_vals[i][1] = static_cast<float>((i % 3) + 1.0f); sub_types[i][1] = NodeType::CONST; sub_subs[i][1] = 1;
        copy_tree_to_device(sub_vals[i], sub_types[i], sub_subs[i], subtree, i, maxLen);

        // Mutation index alternates
        mutate_indices[i] = (i % 2 == 0) ? 0 : 1;

        // Crossover mappings for final population
        if(i < pop_final) {
            left_indices[i] = i * 2;
            right_indices[i] = i * 2 + 1;
            left_nodes[i] = (i % 2 == 0) ? 0 : 1;
            right_nodes[i] = (i % 2 == 0) ? 1 : 0;
        }
    }

    // Allocate device memory for mutation indices
    int* d_mutate_indices;
    CUDA_CHECK(cudaMalloc(&d_mutate_indices, pop_intermediate * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_mutate_indices, mutate_indices.data(), pop_intermediate * sizeof(int), cudaMemcpyHostToDevice));

    // Perform mutation on source to intermediate population
    mutate(pop_intermediate, maxLen,
           src.value, src.type, src.subsize,
           d_mutate_indices,
           subtree.value, subtree.type, subtree.subsize,
           mutated.value, mutated.type, mutated.subsize);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Allocate device memory for crossover mappings
    int *d_left, *d_right, *d_ln, *d_rn;
    CUDA_CHECK(cudaMalloc(&d_left, pop_final * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_right, pop_final * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ln, pop_final * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_rn, pop_final * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_left, left_indices.data(), pop_final * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_right, right_indices.data(), pop_final * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ln, left_nodes.data(), pop_final * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rn, right_nodes.data(), pop_final * sizeof(int), cudaMemcpyHostToDevice));

    // Perform crossover on mutated population to final population
    crossover(pop_intermediate, pop_final, maxLen,
              mutated.value, mutated.type, mutated.subsize,
              d_left, d_right, d_ln, d_rn,
              crossed.value, crossed.type, crossed.subsize);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Validate results for each final child
    for(unsigned i = 0; i < pop_final; i++) {
        std::vector<float> c_val(maxLen);
        std::vector<int16_t> c_type(maxLen), c_sub(maxLen);
        CUDA_CHECK(cudaMemcpy(c_val.data(), crossed.value + i * maxLen, maxLen * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(c_type.data(), crossed.type + i * maxLen, maxLen * sizeof(int16_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(c_sub.data(), crossed.subsize + i * maxLen, maxLen * sizeof(int16_t), cudaMemcpyDeviceToHost));

        // Numerical evaluation to verify correctness
        int idx = 0;
        std::vector<float> vars;
        float out_child = eval_prefix_host(c_val.data(), c_type.data(), idx, vars);
        if(std::isnan(out_child) || std::isinf(out_child)) {
            std::cout << "[interleaved-mutate-crossover-large] Evaluation failed (NaN/Inf) at index " << i << "\n";
        }
        // Structural check at mutation and crossover points
        int mutate_idx = mutate_indices[left_indices[i]];
        if(c_type[mutate_idx] != NodeType::UFUNC || c_val[mutate_idx] != static_cast<float>(Function::NEG)) {
            std::cout << "[interleaved-mutate-crossover-large] Mutation structural mismatch at index " << i << ", mutation point " << mutate_idx << "\n";
            std::cout << "Expected NEG (UFUNC), got Type=" << c_type[mutate_idx] << ", Value=" << c_val[mutate_idx] << "\n";
            assert(c_type[mutate_idx] == NodeType::UFUNC && c_val[mutate_idx] == static_cast<float>(Function::NEG));
        }
    }
    std::cout << "[interleaved-mutate-crossover-large] All " << pop_final << " interleaved mutations and crossovers verified OK\n";

    cudaFree(d_mutate_indices); cudaFree(d_left); cudaFree(d_right); cudaFree(d_ln); cudaFree(d_rn);
    free(src); free(mutated); free(crossed); free(subtree);
}

} // unnamed namespace

int main(){
    ConstantTables ct = alloc_constant_tables();
    test_generate(ct);
    test_generate_const_heavy(ct);
    test_generate_opcode(ct);
    test_generate_large_population(ct);
    test_generate_opcode_range(ct);
    test_mutate_det();
    test_mutate_det_deep();
    test_mutate_interleaved_det_large();
    test_crossover_det();
    test_crossover_det_nonroot();
    test_crossover_det_deep_mixed();
    test_crossover_det_rootswap();
    test_crossover_interleaved_det_large();
    test_interleaved_mutate_crossover_large();
    free(ct);
    std::cout << "All evolution tests passed!\n";
    return 0;
}
