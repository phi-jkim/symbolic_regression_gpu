#pragma once
// GPU evolution kernels for symbolic_regression_gpu
// Signatures deliberately match evogp::kernel.h so upstream code can be shared.

#include <cuda_runtime.h>
#include <stdint.h>

// All constants / enums live in defs.h (re-export opcodes)
#include "defs.h"

#ifdef __cplusplus
extern "C" {
#endif

void generate(
    const unsigned int popSize,
    const unsigned int maxGPLen,
    const unsigned int varLen,
    const unsigned int outLen,
    const unsigned int constSamplesLen,
    const float outProb,
    const float constProb,
    const unsigned int* keys,
    const float* depth2leafProbs,
    const float* rouletteFuncs,
    const float* constSamples,
    float* value_res,
    int16_t* type_res,
    int16_t* subtree_size_res);

void mutate(
    int popSize,
    int gpLen,
    const float* value_ori,
    const int16_t* type_ori,
    const int16_t* subtree_size_ori,
    const int* mutateIndices,
    const float* value_new,
    const int16_t* type_new,
    const int16_t* subtree_size_new,
    float* value_res,
    int16_t* type_res,
    int16_t* subtree_size_res);

void crossover(
    const int pop_size_ori,
    const int pop_size_new,
    const int gpLen,
    const float* value_ori,
    const int16_t* type_ori,
    const int16_t* subtree_size_ori,
    const int* left_idx,
    const int* right_idx,
    const int* left_node_idx,
    const int* right_node_idx,
    float* value_res,
    int16_t* type_res,
    int16_t* subtree_size_res);

#ifdef __cplusplus
}
#endif
