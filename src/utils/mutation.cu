#include "gpu_kernel.h"
#include "defs.h"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

// internal helper identical to evogp::_gpTreeReplace but using our NodeType
__host__ __device__ inline void _gpTreeReplace(
    const int old_node_idx,
    const int new_node_idx,
    const int new_subsize,
    const int old_offset,
    const int old_size,
    const int size_diff,
    const float* value_old,
    const int16_t* type_old,
    const int16_t* subtree_size_old,
    const float* value_new,
    const int16_t* type_new,
    const int16_t* subtree_size_new,
    float* value_res,
    int16_t* type_res,
    int16_t* subtree_size_res)
{
    float*   value_stack       = (float*)alloca(MAX_STACK*sizeof(float));
    int16_t* type_stack        = (int16_t*)alloca(MAX_STACK*sizeof(int16_t));
    int16_t* subtree_stack     = (int16_t*)alloca(MAX_STACK*sizeof(int16_t));

    for(int i=0;i<old_node_idx;i++) {
        value_stack[i]      = value_old[i];
        type_stack[i]       = type_old[i];
        subtree_stack[i]    = subtree_size_old[i];
    }

    int current=0;
    while(current<old_node_idx) {
        subtree_stack[current] += size_diff;
        int16_t node_type = type_stack[current] & NodeType::TYPE_MASK;
        current++;
        if(current>=old_node_idx) break;
        switch(node_type) {
            case NodeType::UFUNC: break;
            case NodeType::BFUNC: {
                int rightIdx = current + subtree_stack[current];
                if(old_node_idx >= rightIdx) current = rightIdx;
                break;}
            case NodeType::TFUNC: {
                int midIdx  = current + subtree_stack[current];
                if(old_node_idx < midIdx) break;
                int rightIdx = midIdx + subtree_stack[midIdx];
                current = (old_node_idx < rightIdx) ? midIdx : rightIdx; break; }
            default: break;
        }
    }

    for(int i=0;i<new_subsize;i++) {
        value_stack[i+old_node_idx]    = value_new[i+new_node_idx];
        type_stack[i+old_node_idx]     = type_new[i+new_node_idx];
        subtree_stack[i+old_node_idx]  = subtree_size_new[i+new_node_idx];
    }

    for(int i=old_offset;i<old_size;i++) {
        value_stack[i+size_diff]    = value_old[i];
        type_stack[i+size_diff]     = type_old[i];
        subtree_stack[i+size_diff]  = subtree_size_old[i];
    }

    const int len = subtree_stack[0];
    for(int i=0;i<len;i++) {
        value_res[i]       = value_stack[i];
        type_res[i]        = type_stack[i];
        subtree_size_res[i]= subtree_stack[i];
    }
}

__global__ void treeGPMutationKernel(
    const float* value_ori,
    const int16_t* type_ori,
    const int16_t* subtree_size_ori,
    const int* mutateIndices,
    const float* value_new,
    const int16_t* type_new,
    const int16_t* subtree_size_new,
    float* value_res,
    int16_t* type_res,
    int16_t* subtree_size_res,
    const int popSize,
    const int maxGPLen)
{
    const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
    if(n>=popSize) return;

    auto out_val = value_res + n*maxGPLen;
    auto out_type = type_res + n*maxGPLen;
    auto out_sub  = subtree_size_res + n*maxGPLen;

    auto old_val = value_ori + n*maxGPLen;
    auto old_type = type_ori + n*maxGPLen;
    auto old_sub = subtree_size_ori + n*maxGPLen;

    const int node_idx = mutateIndices[n];
    const int old_size = old_sub[0];
    if(node_idx<0 || node_idx>=old_size) {
        for(int i=0;i<old_size;i++) {
            out_val[i]=old_val[i]; out_type[i]=old_type[i]; out_sub[i]=old_sub[i];
        }
        return;
    }

    auto new_val = value_new + n*maxGPLen;
    auto new_type = type_new + n*maxGPLen;
    auto new_sub = subtree_size_new + n*maxGPLen;

    const int oldSubtree = old_sub[node_idx];
    const int newSubtree = new_sub[0];
    const int sizeDiff   = newSubtree - oldSubtree;
    const int oldOffset  = node_idx + oldSubtree;
    if(old_size + sizeDiff > maxGPLen) {
        for(int i=0;i<old_size;i++) {
            out_val[i]=old_val[i]; out_type[i]=old_type[i]; out_sub[i]=old_sub[i];
        }
        return;
    }

    _gpTreeReplace(node_idx,0,newSubtree,oldOffset,old_size,sizeDiff,old_val,old_type,old_sub,new_val,new_type,new_sub,out_val,out_type,out_sub);
}

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
    int16_t* subtree_size_res)
{
    int gridSize=0, blockSize=0;
    cudaOccupancyMaxPotentialBlockSize(&gridSize,&blockSize,treeGPMutationKernel);
    if(gridSize*blockSize<popSize) gridSize = (popSize-1)/blockSize+1;
    treeGPMutationKernel<<<gridSize, blockSize>>>(value_ori,type_ori,subtree_size_ori,mutateIndices,value_new,type_new,subtree_size_new,value_res,type_res,subtree_size_res,popSize,gpLen);
}
