#include "gpu_kernel.h"
#include "defs.h"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

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
    int16_t* subtree_size_res);
// local copy of _gpTreeReplace (identical to mutation.cu)
__host__ __device__ inline void _gpTreeReplace(
    const int old_node_idx, const int new_node_idx, const int new_subsize,
    const int old_offset, const int old_size, const int size_diff,
    const float* value_old, const int16_t* type_old, const int16_t* subtree_old,
    const float* value_new, const int16_t* type_new, const int16_t* subtree_new,
    float* value_res, int16_t* type_res, int16_t* subtree_res) {
    float* vstk = (float*)alloca(MAX_STACK*sizeof(float));
    int16_t* tstk = (int16_t*)alloca(MAX_STACK*sizeof(int16_t));
    int16_t* sstk = (int16_t*)alloca(MAX_STACK*sizeof(int16_t));
    for(int i=0;i<old_node_idx;i++){vstk[i]=value_old[i]; tstk[i]=type_old[i]; sstk[i]=subtree_old[i];}
    int cur=0;
    while(cur<old_node_idx){ sstk[cur]+=size_diff; int16_t nt=tstk[cur]&NodeType::TYPE_MASK; ++cur;
        if(cur>=old_node_idx) break;
        switch(nt){case NodeType::UFUNC: break; case NodeType::BFUNC:{int r = cur + sstk[cur]; if(old_node_idx>=r) cur=r; break;} case NodeType::TFUNC:{int mid=cur+sstk[cur]; if(old_node_idx<mid) break; int r=mid+sstk[mid]; cur=(old_node_idx<r)?mid:r; break;} default: break;}}
    for(int i=0;i<new_subsize;i++){vstk[i+old_node_idx]=value_new[i+new_node_idx]; tstk[i+old_node_idx]=type_new[i+new_node_idx]; sstk[i+old_node_idx]=subtree_new[i+new_node_idx];}
    for(int i=old_offset;i<old_size;i++){vstk[i+size_diff]=value_old[i]; tstk[i+size_diff]=type_old[i]; sstk[i+size_diff]=subtree_old[i];}
    int len=sstk[0]; for(int i=0;i<len;i++){value_res[i]=vstk[i]; type_res[i]=tstk[i]; subtree_res[i]=sstk[i];}
}

__global__ void treeGPCrossoverKernel(
    const int pop_size_ori, const int pop_size_new, const int maxGPLen,
    const float* value_ori, const int16_t* type_ori, const int16_t* subtree_ori,
    const int* left_idx, const int* right_idx, const int* left_node_idx, const int* right_node_idx,
    float* value_res, int16_t* type_res, int16_t* subtree_res)
{
    const unsigned int n = blockIdx.x*blockDim.x + threadIdx.x;
    if(n>=pop_size_new) return;
    auto out_val = value_res + n*maxGPLen;
    auto out_type= type_res + n*maxGPLen;
    auto out_sub = subtree_res + n*maxGPLen;
    auto left_val = value_ori + left_idx[n]*maxGPLen;
    auto left_type= type_ori + left_idx[n]*maxGPLen;
    auto left_sub = subtree_ori + left_idx[n]*maxGPLen;
    const int left_size = left_sub[0];
    if(right_idx[n]<0 || right_idx[n]>=pop_size_ori){
        for(int i=0;i<left_size;i++){out_val[i]=left_val[i]; out_type[i]=left_type[i]; out_sub[i]=left_sub[i];} return; }
    auto right_val=value_ori + right_idx[n]*maxGPLen;
    auto right_type=type_ori + right_idx[n]*maxGPLen;
    auto right_sub=subtree_ori + right_idx[n]*maxGPLen;
    const int left_node_size = left_sub[left_node_idx[n]];
    const int right_node_size = right_sub[right_node_idx[n]];
    const int size_diff = right_node_size - left_node_size;
    const int left_offset = left_node_idx[n] + left_node_size;
    if(left_size + size_diff > maxGPLen){
        for(int i=0;i<left_size;i++){out_val[i]=left_val[i]; out_type[i]=left_type[i]; out_sub[i]=left_sub[i];} return; }
    _gpTreeReplace(left_node_idx[n], right_node_idx[n], right_node_size, left_offset, left_size, size_diff, left_val, left_type, left_sub, right_val, right_type, right_sub, out_val, out_type, out_sub);
}

void crossover(
    const int pop_size_ori, const int pop_size_new, const int gpLen,
    const float* value_ori, const int16_t* type_ori, const int16_t* subtree_ori,
    const int* left_idx, const int* right_idx, const int* left_node_idx, const int* right_node_idx,
    float* value_res, int16_t* type_res, int16_t* subtree_res)
{
    int gridSize=0, blockSize=0;
    cudaOccupancyMaxPotentialBlockSize(&gridSize,&blockSize,treeGPCrossoverKernel);
    if(gridSize*blockSize<pop_size_new) gridSize=(pop_size_new-1)/blockSize+1;
    treeGPCrossoverKernel<<<gridSize, blockSize>>>(pop_size_ori,pop_size_new,gpLen,value_ori,type_ori,subtree_ori,left_idx,right_idx,left_node_idx,right_node_idx,value_res,type_res,subtree_res);
}

