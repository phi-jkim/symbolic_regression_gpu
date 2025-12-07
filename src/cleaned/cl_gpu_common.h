#ifndef CL_GPU_COMMON_H
#define CL_GPU_COMMON_H

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <unordered_map>

#ifndef MAX_NUM_FEATURES
#define MAX_NUM_FEATURES 32
#endif

#define MAX_CACHED_SUBTREES 10000

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// Persistent GPU subtree cache context
struct GPUSubtreeStateContext {
  std::unordered_map<uint64_t, int> hash_to_idx;
  float *d_results = nullptr;                   // [capacity][total_dps]
  int num_cached = 0;
  int capacity = MAX_CACHED_SUBTREES;
  int allocated_dps = 0;

  // Cached buffers for subtree evaluation
  int *d_sub_tokens = nullptr;
  float *d_sub_values = nullptr;
  int *d_sub_offsets = nullptr;
  int *d_sub_lengths = nullptr;
  int *d_sub_slots = nullptr;
  size_t alloc_sub_tokens = 0;
  size_t alloc_sub_values = 0;
  size_t alloc_sub_meta = 0; // for offsets, lengths, slots (num_new)

  // Cached buffers for skeleton evaluation
  int *d_skel_tokens = nullptr;
  float *d_skel_values = nullptr;
  int *d_skel_offsets = nullptr;
  int *d_skel_lengths = nullptr;
  float *d_sq_errors = nullptr;
  double *d_mses = nullptr;
  size_t alloc_skel_tokens = 0;
  size_t alloc_skel_values = 0;
  size_t alloc_skel_meta = 0; // for offsets, lengths (num_exprs)
  size_t alloc_sq_errors = 0; // num_exprs * total_dps
  size_t alloc_mses = 0;      // num_exprs

  // Cached buffer for input data
  float *d_X = nullptr;
  size_t alloc_X = 0;

  ~GPUSubtreeStateContext() {
    if (d_results) CUDA_CHECK(cudaFree(d_results));
    if (d_sub_tokens) CUDA_CHECK(cudaFree(d_sub_tokens));
    if (d_sub_values) CUDA_CHECK(cudaFree(d_sub_values));
    if (d_sub_offsets) CUDA_CHECK(cudaFree(d_sub_offsets));
    if (d_sub_lengths) CUDA_CHECK(cudaFree(d_sub_lengths));
    if (d_sub_slots) CUDA_CHECK(cudaFree(d_sub_slots));
    if (d_skel_tokens) CUDA_CHECK(cudaFree(d_skel_tokens));
    if (d_skel_values) CUDA_CHECK(cudaFree(d_skel_values));
    if (d_skel_offsets) CUDA_CHECK(cudaFree(d_skel_offsets));
    if (d_skel_lengths) CUDA_CHECK(cudaFree(d_skel_lengths));
    if (d_sq_errors) CUDA_CHECK(cudaFree(d_sq_errors));
    if (d_mses) CUDA_CHECK(cudaFree(d_mses));
    if (d_X) CUDA_CHECK(cudaFree(d_X));
  }

  void ensure_capacity(int total_dps) {
    if (d_results == nullptr || total_dps > allocated_dps) {
      if (d_results) CUDA_CHECK(cudaFree(d_results));
      allocated_dps = (int)(total_dps * 1.1);
      size_t size = (size_t)capacity * (size_t)allocated_dps * sizeof(float);
      CUDA_CHECK(cudaMalloc(&d_results, size));
    }
  }

  void ensure_sub_buffers(size_t num_tokens, size_t num_values, size_t num_new) {
    if (num_tokens > alloc_sub_tokens) {
        if (d_sub_tokens) CUDA_CHECK(cudaFree(d_sub_tokens));
        alloc_sub_tokens = (size_t)(num_tokens * 1.5);
        CUDA_CHECK(cudaMalloc(&d_sub_tokens, alloc_sub_tokens * sizeof(int)));
    }
    if (num_values > alloc_sub_values) {
        if (d_sub_values) CUDA_CHECK(cudaFree(d_sub_values));
        alloc_sub_values = (size_t)(num_values * 1.5);
        CUDA_CHECK(cudaMalloc(&d_sub_values, alloc_sub_values * sizeof(float)));
    }
    if (num_new > alloc_sub_meta) {
        if (d_sub_offsets) CUDA_CHECK(cudaFree(d_sub_offsets));
        if (d_sub_lengths) CUDA_CHECK(cudaFree(d_sub_lengths));
        if (d_sub_slots) CUDA_CHECK(cudaFree(d_sub_slots));
        alloc_sub_meta = (size_t)(num_new * 1.5);
        CUDA_CHECK(cudaMalloc(&d_sub_offsets, alloc_sub_meta * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_sub_lengths, alloc_sub_meta * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_sub_slots, alloc_sub_meta * sizeof(int)));
    }
  }

  void ensure_skel_buffers(size_t num_tokens, size_t num_values, size_t num_exprs, size_t total_dps) {
    if (num_tokens > alloc_skel_tokens) {
        if (d_skel_tokens) CUDA_CHECK(cudaFree(d_skel_tokens));
        alloc_skel_tokens = (size_t)(num_tokens * 1.5);
        CUDA_CHECK(cudaMalloc(&d_skel_tokens, alloc_skel_tokens * sizeof(int)));
    }
    if (num_values > alloc_skel_values) {
        if (d_skel_values) CUDA_CHECK(cudaFree(d_skel_values));
        alloc_skel_values = (size_t)(num_values * 1.5);
        CUDA_CHECK(cudaMalloc(&d_skel_values, alloc_skel_values * sizeof(float)));
    }
    if (num_exprs > alloc_skel_meta) {
        if (d_skel_offsets) CUDA_CHECK(cudaFree(d_skel_offsets));
        if (d_skel_lengths) CUDA_CHECK(cudaFree(d_skel_lengths));
        if (d_mses) CUDA_CHECK(cudaFree(d_mses));
        alloc_skel_meta = (size_t)(num_exprs * 1.5);
        alloc_mses = alloc_skel_meta;
        CUDA_CHECK(cudaMalloc(&d_skel_offsets, alloc_skel_meta * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_skel_lengths, alloc_skel_meta * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_mses, alloc_mses * sizeof(double)));
    }
    size_t req_sq_errors = num_exprs * total_dps;
    if (req_sq_errors > alloc_sq_errors) {
        if (d_sq_errors) CUDA_CHECK(cudaFree(d_sq_errors));
        alloc_sq_errors = (size_t)(req_sq_errors * 1.2);
        CUDA_CHECK(cudaMalloc(&d_sq_errors, alloc_sq_errors * sizeof(float)));
    }
  }

  void ensure_X_buffer(size_t size_floats) {
      if (size_floats > alloc_X) {
          if (d_X) CUDA_CHECK(cudaFree(d_X));
          alloc_X = (size_t)(size_floats * 1.2);
          CUDA_CHECK(cudaMalloc(&d_X, alloc_X * sizeof(float)));
      }
  }
};

// Persistent GPU context for simple evaluator
struct SimpleGPUContext {
  // Cached buffers for evaluation
  int *d_tokens = nullptr;
  float *d_values = nullptr;
  int *d_offsets = nullptr;
  int *d_lengths = nullptr;
  
  float *d_sq_errors = nullptr;
  double *d_mses = nullptr;
  
  size_t alloc_tokens = 0;
  size_t alloc_values = 0;
  size_t alloc_meta = 0; // for offsets, lengths (num_exprs)
  size_t alloc_sq_errors = 0; // num_exprs * total_dps
  size_t alloc_mses = 0;      // num_exprs

  // Cached buffer for input data
  float *d_X = nullptr;
  size_t alloc_X = 0;

  ~SimpleGPUContext() {
    if (d_tokens) CUDA_CHECK(cudaFree(d_tokens));
    if (d_values) CUDA_CHECK(cudaFree(d_values));
    if (d_offsets) CUDA_CHECK(cudaFree(d_offsets));
    if (d_lengths) CUDA_CHECK(cudaFree(d_lengths));
    if (d_sq_errors) CUDA_CHECK(cudaFree(d_sq_errors));
    if (d_mses) CUDA_CHECK(cudaFree(d_mses));
    if (d_X) CUDA_CHECK(cudaFree(d_X));
  }

  void ensure_buffers(size_t num_tokens, size_t num_values, size_t num_exprs, size_t total_dps) {
    if (num_tokens > alloc_tokens) {
        if (d_tokens) CUDA_CHECK(cudaFree(d_tokens));
        alloc_tokens = (size_t)(num_tokens * 1.5);
        CUDA_CHECK(cudaMalloc(&d_tokens, alloc_tokens * sizeof(int)));
    }
    if (num_values > alloc_values) {
        if (d_values) CUDA_CHECK(cudaFree(d_values));
        alloc_values = (size_t)(num_values * 1.5);
        CUDA_CHECK(cudaMalloc(&d_values, alloc_values * sizeof(float)));
    }
    if (num_exprs > alloc_meta) {
        if (d_offsets) CUDA_CHECK(cudaFree(d_offsets));
        if (d_lengths) CUDA_CHECK(cudaFree(d_lengths));
        if (d_mses) CUDA_CHECK(cudaFree(d_mses));
        alloc_meta = (size_t)(num_exprs * 1.5);
        alloc_mses = alloc_meta;
        CUDA_CHECK(cudaMalloc(&d_offsets, alloc_meta * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_lengths, alloc_meta * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_mses, alloc_mses * sizeof(double)));
    }
    size_t req_sq_errors = num_exprs * total_dps;
    if (req_sq_errors > alloc_sq_errors) {
        if (d_sq_errors) CUDA_CHECK(cudaFree(d_sq_errors));
        alloc_sq_errors = (size_t)(req_sq_errors * 1.2);
        CUDA_CHECK(cudaMalloc(&d_sq_errors, alloc_sq_errors * sizeof(float)));
    }
  }

  void ensure_X_buffer(size_t size_floats) {
      if (size_floats > alloc_X) {
          if (d_X) CUDA_CHECK(cudaFree(d_X));
          alloc_X = (size_t)(size_floats * 1.2);
          CUDA_CHECK(cudaMalloc(&d_X, alloc_X * sizeof(float)));
      }
  }
};

#endif // CL_GPU_COMMON_H
