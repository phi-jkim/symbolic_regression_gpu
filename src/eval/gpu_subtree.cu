#include "../utils/detect.h"
#include "../utils/utils.h"
#include "evaluator.h"
#include <algorithm>
#include <cuda_runtime.h>
#include <iostream>
#include <unordered_map>
#include <vector>

// ============================================================================
// Constants & Macros
// ============================================================================

#define OP_REF -2 // Special token for referencing a cached subtree result
#define MAX_CACHED_SUBTREES 500000 // Max number of subtrees to cache in VRAM

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// ============================================================================
// Data Structures
// ============================================================================

// Persistent context for GPU Subtree Evaluation
struct GPUSubtreeContext {
  std::unordered_map<uint64_t, int> hash_to_idx;
  float *d_results = nullptr; // device-side cached subtree results (float)
  int num_cached = 0;
  int capacity = MAX_CACHED_SUBTREES;
  int allocated_dps = 0;

  ~GPUSubtreeContext() {
    if (d_results)
      CUDA_CHECK(cudaFree(d_results));
  }

  void ensure_capacity(int batch_size) {
    if (d_results == nullptr || batch_size > allocated_dps) {
      if (d_results)
        CUDA_CHECK(cudaFree(d_results));

      allocated_dps = (int)(batch_size * 1.2); 
      size_t size = (size_t)capacity * (size_t)allocated_dps * sizeof(float);

      CUDA_CHECK(cudaMalloc(&d_results, size));

      if (num_cached > 0) {
        // Invalidate cache if we resize (though typically batch size is constant)
        // For simplicity, we just clear it, but in a batched loop we shouldn't resize often.
        // Actually, if we batch over DPs, the cache mapping (hash->idx) is still valid for the *structure*,
        // but the *values* in d_results are overwritten for each batch.
        // So we don't need to clear hash_to_idx, just ensure d_results is big enough.
      }
    }
  }
};

// ============================================================================
// Device Functions (Math Ops)
// ============================================================================

__device__ float eval_op_gpu_double(int op, float val1, float val2) {
  const float DELTA = 1e-9f; 
  const float MAX_VAL = 1e9f;

  switch (op) {
  case 1: return val1 + val2;
  case 2: return val1 - val2;
  case 3: return val1 * val2;
  case 4: return (val2 == 0.0f) ? NAN : val1 / val2;
  case 5: return powf(val1, val2);
  case 6: return fminf(val1, val2);
  case 7: return fmaxf(val1, val2);
  case 8: { // LOOSE_DIV
    float denom = fabsf(val2) <= DELTA ? (val2 < 0.0f ? -DELTA : DELTA) : val2;
    return val1 / denom;
  }
  case 9: // LOOSE_POW
    return (val1 == 0.0f && val2 == 0.0f) ? 0.0f : powf(fabsf(val1), val2);
  case 10: return sinf(val1);
  case 11: return cosf(val1);
  case 12: return tanf(val1);
  case 13: return sinhf(val1);
  case 14: return coshf(val1);
  case 15: return tanhf(val1);
  case 16: return expf(val1);
  case 17: return logf(val1);
  case 18: return 1.0f / val1;
  case 19: return asinf(val1);
  case 20: return acosf(val1);
  case 21: return atanf(val1);
  case 22: return (val1 == 0.0f) ? -MAX_VAL : logf(fabsf(val1)); // LOOSE_LOG
  case 23: {                                           // LOOSE_INV
    float denom = fabsf(val1) <= DELTA ? (val1 < 0.0f ? -DELTA : DELTA) : val1;
    return 1.0f / denom;
  }
  case 24: return fabsf(val1);
  case 25: return -val1;
  case 26: return sqrtf(val1);
  case 27: return sqrtf(fabsf(val1));
  default: return 0.0f;
  }
}

// ============================================================================
// Kernels
// ============================================================================

#define STACK_PUSH(v) do { stk[sp] = (v); sp++; } while(0)
#define STACK_POP() (stk[--sp])
#define MAX_NUM_FEATURES 16 // Limit for register caching

// Stage 1: Evaluate Subtrees (Multi-Subtree per Block)
__global__ void eval_subtrees_kernel(
    const int *d_tokens,   
    const float *d_values, 
    const int *d_offsets,  
    const int *d_lengths,  
    const float *d_X_col_major, // [num_vars+1][batch_size]
    float *d_results,      
    const int *d_cache_indices, 
    int num_new_subtrees, int num_vars, int batch_size, int subtrees_per_block) {
  
  int block_sub_start = blockIdx.x * subtrees_per_block;
  int block_sub_end = min(block_sub_start + subtrees_per_block, num_new_subtrees);
  int dp_start = blockIdx.y * blockDim.x;
  int dp_idx = dp_start + threadIdx.x; 

  if (dp_idx >= batch_size) return;

  // Cache X in registers (Coalesced load)
  float x_cache[MAX_NUM_FEATURES];
  int limit_vars = (num_vars + 1 < MAX_NUM_FEATURES) ? num_vars + 1 : MAX_NUM_FEATURES;
  
  for (int v = 0; v < limit_vars; v++) {
    x_cache[v] = d_X_col_major[v * batch_size + dp_idx];
  }

  // Process multiple subtrees
  for (int sub_idx = block_sub_start; sub_idx < block_sub_end; sub_idx++) {
    int offset = d_offsets[sub_idx];
    int len = d_lengths[sub_idx];
    int cache_idx = d_cache_indices[sub_idx];

    float stk[MAX_STACK_SIZE]; 
    int sp = 0;

    for (int i = len - 1; i >= 0; i--) {
      int tok = d_tokens[offset + i];
      if (tok > 0) { 
        float v1 = STACK_POP();
        float v2 = 0.0f;
        if (tok < 10)
          v2 = STACK_POP(); 
        STACK_PUSH(eval_op_gpu_double(tok, v1, v2));
      } else if (tok == 0) { 
        STACK_PUSH(d_values[offset + i]);
      } else if (tok == -1) { 
        int var_idx = (int)d_values[offset + i];
        if (var_idx < limit_vars)
            STACK_PUSH(x_cache[var_idx]);
        else
            STACK_PUSH(d_X_col_major[var_idx * batch_size + dp_idx]); // Fallback
      }
    }

    d_results[cache_idx * batch_size + dp_idx] = stk[0];
  }
}

// Stage 2: Evaluate Skeleton Expressions (Multi-Expr per Block)
__global__ void
eval_skeletons_kernel(const int *d_tokens, const float *d_values, 
                      const int *d_offsets, const int *d_lengths,
                      const float *d_X_col_major, 
                      const float *d_cache_results, 
                      float *d_preds,               
                      int num_exprs, int num_vars, int batch_size, int exprs_per_block) {
  
  int block_expr_start = blockIdx.x * exprs_per_block;
  int block_expr_end = min(block_expr_start + exprs_per_block, num_exprs);
  int dp_start = blockIdx.y * blockDim.x;
  int dp_idx = dp_start + threadIdx.x; 

  if (dp_idx >= batch_size) return;

  // Cache X in registers
  float x_cache[MAX_NUM_FEATURES];
  int limit_vars = (num_vars + 1 < MAX_NUM_FEATURES) ? num_vars + 1 : MAX_NUM_FEATURES;
  
  for (int v = 0; v < limit_vars; v++) {
    x_cache[v] = d_X_col_major[v * batch_size + dp_idx];
  }

  for (int expr_idx = block_expr_start; expr_idx < block_expr_end; expr_idx++) {
    int offset = d_offsets[expr_idx];
    int len = d_lengths[expr_idx];

    float stk[MAX_STACK_SIZE]; 
    int sp = 0;

    for (int i = len - 1; i >= 0; i--) {
      int tok = d_tokens[offset + i];

      if (tok == OP_REF) {
        int cache_idx = (int)d_values[offset + i];
        STACK_PUSH(d_cache_results[cache_idx * batch_size + dp_idx]);
      } else if (tok > 0) {
        float v1 = STACK_POP();
        float v2 = 0.0f;
        if (tok < 10)
          v2 = STACK_POP();
        STACK_PUSH(eval_op_gpu_double(tok, v1, v2));
      } else if (tok == 0) {
        STACK_PUSH(d_values[offset + i]);
      } else if (tok == -1) {
        int var_idx = (int)d_values[offset + i];
        if (var_idx < limit_vars)
            STACK_PUSH(x_cache[var_idx]);
        else
            STACK_PUSH(d_X_col_major[var_idx * batch_size + dp_idx]);
      }
    }

    d_preds[expr_idx * batch_size + dp_idx] = stk[0]; 
  }
}

// ============================================================================
// Host Functions
// ============================================================================

// Helper to flatten X (Column-Major) for a specific batch
// [v0_dp0, v0_dp1, ..., v1_dp0, v1_dp1, ...]
// Host data is double, but we convert to float for GPU evaluation.
float *flatten_X_col_major_batch(double **vars, int num_vars, int dp_start, int batch_size) {
  float *flat = new float[batch_size * (num_vars + 1)];
  for (int v = 0; v <= num_vars; v++) {
    for (int i = 0; i < batch_size; i++) {
      flat[v * batch_size + i] = (float)vars[v][dp_start + i];
    }
  }
  return flat;
}

// Helper: Compute hash for a subtree
uint64_t compute_subtree_hash(const int *tokens, const double *values, int len) {
  uint64_t h = 14695981039346656037ULL; 
  for (int i = 0; i < len; i++) {
    h ^= (uint64_t)tokens[i];
    h *= 1099511628211ULL; 
    
    uint64_t v_bits;
    // Use double for hash consistency with values
    double v_val = values[i]; 
    memcpy(&v_bits, &v_val, sizeof(double)); 
    
    h ^= v_bits;
    h *= 1099511628211ULL;
  }
  return h;
}

void eval_gpu_subtree_batch(InputInfo &input_info, double ***all_vars,
                            double **all_predictions, void *state) {
  GPUSubtreeContext *ctx = (GPUSubtreeContext *)state;

  int num_exprs = input_info.num_exprs;
  int num_vars = input_info.num_vars[0];
  int total_dps = input_info.num_dps[0];

  printf("total dps: %d\n", total_dps);
  
  // Batching configuration
  const int BATCH_SIZE = 8192; 
  ctx->ensure_capacity(BATCH_SIZE);

  // Detect Common Subtrees (CPU) - Done once for all datapoints (structure is static)
  SubtreeDetectionResult result = detect_common_subtrees(
      num_exprs, num_vars, input_info.num_tokens,
      (const int **)input_info.tokens, (const double **)input_info.values, 3, 2
  );

  // Identify NEW subtrees and update hash map
  std::vector<int> new_subtree_indices; 
  std::vector<int> new_cache_slots;     
  std::vector<int> subtree_id_to_slot(result.num_subs, -1); 

  for (int i = 0; i < result.num_subs; i++) {
    uint64_t hash = compute_subtree_hash(result.sub_tokens[i], result.sub_values[i], result.num_sub_tokens[i]);
    
    if (ctx->hash_to_idx.find(hash) == ctx->hash_to_idx.end()) {
      if (ctx->num_cached >= ctx->capacity) {
        // Simple eviction strategy: clear everything if full (for now)
        // Ideally we would use LRU, but for simplicity we just reset.
        // However, resetting invalidates existing slots, so we'd need to re-evaluate ALL subtrees.
        // For this implementation, we'll just stop caching new ones if full.
        // std::cerr << "GPU Cache Full! Skipping subtree caching." << std::endl;
        continue; 
      }
      int slot = ctx->num_cached++;
      ctx->hash_to_idx[hash] = slot;
      new_subtree_indices.push_back(i);
      new_cache_slots.push_back(slot);
      subtree_id_to_slot[i] = slot;
    } else {
      subtree_id_to_slot[i] = ctx->hash_to_idx[hash];
    }
  }

  // Prepare Subtree Data (Structure) - Static across batches
  int *d_sub_tokens = nullptr, *d_sub_offsets = nullptr, *d_sub_lengths = nullptr, *d_sub_slots = nullptr;
  float *d_sub_values = nullptr;
  int num_new = new_subtree_indices.size();

  if (num_new > 0) {
    std::vector<int> h_sub_tokens;
    std::vector<float> h_sub_values; 
    std::vector<int> h_sub_offsets;
    std::vector<int> h_sub_lengths;

    for (int idx : new_subtree_indices) {
      h_sub_offsets.push_back(h_sub_tokens.size());
      h_sub_lengths.push_back(result.num_sub_tokens[idx]);

      int *src_toks = result.sub_tokens[idx];
      double *src_vals = result.sub_values[idx];

      for (int k = 0; k < result.num_sub_tokens[idx]; k++) {
        h_sub_tokens.push_back(src_toks[k]);
        h_sub_values.push_back((float)src_vals[k]); 
      }
    }
    
    CUDA_CHECK(cudaMalloc(&d_sub_tokens, h_sub_tokens.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sub_values, h_sub_values.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sub_offsets, num_new * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sub_lengths, num_new * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sub_slots, num_new * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_sub_tokens, h_sub_tokens.data(), h_sub_tokens.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sub_values, h_sub_values.data(), h_sub_values.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sub_offsets, h_sub_offsets.data(), num_new * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sub_lengths, h_sub_lengths.data(), num_new * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sub_slots, new_cache_slots.data(), num_new * sizeof(int), cudaMemcpyHostToDevice));
  }

  // Prepare Skeleton Data (Structure) - Static across batches
  int *d_skel_tokens, *d_skel_offsets, *d_skel_lengths;
  float *d_skel_values; 
  float *d_preds; 
  
  std::vector<int> h_skel_tokens;
  std::vector<float> h_skel_values; 
  std::vector<int> h_skel_offsets;
  std::vector<int> h_skel_lengths;

  for (int i = 0; i < num_exprs; i++) {
    h_skel_offsets.push_back(h_skel_tokens.size());

    int *toks = input_info.tokens[i];
    double *vals = input_info.values[i];
    int len = input_info.num_tokens[i];
    int *hints = result.expr_sub_hints[i];

    for (int k = 0; k < len; k++) {
      int sub_id = hints[k];
      
      if (sub_id >= 0 && subtree_id_to_slot[sub_id] != -1) {
        h_skel_tokens.push_back(OP_REF);
        h_skel_values.push_back((float)subtree_id_to_slot[sub_id]);
        k += result.num_sub_tokens[sub_id] - 1; 
      } else if (sub_id == -2) {
          h_skel_tokens.push_back(toks[k]);
          h_skel_values.push_back((float)vals[k]);
      } else {
        h_skel_tokens.push_back(toks[k]);
        h_skel_values.push_back((float)vals[k]);
      }
    }
    h_skel_lengths.push_back(h_skel_tokens.size() - h_skel_offsets.back());
  }

  CUDA_CHECK(cudaMalloc(&d_skel_tokens, h_skel_tokens.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_skel_values, h_skel_values.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_skel_offsets, num_exprs * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_skel_lengths, num_exprs * sizeof(int)));
  // d_preds size depends on batch size, allocated inside loop or max batch size
  CUDA_CHECK(cudaMalloc(&d_preds, num_exprs * BATCH_SIZE * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_skel_tokens, h_skel_tokens.data(), h_skel_tokens.size() * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_skel_values, h_skel_values.data(), h_skel_values.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_skel_offsets, h_skel_offsets.data(), num_exprs * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_skel_lengths, h_skel_lengths.data(), num_exprs * sizeof(int), cudaMemcpyHostToDevice));

  // ==========================================================================
  // Batched Execution Loop
  // ==========================================================================
  float *d_X;
  CUDA_CHECK(cudaMalloc(&d_X, BATCH_SIZE * (num_vars + 1) * sizeof(float)));
  float *h_preds_batch = new float[num_exprs * BATCH_SIZE];

  float total_kernel_ms = 0.0f;
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  for (int dp_start = 0; dp_start < total_dps; dp_start += BATCH_SIZE) {
    int current_batch_size = std::min(BATCH_SIZE, total_dps - dp_start);

    // 1. Transfer X for this batch (host double -> device float)
    float *h_X = flatten_X_col_major_batch(all_vars[0], num_vars, dp_start, current_batch_size);
    CUDA_CHECK(cudaMemcpy(d_X, h_X, current_batch_size * (num_vars + 1) * sizeof(float), cudaMemcpyHostToDevice));
    delete[] h_X;

    int threads = 128;
    int blocks_y = (current_batch_size + threads - 1) / threads;

    CUDA_CHECK(cudaEventRecord(start));

    // 2. Evaluate NEW Subtrees for this batch
    if (num_new > 0) {
      int subtrees_per_block = 4;
      int blocks_x = (num_new + subtrees_per_block - 1) / subtrees_per_block;
      dim3 grid(blocks_x, blocks_y);

      eval_subtrees_kernel<<<grid, threads>>>(
          d_sub_tokens, d_sub_values, d_sub_offsets, d_sub_lengths, d_X,
          ctx->d_results, d_sub_slots, num_new, num_vars, current_batch_size, subtrees_per_block);
    }

    // 3. Evaluate Skeletons for this batch
    {
      int exprs_per_block = 4;
      int blocks_x = (num_exprs + exprs_per_block - 1) / exprs_per_block;
      dim3 grid(blocks_x, blocks_y);

      eval_skeletons_kernel<<<grid, threads>>>(
          d_skel_tokens, d_skel_values, d_skel_offsets, d_skel_lengths, d_X,
          ctx->d_results, d_preds, num_exprs, num_vars, current_batch_size, exprs_per_block);
    }
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    total_kernel_ms += milliseconds;

    // 4. Copy predictions back
    CUDA_CHECK(cudaMemcpy(h_preds_batch, d_preds, num_exprs * current_batch_size * sizeof(float), cudaMemcpyDeviceToHost));

    // 5. Fill all_predictions
    for (int i = 0; i < num_exprs; i++) {
      for (int j = 0; j < current_batch_size; j++) {
        all_predictions[i][dp_start + j] = (double)h_preds_batch[i * current_batch_size + j];
      }
    }
  }

  std::cout << "[gpu_subtree] Total Kernel Time: " << total_kernel_ms << " ms" << std::endl;

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  // Cleanup
  delete[] h_preds_batch;
  CUDA_CHECK(cudaFree(d_X));
  CUDA_CHECK(cudaFree(d_skel_tokens));
  CUDA_CHECK(cudaFree(d_skel_values));
  CUDA_CHECK(cudaFree(d_skel_offsets));
  CUDA_CHECK(cudaFree(d_skel_lengths));
  CUDA_CHECK(cudaFree(d_preds));

  if (num_new > 0) {
    CUDA_CHECK(cudaFree(d_sub_tokens));
    CUDA_CHECK(cudaFree(d_sub_values));
    CUDA_CHECK(cudaFree(d_sub_offsets));
    CUDA_CHECK(cudaFree(d_sub_lengths));
    CUDA_CHECK(cudaFree(d_sub_slots));
  }

  free_subtree_detection_result(result);
}

void eval_gpu_subtree_batch_stateless(InputInfo &input_info, double ***all_vars,
                                      double **all_predictions,
                                      EvalMetrics *metrics) {
  GPUSubtreeContext ctx;
  
  cudaEvent_t start, stop;
  if (metrics) {
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      cudaEventRecord(start);
  }

  eval_gpu_subtree_batch(input_info, all_vars, all_predictions, &ctx);

  if (metrics) {
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      float milliseconds = 0;
      cudaEventElapsedTime(&milliseconds, start, stop);
      metrics->total_gpu_ms = milliseconds;
      metrics->kernel_exec_ms = milliseconds; 
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
  }
}
