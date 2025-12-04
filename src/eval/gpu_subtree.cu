#include "../utils/detect.h"
#include "../utils/utils.h"
#include <algorithm>
#include <cuda_runtime.h>
#include <iostream>
#include <unordered_map>
#include <vector>

// ============================================================================
// Constants & Macros
// ============================================================================

#define OP_REF -2 // Special token for referencing a cached subtree result
#define MAX_CACHED_SUBTREES 10000 // Max number of subtrees to cache in VRAM

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
  // Host-side map: subtree_hash -> index in d_results
  std::unordered_map<uint64_t, int> hash_to_idx;

  // Device buffer: [MAX_CACHED_SUBTREES * max_num_dps]
  // Stores the evaluation results of all cached subtrees
  float *d_results = nullptr;

  // Current number of cached subtrees
  int num_cached = 0;

  // Capacity of the current d_results buffer (in terms of number of subtrees)
  int capacity = MAX_CACHED_SUBTREES;

  // Max datapoints supported by the allocated buffer
  int allocated_dps = 0;

  ~GPUSubtreeContext() {
    if (d_results)
      CUDA_CHECK(cudaFree(d_results));
  }

  // Ensure d_results is allocated and large enough
  void ensure_capacity(int num_dps) {
    if (d_results == nullptr || num_dps > allocated_dps) {
      if (d_results)
        CUDA_CHECK(cudaFree(d_results));

      allocated_dps = (int)(num_dps * 1.2); // 20% growth factor
      size_t size = (size_t)capacity * (size_t)allocated_dps * sizeof(float);

      // std::cout << "Allocating GPU Cache: " << capacity << " subtrees x "
      //           << allocated_dps << " dps (" << size / 1024 / 1024 << " MB)"
      //           << std::endl;

      CUDA_CHECK(cudaMalloc(&d_results, size));

      // Invalidate cache since we lost the data (simple approach: clear cache
      // on resize) Ideally we would copy old data, but for now let's just reset
      // if DPS changes drastically In evolution benchmarks, DPS is usually
      // constant.
      if (num_cached > 0) {
        std::cerr << "Warning: Resizing GPU cache, clearing " << num_cached
                  << " entries." << std::endl;
        hash_to_idx.clear();
        num_cached = 0;
      }
    }
  }
};

// ============================================================================
// Device Functions (Math Ops)
// ============================================================================

__device__ float eval_op_gpu_float(int op, float val1, float val2) {
  // Simplified float version of eval_op
  const float DELTA = 1e-7f;
  switch (op) {
  case 1:
    return val1 + val2;
  case 2:
    return val1 - val2;
  case 3:
    return val1 * val2;
  case 4:
    return (val2 == 0.0f) ? NAN : val1 / val2;
  case 5:
    return powf(val1, val2);
  case 6:
    return fminf(val1, val2);
  case 7:
    return fmaxf(val1, val2);
  case 8: { // LOOSE_DIV
    float denom = fabsf(val2) <= DELTA ? (val2 < 0 ? -DELTA : DELTA) : val2;
    return val1 / denom;
  }
  case 9: // LOOSE_POW
    return (val1 == 0.0f && val2 == 0.0f) ? 0.0f : powf(fabsf(val1), val2);
  case 10:
    return sinf(val1);
  case 11:
    return cosf(val1);
  case 12:
    return tanf(val1);
  case 13:
    return sinhf(val1);
  case 14:
    return coshf(val1);
  case 15:
    return tanhf(val1);
  case 16:
    return expf(val1);
  case 17:
    return logf(val1);
  case 18:
    return 1.0f / val1;
  case 19:
    return asinf(val1);
  case 20:
    return acosf(val1);
  case 21:
    return atanf(val1);
  case 22:
    return (val1 == 0.0f) ? -1e9f : logf(fabsf(val1)); // LOOSE_LOG
  case 23: {                                           // LOOSE_INV
    float denom = fabsf(val1) <= DELTA ? (val1 < 0 ? -DELTA : DELTA) : val1;
    return 1.0f / denom;
  }
  case 24:
    return fabsf(val1);
  case 25:
    return -val1;
  case 26:
    return sqrtf(val1);
  case 27:
    return sqrtf(fabsf(val1));
  default:
    return 0.0f;
  }
}

// ============================================================================
// Kernels
// ============================================================================

// Stack helper
#define STACK_PUSH(v)                                                          \
  {                                                                            \
    stk[sp] = (v);                                                             \
    sp++;                                                                      \
  }
#define STACK_POP() (stk[--sp])

// Stage 1: Evaluate Subtrees
// Each thread evaluates one datapoint for one subtree
// Output: d_results[subtree_idx * num_dps + dp_idx]
__global__ void eval_subtrees_kernel(
    const int *d_tokens,   // Packed tokens for all NEW subtrees
    const float *d_values, // Packed values
    const int *d_offsets,  // [num_new_subtrees]
    const int *d_lengths,  // [num_new_subtrees]
    const float *d_X,      // [num_dps * num_features]
    float *d_results,      // Global cache buffer
    const int
        *d_cache_indices, // [num_new_subtrees] - where to write in d_results
    int num_new_subtrees, int num_vars, int num_dps) {
  int dp_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int sub_idx = blockIdx.y; // Each block Y handles one subtree

  if (dp_idx >= num_dps || sub_idx >= num_new_subtrees)
    return;

  int offset = d_offsets[sub_idx];
  int len = d_lengths[sub_idx];
  int cache_idx = d_cache_indices[sub_idx];

  // Local stack
  float stk[MAX_STACK_SIZE];
  int sp = 0;

  // Evaluate
  for (int i = len - 1; i >= 0; i--) {
    int tok = d_tokens[offset + i];
    if (tok > 0) { // Operator
      float v1 = STACK_POP();
      float v2 = 0.0f;
      if (tok < 10)
        v2 = STACK_POP(); // Binary
      STACK_PUSH(eval_op_gpu_float(tok, v1, v2));
    } else if (tok == 0) { // Const
      STACK_PUSH(d_values[offset + i]);
    } else if (tok == -1) { // Var
      int var_idx = (int)d_values[offset + i];
      STACK_PUSH(d_X[dp_idx * (num_vars + 1) + var_idx]); // Row-major X
    }
  }

  // Write result to cache
  d_results[cache_idx * num_dps + dp_idx] = stk[0];
}

// Stage 2: Evaluate Skeleton Expressions
// Can reference d_results using OP_REF
__global__ void
eval_skeletons_kernel(const int *d_tokens, const float *d_values,
                      const int *d_offsets, const int *d_lengths,
                      const float *d_X,
                      const float *d_cache_results, // Read-only cache
                      float *d_preds,               // Output predictions
                      int num_exprs, int num_vars, int num_dps) {
  int dp_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int expr_idx = blockIdx.y;

  if (dp_idx >= num_dps || expr_idx >= num_exprs)
    return;

  int offset = d_offsets[expr_idx];
  int len = d_lengths[expr_idx];

  float stk[MAX_STACK_SIZE];
  int sp = 0;

  for (int i = len - 1; i >= 0; i--) {
    int tok = d_tokens[offset + i];

    if (tok == OP_REF) {
      // Reference to cached subtree
      int cache_idx = (int)d_values[offset + i];
      STACK_PUSH(d_cache_results[cache_idx * num_dps + dp_idx]);
    } else if (tok > 0) {
      float v1 = STACK_POP();
      float v2 = 0.0f;
      if (tok < 10)
        v2 = STACK_POP();
      STACK_PUSH(eval_op_gpu_float(tok, v1, v2));
    } else if (tok == 0) {
      STACK_PUSH(d_values[offset + i]);
    } else if (tok == -1) {
      int var_idx = (int)d_values[offset + i];
      STACK_PUSH(d_X[dp_idx * (num_vars + 1) + var_idx]);
    }
  }

  d_preds[expr_idx * num_dps + dp_idx] = stk[0];
}

// ============================================================================
// Host Functions
// ============================================================================

// Helper to flatten X (row-major)
float *flatten_X_row_major(double **vars, int num_vars, int num_dps) {
  float *flat = new float[num_dps * (num_vars + 1)];
  for (int dp = 0; dp < num_dps; dp++) {
    for (int v = 0; v <= num_vars; v++) {
      flat[dp * (num_vars + 1) + v] = (float)vars[v][dp];
    }
  }
  return flat;
}

void eval_gpu_subtree_batch(InputInfo &input_info, double ***all_vars,
                            double **all_predictions, void *state) {
  GPUSubtreeContext *ctx = (GPUSubtreeContext *)state;

  int num_exprs = input_info.num_exprs;
  int num_vars = input_info.num_vars[0]; // Assume all same
  int num_dps = input_info.num_dps[0];   // Assume all same

  // 1. Ensure Cache Capacity
  ctx->ensure_capacity(num_dps);

  // 2. Detect Common Subtrees (CPU)
  // We use the existing detect logic but need to adapt it slightly or just
  // re-run it For simplicity, we'll re-run detection on the current batch
  SubtreeDetectionResult result = detect_common_subtrees(
      num_exprs, num_vars, input_info.num_tokens,
      (const int **)input_info.tokens, (const double **)input_info.values, 3,
      2 // min_size=3, min_freq=2 (tunable)
  );

  // 3. Identify NEW subtrees vs CACHED subtrees
  std::vector<int> new_subtree_indices; // Indices into result.candidates
  std::vector<int> new_cache_slots;     // Where they will go in VRAM

  for (int i = 0; i < result.num_candidates; i++) {
    uint64_t hash = result.candidates[i].hash;
    if (ctx->hash_to_idx.find(hash) == ctx->hash_to_idx.end()) {
      // New subtree
      if (ctx->num_cached >= ctx->capacity) {
        std::cerr << "GPU Cache Full! Skipping subtree caching." << std::endl;
        continue; // Skip caching if full (will be eval'd inline or we need
                  // eviction policy)
      }
      int slot = ctx->num_cached++;
      ctx->hash_to_idx[hash] = slot;
      new_subtree_indices.push_back(i);
      new_cache_slots.push_back(slot);
    }
  }

  // 4. Transfer Data (X) - Once per batch
  float *h_X = flatten_X_row_major(all_vars[0], num_vars, num_dps);
  float *d_X;
  CUDA_CHECK(cudaMalloc(&d_X, num_dps * (num_vars + 1) * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_X, h_X, num_dps * (num_vars + 1) * sizeof(float),
                        cudaMemcpyHostToDevice));
  delete[] h_X;

  // 5. Stage 1: Evaluate NEW Subtrees
  if (!new_subtree_indices.empty()) {
    int num_new = new_subtree_indices.size();

    // Pack tokens/values for new subtrees
    std::vector<int> h_sub_tokens;
    std::vector<float> h_sub_values;
    std::vector<int> h_sub_offsets;
    std::vector<int> h_sub_lengths;

    for (int idx : new_subtree_indices) {
      SubtreeCandidate &cand = result.candidates[idx];
      h_sub_offsets.push_back(h_sub_tokens.size());
      h_sub_lengths.push_back(cand.size);

      // Copy tokens from the source expression
      int *src_toks = input_info.tokens[cand.expr_id];
      double *src_vals = input_info.values[cand.expr_id];

      for (int k = 0; k < cand.size; k++) {
        h_sub_tokens.push_back(src_toks[cand.start_idx + k]);
        h_sub_values.push_back((float)src_vals[cand.start_idx + k]);
      }
    }

    // Alloc & Transfer
    int *d_sub_tokens;
    float *d_sub_values;
    int *d_sub_offsets;
    int *d_sub_lengths;
    int *d_sub_slots;
    CUDA_CHECK(cudaMalloc(&d_sub_tokens, h_sub_tokens.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sub_values, h_sub_values.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sub_offsets, num_new * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sub_lengths, num_new * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sub_slots, num_new * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_sub_tokens, h_sub_tokens.data(),
                          h_sub_tokens.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sub_values, h_sub_values.data(),
                          h_sub_values.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sub_offsets, h_sub_offsets.data(),
                          num_new * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sub_lengths, h_sub_lengths.data(),
                          num_new * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sub_slots, new_cache_slots.data(),
                          num_new * sizeof(int), cudaMemcpyHostToDevice));

    // Launch Stage 1 Kernel
    int threads = 128;
    int blocks_x = (num_dps + threads - 1) / threads;
    dim3 grid(blocks_x, num_new);

    eval_subtrees_kernel<<<grid, threads>>>(
        d_sub_tokens, d_sub_values, d_sub_offsets, d_sub_lengths, d_X,
        ctx->d_results, d_sub_slots, num_new, num_vars, num_dps);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Cleanup Stage 1 temps
    CUDA_CHECK(cudaFree(d_sub_tokens));
    CUDA_CHECK(cudaFree(d_sub_values));
    CUDA_CHECK(cudaFree(d_sub_offsets));
    CUDA_CHECK(cudaFree(d_sub_lengths));
    CUDA_CHECK(cudaFree(d_sub_slots));
  }

  // 6. Stage 2: Evaluate Skeletons
  {
    // Build Skeletons (CPU)
    std::vector<int> h_skel_tokens;
    std::vector<float> h_skel_values;
    std::vector<int> h_skel_offsets;
    std::vector<int> h_skel_lengths;

    // We need to map (expr_id, start_idx, size) -> cache_slot
    // The 'result' object has this info implicitly via candidates, but we need
    // to map it back to the expressions. A simpler way: For each expression,
    // greedily replace subtrees with REF tokens. Since we have
    // 'result.candidates', we can build a map: expr_id -> list of (start, size,
    // hash)

    std::vector<std::vector<SubtreeCandidate>> expr_candidates(num_exprs);
    for (int i = 0; i < result.num_candidates; i++) {
      // Only use if it was cached (it should be, unless cache full)
      if (ctx->hash_to_idx.count(result.candidates[i].hash)) {
        expr_candidates[result.candidates[i].expr_id].push_back(
            result.candidates[i]);
      }
    }

    for (int i = 0; i < num_exprs; i++) {
      h_skel_offsets.push_back(h_skel_tokens.size());

      // Sort candidates by start_idx to process linearly
      std::sort(expr_candidates[i].begin(), expr_candidates[i].end(),
                [](const SubtreeCandidate &a, const SubtreeCandidate &b) {
                  return a.start_idx < b.start_idx;
                });

      int *toks = input_info.tokens[i];
      double *vals = input_info.values[i];
      int len = input_info.num_tokens[i];
      int curr = 0;
      int cand_idx = 0;

      while (curr < len) {
        // Check if current position starts a cached subtree
        bool replaced = false;
        if (cand_idx < expr_candidates[i].size()) {
          const auto &cand = expr_candidates[i][cand_idx];
          if (curr == cand.start_idx) {
            // Replace with REF
            h_skel_tokens.push_back(OP_REF);
            h_skel_values.push_back((float)ctx->hash_to_idx[cand.hash]);

            curr += cand.size;
            cand_idx++;

            // Skip overlapping candidates
            while (cand_idx < expr_candidates[i].size() &&
                   expr_candidates[i][cand_idx].start_idx < curr) {
              cand_idx++;
            }
            replaced = true;
          }
        }

        if (!replaced) {
          h_skel_tokens.push_back(toks[curr]);
          h_skel_values.push_back((float)vals[curr]);
          curr++;
        }
      }
      h_skel_lengths.push_back(h_skel_tokens.size() - h_skel_offsets.back());
    }

    // Alloc & Transfer Skeletons
    int *d_skel_tokens;
    float *d_skel_values;
    int *d_skel_offsets;
    int *d_skel_lengths;
    float *d_preds;
    CUDA_CHECK(cudaMalloc(&d_skel_tokens, h_skel_tokens.size() * sizeof(int)));
    CUDA_CHECK(
        cudaMalloc(&d_skel_values, h_skel_values.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_skel_offsets, num_exprs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_skel_lengths, num_exprs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_preds, num_exprs * num_dps * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_skel_tokens, h_skel_tokens.data(),
                          h_skel_tokens.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_skel_values, h_skel_values.data(),
                          h_skel_values.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_skel_offsets, h_skel_offsets.data(),
                          num_exprs * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_skel_lengths, h_skel_lengths.data(),
                          num_exprs * sizeof(int), cudaMemcpyHostToDevice));

    // Launch Stage 2 Kernel
    int threads = 128;
    int blocks_x = (num_dps + threads - 1) / threads;
    dim3 grid(blocks_x, num_exprs);

    eval_skeletons_kernel<<<grid, threads>>>(
        d_skel_tokens, d_skel_values, d_skel_offsets, d_skel_lengths, d_X,
        ctx->d_results, d_preds, num_exprs, num_vars, num_dps);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy Results Back
    float *h_preds = new float[num_exprs * num_dps];
    CUDA_CHECK(cudaMemcpy(h_preds, d_preds, num_exprs * num_dps * sizeof(float),
                          cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_exprs; i++) {
      for (int j = 0; j < num_dps; j++) {
        all_predictions[i][j] = (double)h_preds[i * num_dps + j];
      }
    }

    // Cleanup Stage 2
    delete[] h_preds;
    CUDA_CHECK(cudaFree(d_skel_tokens));
    CUDA_CHECK(cudaFree(d_skel_values));
    CUDA_CHECK(cudaFree(d_skel_offsets));
    CUDA_CHECK(cudaFree(d_skel_lengths));
    CUDA_CHECK(cudaFree(d_preds));
  }

  // Cleanup Common
  CUDA_CHECK(cudaFree(d_X));
  free_subtree_detection_result(result);
}
