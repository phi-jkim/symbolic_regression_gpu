#include "../utils/detect.h"
#include "../utils/utils.h"
#include <cuda_runtime.h>
#include <iostream>
#include <unordered_map>
#include <vector>

// =============================================================================
// GPU Subtree Evaluator (Stateful, No Batching)
// - Evaluates all datapoints in one shot (no dp batching)
// - Maintains a persistent GPU context across generations via evaluate_evolution_benchmark
// =============================================================================

#define OP_REF -2
#define MAX_CACHED_SUBTREES 10000
#define MIN_SUBTREE_SIZE 3

#define MAX_CACHED_PER_THREAD ((MAX_STACK_SIZE + MIN_SUBTREE_SIZE - 1) / MIN_SUBTREE_SIZE)

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// Persistent GPU subtree cache context (float on device)
struct GPUSubtreeStateContext {
  std::unordered_map<uint64_t, int> hash_to_idx; // subtree hash -> cache slot
  float *d_results = nullptr;                   // [capacity][total_dps]
  int num_cached = 0;
  int capacity = MAX_CACHED_SUBTREES;
  int allocated_dps = 0;                        // how many dps d_results is sized for

  ~GPUSubtreeStateContext() {
    if (d_results) CUDA_CHECK(cudaFree(d_results));
  }

  void ensure_capacity(int total_dps) {
    if (d_results == nullptr || total_dps > allocated_dps) {
      if (d_results) CUDA_CHECK(cudaFree(d_results));
      allocated_dps = (int)(total_dps * 1.2); // small headroom
      size_t size = (size_t)capacity * (size_t)allocated_dps * sizeof(float);
      CUDA_CHECK(cudaMalloc(&d_results, size));
    }
  }
};

// =============================================================================
// Device helpers
// =============================================================================

__device__ float eval_op_gpu_state(int op, float val1, float val2) {
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
  case 8: {
    float denom = fabsf(val2) <= DELTA ? (val2 < 0.0f ? -DELTA : DELTA) : val2;
    return val1 / denom;
  }
  case 9: return (val1 == 0.0f && val2 == 0.0f) ? 0.0f : powf(fabsf(val1), val2);
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
  case 22: return (val1 == 0.0f) ? -MAX_VAL : logf(fabsf(val1));
  case 23: {
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

#define STACK_PUSH(v) (stk[sp++] = (v))
#define STACK_POP()   (stk[--sp])

#ifndef MAX_NUM_FEATURES
#define MAX_NUM_FEATURES 16
#endif

#ifndef MAX_FEATURES
#define MAX_FEATURES MAX_NUM_FEATURES
#endif

// =============================================================================
// Kernels (stateful variant – no dp batching, but still 2D grid over dps/subtrees)
// =============================================================================

// Stage 1: evaluate NEW subtrees into ctx->d_results (all datapoints)
__global__ void eval_subtrees_kernel_state(
    const int *d_tokens,
    const float *d_values,
    const int *d_offsets,
    const int *d_lengths,
    const float *d_X_col_major, // [num_vars+1][total_dps]
    float *d_results,           // [capacity][total_dps]
    const int *d_cache_indices,
    int num_new_subtrees,
    int num_vars,
    int total_dps,
    int subtrees_per_block) {

  int block_sub_start = blockIdx.x * subtrees_per_block;
  int block_sub_end = min(block_sub_start + subtrees_per_block, num_new_subtrees);
  int dp_start = blockIdx.y * blockDim.x;
  int dp_idx   = dp_start + threadIdx.x;

  if (dp_idx >= total_dps) return;

  // Cache a small slice of X in registers
//   float x_cache[MAX_STACK];
  // int limit_vars = (num_vars + 1 < MAX_NUM_FEATURES) ? num_vars + 1 : MAX_NUM_FEATURES;

//   for (int v = 0; v < limit_vars; v++) {
//     x_cache[v] = d_X_col_major[v * total_dps + dp_idx];
//   }

  for (int sub_idx = block_sub_start; sub_idx < block_sub_end; sub_idx++) {
    int offset    = d_offsets[sub_idx];
    int len       = d_lengths[sub_idx];
    int cache_idx = d_cache_indices[sub_idx];

    float stk[MAX_STACK_SIZE];
    int sp = 0;

    for (int i = len - 1; i >= 0; i--) {
      int tok = d_tokens[offset + i];
      if (tok > 0) {
        float v1 = STACK_POP();
        float v2 = 0.0f;
        if (tok < 10) v2 = STACK_POP();
        STACK_PUSH(eval_op_gpu_state(tok, v1, v2));
      } else if (tok == 0) {
        STACK_PUSH(d_values[offset + i]);
      } else if (tok == -1) {
        int var_idx = (int)d_values[offset + i];
        // if (var_idx < limit_vars)
        //   STACK_PUSH(x_cache[var_idx]);
        // else
        STACK_PUSH(d_X_col_major[var_idx * total_dps + dp_idx]);
      }
    }

    d_results[cache_idx * total_dps + dp_idx] = stk[0];
  }
}

// // Stage 2: evaluate skeleton expressions using cached subtree results
// __global__ void eval_skeletons_kernel_state(
//     const int *d_tokens,
//     const float *d_values,
//     const int *d_offsets,
//     const int *d_lengths,
//     const float *d_X_col_major,
//     const float *d_cache_results,
//     float *d_preds,
//     int num_exprs,
//     int num_vars,
//     int total_dps,
//     int exprs_per_block) {

//   int block_expr_start = blockIdx.x * exprs_per_block;
//   int block_expr_end   = min(block_expr_start + exprs_per_block, num_exprs);
//   int dp_start         = blockIdx.y * blockDim.x;
//   int dp_idx           = dp_start + threadIdx.x;

//   if (dp_idx >= total_dps) return;
//   extern __shared__ float cached_results_shared[];
//   float *cached_results = &cached_results_shared[threadIdx.x * MAX_CACHED_PER_THREAD];

//   for (int expr_idx = block_expr_start; expr_idx < block_expr_end; expr_idx++) {
//     int offset = d_offsets[expr_idx];
//     int len    = d_lengths[expr_idx];

//     float stk[MAX_STACK_SIZE];
//     int sp = 0;
//     int ref_count = 0;

//     for (int i = len - 1; i >= 0; i--) {
//       int tok = d_tokens[offset + i];
//       if (tok == OP_REF) {
//         int cache_idx = (int)d_values[offset + i];
//         if (ref_count < MAX_CACHED_PER_THREAD) {
//           cached_results[ref_count++] = d_cache_results[cache_idx * total_dps + dp_idx];
//         }
//       }
//     }

//     int ref_pos = 0;

//     for (int i = len - 1; i >= 0; i--) {
//       int tok = d_tokens[offset + i];

//       if (tok == OP_REF) {
//         // int cache_idx = (int)d_values[offset + i];
//         // STACK_PUSH(d_cache_results[cache_idx * total_dps + dp_idx]);
//         STACK_PUSH(cached_results[ref_pos++]);
//       } else if (tok > 0) {
//         float v1 = STACK_POP();
//         float v2 = 0.0f;
//         if (tok < 10) v2 = STACK_POP();
//         STACK_PUSH(eval_op_gpu_state(tok, v1, v2));
//       } else if (tok == 0) {
//         STACK_PUSH(d_values[offset + i]);
//       } else if (tok == -1) {
//         int var_idx = (int)d_values[offset + i];
//         // if (var_idx < limit_vars)
//         //   STACK_PUSH(x_cache[var_idx]);
//         // else
//         STACK_PUSH(d_X_col_major[var_idx * total_dps + dp_idx]);
//       }
//     }

//     d_preds[expr_idx * total_dps + dp_idx] = stk[0];
//   }
// }

// Stage 2: evaluate skeleton expressions using cached subtree results
// d_ref_slots layout: [expr_idx * MAX_CACHED_PER_THREAD + ref_pos] = cache slot ID,
// where ref_pos enumerates OP_REF tokens in that expression when scanning tokens
// right-to-left (ref_pos = 0 is the first OP_REF seen from the right).
// d_ref_counts[expr_idx] stores how many valid entries exist for that expr.
__global__ void eval_skeletons_kernel_state(
    const int *d_tokens,
    const float *d_values,
    const int *d_offsets,
    const int *d_lengths,
    const float *d_X_col_major,
    const int *d_ref_slots,
    const int *d_ref_counts,
    const float *d_cache_results,
    float *d_preds,
    int num_exprs,
    int num_vars,
    int total_dps,
    int exprs_per_block) {

  int block_expr_start = blockIdx.x * exprs_per_block;
  int block_expr_end   = min(block_expr_start + exprs_per_block, num_exprs);
  int dp_start         = blockIdx.y * blockDim.x;
  int dp_idx           = dp_start + threadIdx.x;

  if (dp_idx >= total_dps) return;

  // extern __shared__ float cached_results_shared[];
  // float *thread_cache = &cached_results_shared[threadIdx.x * MAX_CACHED_PER_THREAD];

  extern __shared__ float x_cache[];
  // float *cached_results = thread_cache;
  int limit_vars = (num_vars < MAX_FEATURES) ? (num_vars) : MAX_FEATURES;
  for (int v = 0; v < limit_vars; ++v) {
    x_cache[v * blockDim.x + threadIdx.x] = d_X_col_major[v * total_dps + dp_idx];
  }

  for (int expr_idx = block_expr_start; expr_idx < block_expr_end; expr_idx++) {
    int offset = d_offsets[expr_idx];
    int len    = d_lengths[expr_idx];

    float stk[MAX_STACK_SIZE];
    int sp = 0;
    // int base = expr_idx * MAX_CACHED_PER_THREAD;
    // int ref_count = d_ref_counts[expr_idx];
    // for (int r = 0; r < ref_count && r < MAX_CACHED_PER_THREAD; ++r) {
    //   int cache_idx = d_ref_slots[base + r];
    //   cached_results[r] = d_cache_results[cache_idx * total_dps + dp_idx];
    // }
    // float *cached_results = thread_cache;
    // // Preload cached subtree results for this expression using precomputed
    // // cache slot indices (right-to-left ordering in d_ref_slots).
    // int base = expr_idx * MAX_CACHED_PER_THREAD;
    // int ref_count = d_ref_counts[expr_idx];
    // #pragma unroll 
    // for (int r = 0; r < ref_count; ++r) {
    //   int cache_idx = d_ref_slots[base + r];
    //   cached_results[r] = d_cache_results[cache_idx * total_dps + dp_idx];
    // }

    int ref_pos = 0;

    for (int i = len - 1; i >= 0; i--) {
      int tok = d_tokens[offset + i];

      if (tok == OP_REF) {
        int cache_idx = (int)d_values[offset + i];
        STACK_PUSH(d_cache_results[cache_idx * total_dps + dp_idx]);
        // STACK_PUSH(cached_results[ref_pos++]);
      } else if (tok > 0) {
        float v1 = STACK_POP();
        float v2 = 0.0f;
        if (tok < 10) v2 = STACK_POP();
        STACK_PUSH(eval_op_gpu_state(tok, v1, v2));
      } else if (tok == 0) {
        STACK_PUSH(d_values[offset + i]);
      } else if (tok == -1) {
        int var_idx = (int)d_values[offset + i];
        if (var_idx >= 0 && var_idx < limit_vars) {
          STACK_PUSH(x_cache[var_idx * blockDim.x + threadIdx.x]);
        } else {
          STACK_PUSH(d_X_col_major[var_idx * total_dps + dp_idx]);
        }
        // if (var_idx < limit_vars)
        //   STACK_PUSH(x_cache[var_idx]);
        // else
        // STACK_PUSH(d_X_col_major[var_idx * total_dps + dp_idx]);
      }
    }

    d_preds[expr_idx * total_dps + dp_idx] = stk[0];
  }
}

// =============================================================================
// Host helpers
// =============================================================================

// Flatten X for the entire dataset (column-major by feature)
static float *flatten_X_col_major_full(double **vars, int num_vars, int total_dps) {
  float *flat = new float[(num_vars + 1) * total_dps];
  for (int v = 0; v <= num_vars; v++) {
    for (int dp = 0; dp < total_dps; dp++) {
      flat[v * total_dps + dp] = (float)vars[v][dp];
    }
  }
  return flat;
}

// Hash is still computed in double on CPU for consistency with CPU subtree detection
static uint64_t compute_subtree_hash_gpu(const int *tokens, const double *values, int len) {
  uint64_t h = 14695981039346656037ULL;
  const uint64_t prime = 1099511628211ULL;
  for (int i = 0; i < len; i++) {
    h ^= (uint64_t)tokens[i];
    h *= prime;
    uint64_t v_bits;
    double v_val = values[i];
    memcpy(&v_bits, &v_val, sizeof(double));
    h ^= v_bits;
    h *= prime;
  }
  return h;
}

// Local helpers to compute subtree size (prefix notation) on the host
static int simple_op_arity_state(int token) {
  if (token < 10) return 2;  // binary operators 1-9
  return 1;                  // unary operators >= 10
}

static int compute_subtree_size_state(const int *tokens, int start_idx, int total_tokens) {
  if (start_idx >= total_tokens) return 0;

  int token = tokens[start_idx];
  if (token <= 0) {
    // Variable or constant
    return 1;
  }

  int arity = simple_op_arity_state(token);
  int size  = 1;  // count the operator itself
  int pos   = start_idx + 1;

  for (int i = 0; i < arity && pos < total_tokens; ++i) {
    int child_size = compute_subtree_size_state(tokens, pos, total_tokens);
    size += child_size;
    pos  += child_size;
  }
  return size;
}

// Mark cached subtrees from persistent GPU state into expr_sub_hints.
// Convention:
//   hints[pos] >= 0 : root of cached subtree, value is CACHE SLOT ID
//   hints[pos] == -2: inside cached subtree
//   hints[pos] == -1: no cached subtree
static void mark_cached_subtrees_from_state(
    InputInfo &input_info,
    GPUSubtreeStateContext &ctx,
    SubtreeDetectionResult &result) {

  int num_exprs = input_info.num_exprs;
  const int min_size_state = MIN_SUBTREE_SIZE;  // same as detect_common_subtrees default

  for (int expr = 0; expr < num_exprs; ++expr) {
    int    len   = input_info.num_tokens[expr];
    int   *toks  = input_info.tokens[expr];
    double *vals = input_info.values[expr];
    int   *hints = result.expr_sub_hints[expr];

    // Reset hints so we control all cached subtree marking here
    for (int i = 0; i < len; ++i) hints[i] = -1;

    for (int pos = 0; pos < len; ++pos) {
      if (hints[pos] != -1) continue;  // already claimed by a larger subtree

      int size = compute_subtree_size_state(toks, pos, len);
      if (size < min_size_state) continue;

      uint64_t h = compute_subtree_hash_gpu(toks + pos, vals + pos, size);
      auto it    = ctx.hash_to_idx.find(h);
      if (it == ctx.hash_to_idx.end()) continue;  // not in GPU cache

      int slot = it->second;

      // Check for overlap (defensive, though we already skip if hints[pos]!=-1)
      bool overlap = false;
      for (int k = 0; k < size; ++k) {
        if (hints[pos + k] != -1) { overlap = true; break; }
      }
      if (overlap) continue;

      // Mark root and body
      hints[pos] = slot;      // cache-slot ID
      for (int k = 1; k < size && pos + k < len; ++k) {
        hints[pos + k] = -2;  // inside subtree
      }
    }
  }
}

// =============================================================================
// Core stateful GPU subtree evaluator (no dp batching)
// =============================================================================

void eval_gpu_subtree_batch_state(
    InputInfo &input_info,
    double ***all_vars,
    double **all_predictions,
    GPUSubtreeStateContext &ctx) {

  int num_exprs = input_info.num_exprs;
  if (num_exprs == 0) return;

  int num_vars  = input_info.num_vars[0];
  int total_dps = input_info.num_dps[0];

  // Assume shared data (same dataset for all expressions), like CPU stateful path
  ctx.ensure_capacity(total_dps);

  // Step 1: Detect common subtrees for this generation (structure only)
  SubtreeDetectionResult result = detect_common_subtrees(
      num_exprs,
      num_vars,
      input_info.num_tokens,
      (const int **)input_info.tokens,
      (const double **)input_info.values,
      MIN_SUBTREE_SIZE,
      2);

  // Step 2: Map detected subtrees to persistent cache slots
  std::vector<int> new_subtree_indices;
  std::vector<int> new_cache_slots;
  std::vector<int> subtree_id_to_slot(result.num_subs, -1);

  for (int i = 0; i < result.num_subs; i++) {
    uint64_t h = compute_subtree_hash_gpu(result.sub_tokens[i],
                                          result.sub_values[i],
                                          result.num_sub_tokens[i]);
    auto it = ctx.hash_to_idx.find(h);
    if (it == ctx.hash_to_idx.end()) {
      if (ctx.num_cached >= ctx.capacity) {
        // Capacity reached; skip adding new subtrees
        continue;
      }
      int slot = ctx.num_cached++;
      ctx.hash_to_idx[h] = slot;
      new_subtree_indices.push_back(i);
      new_cache_slots.push_back(slot);
      subtree_id_to_slot[i] = slot;
    } else {
      subtree_id_to_slot[i] = it->second;
    }
  }

  std::cout << "[gpu_subtree_state] New subtrees this gen: "
            << new_subtree_indices.size() << std::endl;

  // Step 2b: Mark any subtree (including those from previous generations)
  // whose hash is present in the persistent GPU cache. This populates
  // expr_sub_hints with cache-slot IDs at subtree roots.
  mark_cached_subtrees_from_state(input_info, ctx, result);

  // Step 3: Prepare subtree structures for NEW subtrees (tokens/values)
  int *d_sub_tokens = nullptr, *d_sub_offsets = nullptr, *d_sub_lengths = nullptr, *d_sub_slots = nullptr;
  float *d_sub_values = nullptr;
  int num_new = (int)new_subtree_indices.size();

  if (num_new > 0) {
    std::vector<int>   h_sub_tokens;
    std::vector<float> h_sub_values;
    std::vector<int>   h_sub_offsets;
    std::vector<int>   h_sub_lengths;

    for (int idx : new_subtree_indices) {
      h_sub_offsets.push_back((int)h_sub_tokens.size());
      h_sub_lengths.push_back(result.num_sub_tokens[idx]);

      int    *src_toks = result.sub_tokens[idx];
      double *src_vals = result.sub_values[idx];

      for (int k = 0; k < result.num_sub_tokens[idx]; k++) {
        h_sub_tokens.push_back(src_toks[k]);
        h_sub_values.push_back((float)src_vals[k]);
      }
    }

    CUDA_CHECK(cudaMalloc(&d_sub_tokens,  h_sub_tokens.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sub_values,  h_sub_values.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sub_offsets, num_new * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sub_lengths, num_new * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sub_slots,   num_new * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_sub_tokens,  h_sub_tokens.data(),  h_sub_tokens.size() * sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sub_values,  h_sub_values.data(),  h_sub_values.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sub_offsets, h_sub_offsets.data(), num_new * sizeof(int),               cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sub_lengths, h_sub_lengths.data(), num_new * sizeof(int),               cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sub_slots,   new_cache_slots.data(), num_new * sizeof(int),             cudaMemcpyHostToDevice));
  }

  // Step 4: Prepare skeleton expressions (with OP_REF placeholders)
  int *d_skel_tokens = nullptr, *d_skel_offsets = nullptr, *d_skel_lengths = nullptr,
      *d_skel_ref_slots = nullptr, *d_skel_ref_counts = nullptr;
  float *d_skel_values = nullptr;
  float *d_preds = nullptr;

  std::vector<int>   h_skel_tokens;
  std::vector<float> h_skel_values;
  std::vector<int>   h_skel_offsets;
  std::vector<int>   h_skel_lengths;

  for (int i = 0; i < num_exprs; i++) {
    h_skel_offsets.push_back((int)h_skel_tokens.size());

    int    *toks  = input_info.tokens[i];
    double *vals  = input_info.values[i];
    int     len   = input_info.num_tokens[i];
    // printf("len: %d\n", len);
    int    *hints = result.expr_sub_hints[i];

    for (int k = 0; k < len; k++) {
      int hint = hints[k];

      if (hint >= 0) {
        // hint is a cache-slot ID; compute subtree size on the fly
        int sub_size = compute_subtree_size_state(toks, k, len);
        h_skel_tokens.push_back(OP_REF);
        h_skel_values.push_back((float)hint); // cache slot stored as float
        k += sub_size - 1; // skip over subtree body
      } else if (hint == -2) {
        // Inside a cached subtree; body is skipped by the root case above
        continue;
      } else {
        h_skel_tokens.push_back(toks[k]);
        h_skel_values.push_back((float)vals[k]);
      }
    }
    h_skel_lengths.push_back((int)h_skel_tokens.size() - h_skel_offsets.back());
  }

  // Precompute cache slot indices and counts per expression for skeleton OP_REFs.
  // For each expr i:
  //   - h_skel_ref_slots[i * MAX_CACHED_PER_THREAD + ref_pos] stores the cache slot ID
  //     of the ref_pos-th OP_REF encountered when scanning that expression's
  //     skeleton tokens from right to left.
  //   - h_skel_ref_counts[i] stores how many valid entries exist for that expr.
  std::vector<int> h_skel_ref_slots((size_t)num_exprs * (size_t)MAX_CACHED_PER_THREAD, -1);
  std::vector<int> h_skel_ref_counts(num_exprs, 0);
  for (int i = 0; i < num_exprs; ++i) {
    int start = h_skel_offsets[i];
    int len   = h_skel_lengths[i];
    int ref_pos = 0;
    for (int idx = start + len - 1;
         idx >= start && ref_pos < MAX_CACHED_PER_THREAD;
         --idx) {
      if (h_skel_tokens[idx] == OP_REF) {
        int slot = (int)h_skel_values[idx];
        h_skel_ref_slots[(size_t)i * (size_t)MAX_CACHED_PER_THREAD + (size_t)ref_pos] = slot;
        ++ref_pos;
      }
    }
    h_skel_ref_counts[i] = ref_pos;
  }

  CUDA_CHECK(cudaMalloc(&d_skel_tokens,  h_skel_tokens.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_skel_values,  h_skel_values.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_skel_offsets, num_exprs * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_skel_lengths, num_exprs * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_skel_ref_slots,
                        (size_t)num_exprs * (size_t)MAX_CACHED_PER_THREAD * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_skel_ref_counts, num_exprs * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_preds,        (size_t)num_exprs * (size_t)total_dps * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_skel_tokens,  h_skel_tokens.data(),  h_skel_tokens.size() * sizeof(int),   cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_skel_values,  h_skel_values.data(),  h_skel_values.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_skel_offsets, h_skel_offsets.data(), num_exprs * sizeof(int),              cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_skel_lengths, h_skel_lengths.data(), num_exprs * sizeof(int),              cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_skel_ref_slots, h_skel_ref_slots.data(),
                        (size_t)num_exprs * (size_t)MAX_CACHED_PER_THREAD * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_skel_ref_counts, h_skel_ref_counts.data(),
                        num_exprs * sizeof(int),
                        cudaMemcpyHostToDevice));

  // Step 5: Upload full dataset X once (no batching over datapoints)
  float *d_X = nullptr;
  CUDA_CHECK(cudaMalloc(&d_X, (size_t)(num_vars + 1) * (size_t)total_dps * sizeof(float)));

  double h2d_ms = 0.0;
  double d2h_ms = 0.0;

  TimePoint t_h2d = measure_clock();
  float *h_X = flatten_X_col_major_full(all_vars[0], num_vars, total_dps);
  CUDA_CHECK(cudaMemcpy(d_X, h_X, (size_t)(num_vars + 1) * (size_t)total_dps * sizeof(float), cudaMemcpyHostToDevice));
  delete[] h_X;
  h2d_ms = clock_to_ms(t_h2d, measure_clock());

  // Step 6: Launch kernels over all datapoints at once
  int threads  = 128;
  int blocks_y = (total_dps + threads - 1) / threads;
  // const int max_features_sh = num_vars;
  printf("num_vars: %d\n", num_vars);
  size_t shared_bytes = (size_t)num_vars *
                        (size_t)threads *
                        sizeof(float);

  // Use CUDA events to time subtree kernel and skeleton kernel separately
  cudaEvent_t ev_start, ev_mid, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_mid));
  CUDA_CHECK(cudaEventCreate(&ev_stop));
  CUDA_CHECK(cudaEventRecord(ev_start));

  // 6a. New subtrees
  if (num_new > 0) {
    int subtrees_per_block = 4;
    int blocks_x = (num_new + subtrees_per_block - 1) / subtrees_per_block;
    dim3 grid(blocks_x, blocks_y);
    eval_subtrees_kernel_state<<<grid, threads>>>(
        d_sub_tokens, d_sub_values, d_sub_offsets, d_sub_lengths,
        d_X,
        ctx.d_results,
        d_sub_slots,
        num_new,
        num_vars,
        total_dps,
        subtrees_per_block);
  }

  // Mark end of subtree kernel region
  CUDA_CHECK(cudaEventRecord(ev_mid));

  // 6b. Skeleton expressions
  {
    int exprs_per_block = 4;
    int blocks_x = (num_exprs + exprs_per_block - 1) / exprs_per_block;
    dim3 grid(blocks_x, blocks_y);
    eval_skeletons_kernel_state<<<grid, threads, shared_bytes>>>(
        d_skel_tokens, d_skel_values, d_skel_offsets, d_skel_lengths,
        d_X,
        d_skel_ref_slots,
        d_skel_ref_counts,
        ctx.d_results,
        d_preds,
        num_exprs,
        num_vars,
        total_dps,
        exprs_per_block);
    
  }

  CUDA_CHECK(cudaEventRecord(ev_stop));
  CUDA_CHECK(cudaEventSynchronize(ev_stop));

  float subtree_ms = 0.0f;
  float total_kernel_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&subtree_ms, ev_start, ev_mid));
  CUDA_CHECK(cudaEventElapsedTime(&total_kernel_ms, ev_start, ev_stop));
  float skeleton_ms = total_kernel_ms - subtree_ms;

  std::cout << "[gpu_subtree_state] Subtree Kernel: " << subtree_ms
            << " ms, Skeleton Kernel: " << skeleton_ms
            << " ms, Combined Kernel: " << total_kernel_ms << " ms" << std::endl;

  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_mid));
  CUDA_CHECK(cudaEventDestroy(ev_stop));

  // Step 7: Copy predictions back and convert to double
  std::vector<float> h_preds((size_t)num_exprs * (size_t)total_dps);
  TimePoint t_d2h = measure_clock();
  CUDA_CHECK(cudaMemcpy(h_preds.data(), d_preds,
                        (size_t)num_exprs * (size_t)total_dps * sizeof(float),
                        cudaMemcpyDeviceToHost));
  d2h_ms = clock_to_ms(t_d2h, measure_clock());

  double total_gpu_ms = h2d_ms + (double)total_kernel_ms + d2h_ms;
  std::cout << "[gpu_subtree_state] H2D: " << h2d_ms
            << " ms, Kernel: " << total_kernel_ms
            << " ms, D2H: " << d2h_ms
            << " ms, Total GPU: " << total_gpu_ms << " ms" << std::endl;

  for (int expr = 0; expr < num_exprs; expr++) {
    double *pred = all_predictions[expr];
    for (int dp = 0; dp < total_dps; dp++) {
      pred[dp] = (double)h_preds[(size_t)expr * (size_t)total_dps + dp];
    }
  }

  // Cleanup temporary GPU buffers (but keep ctx.d_results alive across generations)
  CUDA_CHECK(cudaFree(d_X));
  CUDA_CHECK(cudaFree(d_skel_tokens));
  CUDA_CHECK(cudaFree(d_skel_values));
  CUDA_CHECK(cudaFree(d_skel_offsets));
  CUDA_CHECK(cudaFree(d_skel_lengths));
  CUDA_CHECK(cudaFree(d_skel_ref_slots));
  CUDA_CHECK(cudaFree(d_skel_ref_counts));
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

// =============================================================================
// Evolution benchmark helper (stateful GPU subtree)
// =============================================================================

int run_evolution_benchmark_gpu_subtree_state(int start_gen, int end_gen, const std::string &data_dir) {
  GPUSubtreeStateContext ctx;

  // Adapter for evaluate_evolution_benchmark
  auto eval_cb = [](InputInfo &info, double ***vars, double **preds, void *state) {
    auto *c = reinterpret_cast<GPUSubtreeStateContext *>(state);
    eval_gpu_subtree_batch_state(info, vars, preds, *c);
  };

  return evaluate_evolution_benchmark(start_gen, end_gen, data_dir, eval_cb, &ctx);
}
