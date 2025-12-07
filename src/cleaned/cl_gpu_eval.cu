#include "../utils/utils.h"
#include "../utils/detect.h"
#include "../utils/opcodes.h"
#include <cuda_runtime.h>
#include <vector>
#include <unordered_map>
#include <iostream>

#define OP_REF -2
#define MAX_CACHED_SUBTREES 10000

#ifndef MAX_NUM_FEATURES
#define MAX_NUM_FEATURES 32
#endif

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

// Device helper: Evaluate op
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

#define STACK_PUSH(v) do { stk[sp] = (v); sp++; } while (0)
#define STACK_POP()   (stk[--sp])

// Kernel 1: Evaluate subtrees
__global__ void eval_subtrees_kernel(
    const int *d_tokens,
    const float *d_values,
    const int *d_offsets,
    const int *d_lengths,
    const float *d_X_col_major,
    float *d_results,
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

  float x_cache[MAX_NUM_FEATURES];
  int limit_vars = (num_vars + 1 < MAX_NUM_FEATURES) ? num_vars + 1 : MAX_NUM_FEATURES;

  for (int v = 0; v < limit_vars; v++) {
    x_cache[v] = d_X_col_major[v * total_dps + dp_idx];
  }

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
        if (var_idx < limit_vars)
          STACK_PUSH(x_cache[var_idx]);
        else
          STACK_PUSH(d_X_col_major[var_idx * total_dps + dp_idx]);
      }
    }
    d_results[cache_idx * total_dps + dp_idx] = stk[0];
  }
}

// Kernel 2: Evaluate skeletons AND compute squared error
// Instead of writing predictions, we write squared errors to d_mse_accum (one per dp)
// Wait, we need to reduce.
// Strategy:
// 1. Evaluate skeleton -> get pred
// 2. Compute diff = pred - y_true
// 3. Store diff*diff in a temporary buffer OR do block reduction.
// For simplicity in this "minimal" version, let's just write the squared error to a buffer
// and do a second reduction kernel (or Thrust reduction if available, but let's stick to raw CUDA).
// Actually, let's write the squared error to d_preds (reused as d_sq_errors) and then reduce.

__global__ void eval_skeletons_kernel(
    const int *d_tokens,
    const float *d_values,
    const int *d_offsets,
    const int *d_lengths,
    const float *d_X_col_major,
    const float *d_cache_results,
    float *d_sq_errors, // [num_exprs * total_dps]
    int num_exprs,
    int num_vars,
    int total_dps,
    int exprs_per_block) {

  int block_expr_start = blockIdx.x * exprs_per_block;
  int block_expr_end   = min(block_expr_start + exprs_per_block, num_exprs);
  int dp_start         = blockIdx.y * blockDim.x;
  int dp_idx           = dp_start + threadIdx.x;

  if (dp_idx >= total_dps) return;

  float x_cache[MAX_NUM_FEATURES];
  int limit_vars = (num_vars + 1 < MAX_NUM_FEATURES) ? num_vars + 1 : MAX_NUM_FEATURES;

  for (int v = 0; v < limit_vars; v++) {
    x_cache[v] = d_X_col_major[v * total_dps + dp_idx];
  }
  
  // Ground truth is at index num_vars
  float y_true = d_X_col_major[num_vars * total_dps + dp_idx];

  for (int expr_idx = block_expr_start; expr_idx < block_expr_end; expr_idx++) {
    int offset = d_offsets[expr_idx];
    int len    = d_lengths[expr_idx];

    float stk[MAX_STACK_SIZE];
    int sp = 0;

    for (int i = len - 1; i >= 0; i--) {
      int tok = d_tokens[offset + i];

      if (tok == OP_REF) {
        int cache_idx = (int)d_values[offset + i];
        STACK_PUSH(d_cache_results[cache_idx * total_dps + dp_idx]);
      } else if (tok > 0) {
        float v1 = STACK_POP();
        float v2 = 0.0f;
        if (tok < 10) v2 = STACK_POP();
        STACK_PUSH(eval_op_gpu_state(tok, v1, v2));
      } else if (tok == 0) {
        STACK_PUSH(d_values[offset + i]);
      } else if (tok == -1) {
        int var_idx = (int)d_values[offset + i];
        if (var_idx < limit_vars)
          STACK_PUSH(x_cache[var_idx]);
        else
          STACK_PUSH(d_X_col_major[var_idx * total_dps + dp_idx]);
      }
    }

    float pred = stk[0];
    float diff = pred - y_true;
    // Handle NAN/INF: if invalid, set error to 0 (or handle gracefully)
    // For now, let's just let it propagate or clamp?
    // If we want to ignore invalid dps, we need a count.
    // Let's assume valid for now for perf, or use 0 if nan.
    if (isnan(pred) || isinf(pred)) {
        d_sq_errors[expr_idx * total_dps + dp_idx] = NAN; 
    } else {
        d_sq_errors[expr_idx * total_dps + dp_idx] = diff * diff;
    }
  }
}

// Simple reduction kernel (one block per expression)
// Reduces [total_dps] -> [1]
__global__ void reduce_mse_kernel(
    const float *d_sq_errors,
    double *d_mses,
    int num_exprs,
    int total_dps) {
    
    int expr_idx = blockIdx.x;
    if (expr_idx >= num_exprs) return;

    double sum = 0.0;
    int valid_count = 0;
    
    // Grid-stride loop for reduction within the expression's data
    for (int i = threadIdx.x; i < total_dps; i += blockDim.x) {
        float val = d_sq_errors[expr_idx * total_dps + i];
        if (!isnan(val)) {
            sum += (double)val;
            valid_count++;
        }
    }

    // Block reduction
    // Using shared memory
    extern __shared__ double sdata[];
    int* scount = (int*)&sdata[blockDim.x];
    
    int tid = threadIdx.x;
    sdata[tid] = sum;
    scount[tid] = valid_count;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
            scount[tid] += scount[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (scount[0] > 0) {
            d_mses[expr_idx] = sdata[0] / scount[0];
        } else {
            d_mses[expr_idx] = NAN;
        }
    }
}


// Host helpers
static float *flatten_X_col_major_full(double **vars, int num_vars, int total_dps) {
  float *flat = new float[(num_vars + 1) * total_dps];
  for (int v = 0; v <= num_vars; v++) {
    for (int dp = 0; dp < total_dps; dp++) {
      flat[v * total_dps + dp] = (float)vars[v][dp];
    }
  }
  return flat;
}

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

static int simple_op_arity_state(int token) {
  if (token < 10) return 2;
  return 1;
}

static int compute_subtree_size_state(const int *tokens, int start_idx, int total_tokens) {
  if (start_idx >= total_tokens) return 0;
  int token = tokens[start_idx];
  if (token <= 0) return 1;
  int arity = simple_op_arity_state(token);
  int size  = 1;
  int pos   = start_idx + 1;
  for (int i = 0; i < arity && pos < total_tokens; ++i) {
    int child_size = compute_subtree_size_state(tokens, pos, total_tokens);
    size += child_size;
    pos  += child_size;
  }
  return size;
}

static void mark_cached_subtrees_from_state(
    InputInfo &input_info,
    GPUSubtreeStateContext &ctx,
    SubtreeDetectionResult &result) {

  int num_exprs = input_info.num_exprs;
  const int min_size_state = 3;

  for (int expr = 0; expr < num_exprs; ++expr) {
    int    len   = input_info.num_tokens[expr];
    int   *toks  = input_info.tokens[expr];
    double *vals = input_info.values[expr];
    int   *hints = result.expr_sub_hints[expr];

    for (int i = 0; i < len; ++i) hints[i] = -1;

    for (int pos = 0; pos < len; ++pos) {
      if (hints[pos] != -1) continue;
      int size = compute_subtree_size_state(toks, pos, len);
      if (size < min_size_state) continue;
      uint64_t h = compute_subtree_hash_gpu(toks + pos, vals + pos, size);
      auto it    = ctx.hash_to_idx.find(h);
      if (it == ctx.hash_to_idx.end()) continue;
      int slot = it->second;
      bool overlap = false;
      for (int k = 0; k < size; ++k) {
        if (hints[pos + k] != -1) { overlap = true; break; }
      }
      if (overlap) continue;
      hints[pos] = slot;
      for (int k = 1; k < size && pos + k < len; ++k) {
        hints[pos + k] = -2;
      }
    }
  }
}

// Main GPU evaluation function
void evaluate_gpu_mse(
    InputInfo &input_info,
    double ***all_vars,
    std::vector<double> &mses,
    GPUSubtreeStateContext &ctx) {

  int num_exprs = input_info.num_exprs;
  if (num_exprs == 0) return;

  int num_vars  = input_info.num_vars[0];
  int total_dps = input_info.num_dps[0];

  ctx.ensure_capacity(total_dps);

  // 1. Detect subtrees
  SubtreeDetectionResult result = detect_common_subtrees(
      num_exprs, num_vars, input_info.num_tokens,
      (const int **)input_info.tokens, (const double **)input_info.values, 3, 2);

  // 2. Update cache
  std::vector<int> new_subtree_indices;
  std::vector<int> new_cache_slots;
  for (int i = 0; i < result.num_subs; i++) {
    uint64_t h = compute_subtree_hash_gpu(result.sub_tokens[i], result.sub_values[i], result.num_sub_tokens[i]);
    if (ctx.hash_to_idx.find(h) == ctx.hash_to_idx.end()) {
      if (ctx.num_cached < ctx.capacity) {
        int slot = ctx.num_cached++;
        ctx.hash_to_idx[h] = slot;
        new_subtree_indices.push_back(i);
        new_cache_slots.push_back(slot);
      }
    }
  }
  mark_cached_subtrees_from_state(input_info, ctx, result);

  // 3. Prepare GPU buffers for new subtrees
  int num_new = (int)new_subtree_indices.size();

  if (num_new > 0) {
    std::vector<int> h_sub_tokens, h_sub_offsets, h_sub_lengths;
    std::vector<float> h_sub_values;
    for (int idx : new_subtree_indices) {
      h_sub_offsets.push_back((int)h_sub_tokens.size());
      h_sub_lengths.push_back(result.num_sub_tokens[idx]);
      for (int k = 0; k < result.num_sub_tokens[idx]; k++) {
        h_sub_tokens.push_back(result.sub_tokens[idx][k]);
        h_sub_values.push_back((float)result.sub_values[idx][k]);
      }
    }
    
    ctx.ensure_sub_buffers(h_sub_tokens.size(), h_sub_values.size(), num_new);

    CUDA_CHECK(cudaMemcpy(ctx.d_sub_tokens, h_sub_tokens.data(), h_sub_tokens.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx.d_sub_values, h_sub_values.data(), h_sub_values.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx.d_sub_offsets, h_sub_offsets.data(), num_new * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx.d_sub_lengths, h_sub_lengths.data(), num_new * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx.d_sub_slots, new_cache_slots.data(), num_new * sizeof(int), cudaMemcpyHostToDevice));
  }

  // 4. Prepare skeletons
  std::vector<int> h_skel_tokens, h_skel_offsets, h_skel_lengths;
  std::vector<float> h_skel_values;

  for (int i = 0; i < num_exprs; i++) {
    h_skel_offsets.push_back((int)h_skel_tokens.size());
    int *toks = input_info.tokens[i];
    double *vals = input_info.values[i];
    int len = input_info.num_tokens[i];
    int *hints = result.expr_sub_hints[i];

    for (int k = 0; k < len; k++) {
      int hint = hints[k];
      if (hint >= 0) {
        int sub_size = compute_subtree_size_state(toks, k, len);
        h_skel_tokens.push_back(OP_REF);
        h_skel_values.push_back((float)hint);
        k += sub_size - 1;
      } else if (hint == -2) {
        continue;
      } else {
        h_skel_tokens.push_back(toks[k]);
        h_skel_values.push_back((float)vals[k]);
      }
    }
    h_skel_lengths.push_back((int)h_skel_tokens.size() - h_skel_offsets.back());
  }

  ctx.ensure_skel_buffers(h_skel_tokens.size(), h_skel_values.size(), num_exprs, total_dps);

  CUDA_CHECK(cudaMemcpy(ctx.d_skel_tokens, h_skel_tokens.data(), h_skel_tokens.size() * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ctx.d_skel_values, h_skel_values.data(), h_skel_values.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ctx.d_skel_offsets, h_skel_offsets.data(), num_exprs * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ctx.d_skel_lengths, h_skel_lengths.data(), num_exprs * sizeof(int), cudaMemcpyHostToDevice));

  // 5. Upload X
  // Only upload if needed, but for now we upload every time to be safe (unless we track changes)
  // But we reuse the buffer.
  size_t x_size_floats = (size_t)(num_vars + 1) * (size_t)total_dps;
  ctx.ensure_X_buffer(x_size_floats);
  
  float *h_X = flatten_X_col_major_full(all_vars[0], num_vars, total_dps);
  CUDA_CHECK(cudaMemcpy(ctx.d_X, h_X, x_size_floats * sizeof(float), cudaMemcpyHostToDevice));
  delete[] h_X;

  // 6. Launch kernels
  int threads = 128;
  int blocks_y = (total_dps + threads - 1) / threads;

  if (num_new > 0) {
    int subtrees_per_block = 4;
    int blocks_x = (num_new + subtrees_per_block - 1) / subtrees_per_block;
    dim3 grid(blocks_x, blocks_y);
    eval_subtrees_kernel<<<grid, threads>>>(
        ctx.d_sub_tokens, ctx.d_sub_values, ctx.d_sub_offsets, ctx.d_sub_lengths,
        ctx.d_X, ctx.d_results, ctx.d_sub_slots, num_new, num_vars, total_dps, subtrees_per_block);
  }

  {
    int exprs_per_block = 4;
    int blocks_x = (num_exprs + exprs_per_block - 1) / exprs_per_block;
    dim3 grid(blocks_x, blocks_y);
    eval_skeletons_kernel<<<grid, threads>>>(
        ctx.d_skel_tokens, ctx.d_skel_values, ctx.d_skel_offsets, ctx.d_skel_lengths,
        ctx.d_X, ctx.d_results, ctx.d_sq_errors, num_exprs, num_vars, total_dps, exprs_per_block);
  }
  
  // 7. Reduce MSE
  // One block per expression
  int reduce_threads = 256; // Must be power of 2
  size_t shared_mem = reduce_threads * (sizeof(double) + sizeof(int));
  reduce_mse_kernel<<<num_exprs, reduce_threads, shared_mem>>>(ctx.d_sq_errors, ctx.d_mses, num_exprs, total_dps);

  // 8. Copy back MSEs
  mses.resize(num_exprs);
  CUDA_CHECK(cudaMemcpy(mses.data(), ctx.d_mses, num_exprs * sizeof(double), cudaMemcpyDeviceToHost));

  free_subtree_detection_result(result);
}

// C-style wrappers for linking with C++ main
// extern "C" {

void* create_gpu_context() {
    return new GPUSubtreeStateContext();
}

void destroy_gpu_context(void* ctx) {
    delete static_cast<GPUSubtreeStateContext*>(ctx);
}

void evaluate_gpu_mse_wrapper(InputInfo& input_info, double*** all_vars, std::vector<double>& mses, void* ctx) {
    evaluate_gpu_mse(input_info, all_vars, mses, *static_cast<GPUSubtreeStateContext*>(ctx));
}

// }
