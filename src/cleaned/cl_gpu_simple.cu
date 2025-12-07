#include "../utils/utils.h"
#include "../utils/opcodes.h"
#include "cl_gpu_common.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

#define OP_REF -2

// Device helper: Evaluate op (same as optimized version)
__device__ float eval_op_gpu_simple(int op, float val1, float val2) {
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

// Kernel: Evaluate expressions directly (no subtree caching)
__global__ void eval_simple_kernel(
    const int *d_tokens,
    const float *d_values,
    const int *d_offsets,
    const int *d_lengths,
    const float *d_X_col_major,
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

      if (tok > 0) {
        float v1 = STACK_POP();
        float v2 = 0.0f;
        if (tok < 10) v2 = STACK_POP();
        STACK_PUSH(eval_op_gpu_simple(tok, v1, v2));
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
    if (isnan(pred) || isinf(pred)) {
        d_sq_errors[expr_idx * total_dps + dp_idx] = NAN; 
    } else {
        d_sq_errors[expr_idx * total_dps + dp_idx] = diff * diff;
    }
  }
}

// Simple reduction kernel (same as optimized version)
__global__ void reduce_mse_kernel_simple(
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

// Host helper
static float *flatten_X_col_major_full_simple(double **vars, int num_vars, int total_dps) {
  float *flat = new float[(num_vars + 1) * total_dps];
  for (int v = 0; v <= num_vars; v++) {
    for (int dp = 0; dp < total_dps; dp++) {
      flat[v * total_dps + dp] = (float)vars[v][dp];
    }
  }
  return flat;
}

// Main GPU evaluation function
void evaluate_gpu_simple(
    InputInfo &input_info,
    double ***all_vars,
    std::vector<double> &mses,
    SimpleGPUContext &ctx) {

  int num_exprs = input_info.num_exprs;
  if (num_exprs == 0) return;

  int num_vars  = input_info.num_vars[0];
  int total_dps = input_info.num_dps[0];

  // Timers
  float t_h2d_tokens = 0.0f;
  float t_h2d_X = 0.0f;
  float t_kernel_eval = 0.0f;
  float t_kernel_reduce = 0.0f;
  float t_d2h = 0.0f;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // 1. Prepare buffers
  std::vector<int> h_tokens, h_offsets, h_lengths;
  std::vector<float> h_values;

  for (int i = 0; i < num_exprs; i++) {
    h_offsets.push_back((int)h_tokens.size());
    int *toks = input_info.tokens[i];
    double *vals = input_info.values[i];
    int len = input_info.num_tokens[i];

    for (int k = 0; k < len; k++) {
        h_tokens.push_back(toks[k]);
        h_values.push_back((float)vals[k]);
    }
    h_lengths.push_back(len);
  }

  ctx.ensure_buffers(h_tokens.size(), h_values.size(), num_exprs, total_dps);

  cudaEventRecord(start);
  CUDA_CHECK(cudaMemcpy(ctx.d_tokens, h_tokens.data(), h_tokens.size() * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ctx.d_values, h_values.data(), h_values.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ctx.d_offsets, h_offsets.data(), num_exprs * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ctx.d_lengths, h_lengths.data(), num_exprs * sizeof(int), cudaMemcpyHostToDevice));
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  t_h2d_tokens += ms;

  // 2. Upload X
  size_t x_size_floats = (size_t)(num_vars + 1) * (size_t)total_dps;
  ctx.ensure_X_buffer(x_size_floats);
  
  float *h_X = flatten_X_col_major_full_simple(all_vars[0], num_vars, total_dps);
  
  cudaEventRecord(start);
  CUDA_CHECK(cudaMemcpy(ctx.d_X, h_X, x_size_floats * sizeof(float), cudaMemcpyHostToDevice));
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms, start, stop);
  t_h2d_X += ms;
  
  delete[] h_X;

  // 3. Launch kernel
  int threads = 128;
  int blocks_y = (total_dps + threads - 1) / threads;
  int exprs_per_block = 4;
  int blocks_x = (num_exprs + exprs_per_block - 1) / exprs_per_block;
  dim3 grid(blocks_x, blocks_y);
  
  cudaEventRecord(start);
  eval_simple_kernel<<<grid, threads>>>(
      ctx.d_tokens, ctx.d_values, ctx.d_offsets, ctx.d_lengths,
      ctx.d_X, ctx.d_sq_errors, num_exprs, num_vars, total_dps, exprs_per_block);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms, start, stop);
  t_kernel_eval += ms;

  // 4. Reduce MSE
  int reduce_threads = 256;
  size_t shared_mem = reduce_threads * (sizeof(double) + sizeof(int));
  
  cudaEventRecord(start);
  reduce_mse_kernel_simple<<<num_exprs, reduce_threads, shared_mem>>>(ctx.d_sq_errors, ctx.d_mses, num_exprs, total_dps);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms, start, stop);
  t_kernel_reduce += ms;

  // 5. Copy back MSEs
  mses.resize(num_exprs);
  
  cudaEventRecord(start);
  CUDA_CHECK(cudaMemcpy(mses.data(), ctx.d_mses, num_exprs * sizeof(double), cudaMemcpyDeviceToHost));
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ms, start, stop);
  t_d2h += ms;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // Print breakdown
  std::cout << "  [GPU Simple Breakdown]" << std::endl;
  std::cout << "    H2D (Tokens/Vals): " << t_h2d_tokens << " ms" << std::endl;
  std::cout << "    H2D (Data X):      " << t_h2d_X << " ms" << std::endl;
  std::cout << "    Kernel (Eval):     " << t_kernel_eval << " ms" << std::endl;
  std::cout << "    Kernel (Reduce):   " << t_kernel_reduce << " ms" << std::endl;
  std::cout << "    D2H (Results):     " << t_d2h << " ms" << std::endl;
}

// C-style wrappers
// extern "C" {

void* create_gpu_simple_context() {
    return new SimpleGPUContext();
}

void destroy_gpu_simple_context(void* ctx) {
    delete static_cast<SimpleGPUContext*>(ctx);
}

void evaluate_gpu_simple_wrapper(InputInfo& input_info, double*** all_vars, std::vector<double>& mses, void* ctx) {
    evaluate_gpu_simple(input_info, all_vars, mses, *static_cast<SimpleGPUContext*>(ctx));
}

// }
