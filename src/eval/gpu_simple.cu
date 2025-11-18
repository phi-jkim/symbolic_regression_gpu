#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <cmath>
#include "../utils/utils.h"

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// GPU version of eval_op
__device__ double eval_op_gpu(int op, double val1, double val2)
{
    const double DELTA = 1e-9;
    const double MAX_VAL = 1e9;

    switch (op)
    {
    // Binary operators (1-9)
    case 1:
        return val1 + val2; // ADD
    case 2:
        return val1 - val2; // SUB
    case 3:
        return val1 * val2; // MUL
    case 4:
        return (val2 == 0.0) ? NAN : val1 / val2; // DIV
    case 5:
        return pow(val1, val2); // POW
    case 6:
        return (val1 <= val2) ? val1 : val2; // MIN
    case 7:
        return (val1 >= val2) ? val1 : val2; // MAX
    case 8: // LOOSE_DIV
    {
        double denom = fabs(val2) <= DELTA ? (val2 < 0 ? -DELTA : DELTA) : val2;
        return val1 / denom;
    }
    case 9: // LOOSE_POW
        return (val1 == 0.0 && val2 == 0.0) ? 0.0 : pow(fabs(val1), val2);

    // Unary operators (10-27)
    case 10:
        return sin(val1); // SIN
    case 11:
        return cos(val1); // COS
    case 12:
        return tan(val1); // TAN
    case 13:
        return sinh(val1); // SINH
    case 14:
        return cosh(val1); // COSH
    case 15:
        return tanh(val1); // TANH
    case 16:
        return exp(val1); // EXP
    case 17:
        return log(val1); // LOG
    case 18:
        return 1.0 / val1; // INV
    case 19:
        return asin(val1); // ASIN
    case 20:
        return acos(val1); // ACOS
    case 21:
        return atan(val1); // ATAN
    case 22: // LOOSE_LOG
        return (val1 == 0.0) ? -MAX_VAL : log(fabs(val1));
    case 23: // LOOSE_INV
    {
        double denom = fabs(val1) <= DELTA ? (val1 < 0 ? -DELTA : DELTA) : val1;
        return 1.0 / denom;
    }
    case 24:
        return fabs(val1); // ABS
    case 25:
        return -val1; // NEG
    case 26:
        return sqrt(val1); // SQRT
    case 27:
        return sqrt(fabs(val1)); // LOOSE_SQRT
    default:
        return 0;
    }
}



__device__ inline void static_stack_push(double *stk, double val, int &sp)
{
    // stk[sp] = val;
    // sp++;
    #pragma unroll
    for(int i=MAX_STACK_SIZE-2; i>=0; i--)
        stk[i+1] = stk[i];
    stk[0] = val;
}

__device__ inline double static_stack_pop(double *stk, int &sp)
{
    // sp--;
    // return stk[sp];
    double val = stk[0];
    #pragma unroll
    for(int i=MAX_STACK_SIZE-2; i>=0; i--)
        stk[i] = stk[i+1];
    return val;
}

__device__ inline void dyn_stack_push(double *stk, double val, int &sp)
{
    stk[sp] = val;
    sp++;
}

__device__ inline double dyn_stack_pop(double *stk, int &sp)
{
    sp--;
    return stk[sp];
}


// GPU version of eval_tree - each thread has its own stack
__device__ double eval_tree_gpu(int *tokens, double *values, double *x, int num_tokens, int num_vars)
{
    double stk[MAX_STACK_SIZE]; // Local stack per thread
    int sp = 0;
    double tmp, val1, val2;
    
    for (int i = num_tokens - 1; i >= 0; i--)
    {
        int tok = tokens[i];
        if (tok > 0) // operation
        {
            // val1 = stk[sp - 1], sp--;
            val1 = dyn_stack_pop(stk, sp);
            if (tok < 10) // binary operation (1-9)
                // val2 = stk[sp - 1], sp--;
                val2 = dyn_stack_pop(stk, sp);

            tmp = eval_op_gpu(tok, val1, val2);
            // stk[sp] = tmp, sp++;
            dyn_stack_push(stk, tmp, sp);
        }
        else if (tok == 0) // constant
        {
            // stk[sp] = values[i], sp++;
            dyn_stack_push(stk, values[i], sp);
        }
        else if (tok == -1) // variable
        {
            // stk[sp] = x[(int)values[i]], sp++;
            dyn_stack_push(stk, x[(int)values[i]], sp);
        }
    }
    return stk[0];
}

// GPU kernel: Each thread evaluates one datapoint
__global__ void eval_kernel(int *d_tokens, double *d_values,
                            double *d_vars_flat, double *d_pred,
                            int num_tokens, int num_vars, int num_dps)
{
    int dp_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (dp_idx < num_dps)
    {
        // Prepare input variables for this datapoint
        double x[MAX_VAR_NUM]; // Max variables
        
        for (int i = 0; i <= num_vars; i++)
        {
            // vars_flat is laid out as [var0_dp0, var0_dp1, ..., var1_dp0, var1_dp1, ...]
            x[i] = d_vars_flat[i * num_dps + dp_idx];
        }
        
        // Evaluate expression for this datapoint
        d_pred[dp_idx] = eval_tree_gpu(d_tokens, d_values, x, num_tokens, num_vars);
    }
}

// Helper: Flatten 2D host array to 1D for GPU transfer
double* flatten_vars(double **vars, int num_vars, int num_dps)
{
    double *flat = new double[(num_vars + 1) * num_dps];
    
    for (int i = 0; i <= num_vars; i++)
    {
        for (int dp = 0; dp < num_dps; dp++)
        {
            flat[i * num_dps + dp] = vars[i][dp];
        }
    }
    
    return flat;
}

// GPU Timer class (from demo)
class GPUTimer {
private:
    cudaEvent_t start_event, stop_event;
    
public:
    GPUTimer() {
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));
    }
    
    ~GPUTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }
    
    void start() {
        CUDA_CHECK(cudaEventRecord(start_event));
    }
    
    float stop() {
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_event, stop_event));
        return milliseconds;
    }
};

// Batch evaluation function for GPU (matches MultiEvalFunc signature)
void eval_gpu_batch(InputInfo &input_info, double ***all_vars, double **all_predictions)
{
    float total_h2d_time = 0.0f;
    float total_kernel_time = 0.0f;
    float total_d2h_time = 0.0f;
    
    // Process each expression
    for (int expr_id = 0; expr_id < input_info.num_exprs; expr_id++)
    {
        int num_vars = input_info.num_vars[expr_id];
        int num_dps = input_info.num_dps[expr_id];
        int num_tokens = input_info.num_tokens[expr_id];
        int *tokens = input_info.tokens[expr_id];
        double *values = input_info.values[expr_id];
        double **vars = all_vars[expr_id];
        double *pred = all_predictions[expr_id];
        
        // Allocate GPU memory
        int *d_tokens;
        double *d_values, *d_vars_flat, *d_pred;
        
        CUDA_CHECK(cudaMalloc(&d_tokens, num_tokens * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_values, num_tokens * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_vars_flat, (num_vars + 1) * num_dps * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_pred, num_dps * sizeof(double)));
        
        // Flatten vars for GPU transfer
        double *h_vars_flat = flatten_vars(vars, num_vars, num_dps);
        
        // Transfer to GPU (H2D)
        GPUTimer h2d_timer;
        h2d_timer.start();
        CUDA_CHECK(cudaMemcpy(d_tokens, tokens, num_tokens * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_values, values, num_tokens * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vars_flat, h_vars_flat, (num_vars + 1) * num_dps * sizeof(double), cudaMemcpyHostToDevice));
        float h2d_time = h2d_timer.stop();
        total_h2d_time += h2d_time;
        
        // Launch kernel
        int threads_per_block = 256;
        int blocks = (num_dps + threads_per_block - 1) / threads_per_block;
        
        GPUTimer kernel_timer;
        kernel_timer.start();
        eval_kernel<<<blocks, threads_per_block>>>(d_tokens, d_values, d_vars_flat, d_pred, num_tokens, num_vars, num_dps);
        float kernel_time = kernel_timer.stop();
        total_kernel_time += kernel_time;
        
        // Check for kernel errors
        CUDA_CHECK(cudaGetLastError());
        
        // Transfer results back (D2H)
        GPUTimer d2h_timer;
        d2h_timer.start();
        CUDA_CHECK(cudaMemcpy(pred, d_pred, num_dps * sizeof(double), cudaMemcpyDeviceToHost));
        float d2h_time = d2h_timer.stop();
        total_d2h_time += d2h_time;
        
        // Free GPU memory
        CUDA_CHECK(cudaFree(d_tokens));
        CUDA_CHECK(cudaFree(d_values));
        CUDA_CHECK(cudaFree(d_vars_flat));
        CUDA_CHECK(cudaFree(d_pred));
        
        delete[] h_vars_flat;
    }
    
    // Print internal GPU timing breakdown
    std::cout << "\nGPU Internal Timing:" << std::endl;
    std::cout << "  H→D Transfer:   " << total_h2d_time << " ms" << std::endl;
    std::cout << "  Kernel Exec:    " << total_kernel_time << " ms" << std::endl;
    std::cout << "  D→H Transfer:   " << total_d2h_time << " ms" << std::endl;
    std::cout << "  GPU Subtotal:   " << (total_h2d_time + total_kernel_time + total_d2h_time) << " ms" << std::endl;
}
