#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include "../utils/utils.h"
#include "../utils/gpu_kernel.h"


// Import eval_tree_gpu_batch from utils.cu
extern "C" void eval_tree_gpu_batch(
    const int* tokens,
    const float* values,
    const float* X,           // [dataPoints, num_features]
    int len,
    int num_features,
    int dataPoints,
    float* out_dev,
    int blocks,
    int threads,
    float* s_val);

#ifndef MAX_EVAL_STACK
#define MAX_EVAL_STACK 60
#endif

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

void eval_jinha_batch(InputInfo &input_info, double ***all_vars, double **all_predictions)
{
    double alloc_ms = 0.0;
    double memcpy_h2d_ms_total = 0.0;
    double memcpy_d2h_ms_total = 0.0;
    float total_gpu_ms = 0.0f; // sum of kernel times over expressions (excluding expr 0)
    float total_gpu_ms_incl0 = 0.0f; // sum including expr 0
    int   counted_exprs = 0;   // number of expressions included in totals/averages
    std::vector<float> gpu_ms_list; // per-expression kernel times (excluding expr 0)
    // allocate device buffers once using precomputed maxima in InputInfo
    int *d_tokens = nullptr;
    float *d_values = nullptr;
    float *d_X = nullptr;
    float *d_out = nullptr;
    
    TimePoint t_alloc = measure_clock();
    CUDA_CHECK(cudaMalloc(&d_tokens, (size_t)input_info.max_tokens * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values, (size_t)input_info.max_tokens * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_X, (size_t)input_info.max_num_dps * (size_t)input_info.max_num_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)input_info.max_num_dps * sizeof(float)));
    CUDA_CHECK(cudaDeviceSynchronize());
    alloc_ms = clock_to_ms(t_alloc, measure_clock());

    // reusable timing events
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    for (int expr_id = 0; expr_id < input_info.num_exprs; expr_id++)
    {
        int num_vars = input_info.num_vars[expr_id];
        int num_dps = input_info.num_dps[expr_id];
        int num_tokens = input_info.num_tokens[expr_id];
        int num_features = num_vars + 1;  // Include output variable
        
        // ============================================
        // Step 1: Copy tokens (no conversion needed)
        // ============================================
        const int* tokens_host_ptr = nullptr;
        std::vector<int> tokens_temp;
        if (input_info.tokens_packed && input_info.tokens_packed[expr_id]) {
            tokens_host_ptr = input_info.tokens_packed[expr_id];
        } else {
            // redudant copying
            tokens_temp.resize(num_tokens);
            for (int i = 0; i < num_tokens; i++) 
                tokens_temp[i] = input_info.tokens[expr_id][i];
            tokens_host_ptr = tokens_temp.data();
        }
        
        // ============================================
        // Step 2: Convert values double -> float
        // ============================================
        const float* values_host_ptr = nullptr;
        std::vector<float> values_temp;
        if (input_info.values_packed_f32 && input_info.values_packed_f32[expr_id]) {
            values_host_ptr = input_info.values_packed_f32[expr_id];
        } else {
            // redudant copying
            values_temp.resize(num_tokens);
            for (int i = 0; i < num_tokens; i++) 
                values_temp[i] = (float)input_info.values[expr_id][i];
            values_host_ptr = values_temp.data();
        }
        
        const float* X_host_ptr = nullptr;
        std::vector<float> X_temp;
        if (input_info.X_packed_f32 && input_info.X_packed_f32[expr_id]) {
            X_host_ptr = input_info.X_packed_f32[expr_id];
        } else {
            X_temp.resize((size_t)num_dps * (size_t)num_features);
            for (int dp = 0; dp < num_dps; dp++) {
                for (int var = 0; var < num_features; var++) {
                    X_temp[(size_t)dp * (size_t)num_features + var] = (float) all_vars[expr_id][var][dp];
                }
            }
            X_host_ptr = X_temp.data();
        }
        
        // ============================================
        // Step 4: Device memory already allocated (reused)
        // ============================================
        
        // ============================================
        // Step 5: Copy to device
        // ============================================
        TimePoint t_h2d = measure_clock();
        CUDA_CHECK(cudaMemcpy(d_tokens, tokens_host_ptr, 
                             num_tokens * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_values, values_host_ptr, 
                             num_tokens * sizeof(float), cudaMemcpyHostToDevice));
        
        // Host-side early-exit check (no device symbol):
        const bool host_early = (num_tokens > MAX_EVAL_STACK);
        fprintf(stderr,
                "[expr %d] host_early_exit(len>%d): %s (len=%d)\n",
                expr_id, MAX_EVAL_STACK, host_early ? "YES" : "NO", num_tokens);
        
        CUDA_CHECK(cudaMemcpy(d_X, X_host_ptr,
                              (size_t)num_dps * (size_t)num_features * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
        memcpy_h2d_ms_total += clock_to_ms(t_h2d, measure_clock());
        
        // ============================================
        // Step 6: Call eval_tree_gpu_batch kernel
        // ============================================
        int threads = 128;
        // int threads=64; 
        int blocks = (num_dps + threads - 1) / threads;
        
        // Time only the GPU kernel execution
        CUDA_CHECK(cudaEventRecord(ev_start));

        eval_tree_gpu_batch(d_tokens, d_values, d_X, 
                           num_tokens, num_features, num_dps,
                           d_out, blocks, threads, nullptr);

        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        total_gpu_ms_incl0 += ms;
        if (expr_id == 0) {
            // Print first expression's time but do not add it to totals
            std::cout << "[expr 0] kernel time: " << ms << " ms (excluded from totals)" << std::endl;
        } else {
            total_gpu_ms += ms;
            counted_exprs += 1;
            gpu_ms_list.push_back(ms);
        }

        // ============================================
        // Step 7: Copy results back and convert float -> double
        // ============================================
        std::vector<float> out_float(num_dps);
        TimePoint t_d2h = measure_clock();
        CUDA_CHECK(cudaMemcpy(out_float.data(), d_out,
                              num_dps * sizeof(float), cudaMemcpyDeviceToHost));
        memcpy_d2h_ms_total += clock_to_ms(t_d2h, measure_clock());
        for (int dp = 0; dp < num_dps; dp++) {
            all_predictions[expr_id][dp] = (double)out_float[dp];
        }
    }
    // cleanup reusable resources
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaFree(d_tokens));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_out));

    // Summarize kernel timings in the same style as gpu_custom_kernel_per_expression
    if (!gpu_ms_list.empty()) {
        float total_ms = 0.0f;
        for (float v : gpu_ms_list) total_ms += v;
        float first_ms = gpu_ms_list.front();

        std::vector<float> sorted_all = gpu_ms_list;
        std::sort(sorted_all.begin(), sorted_all.end());
        float  median_ms = 0.0f;
        size_t n         = sorted_all.size();
        if (n % 2 == 1)
            median_ms = sorted_all[n / 2];
        else
            median_ms = 0.5f * (sorted_all[n / 2 - 1] + sorted_all[n / 2]);

        float  mean_excl_first_ms = 0.0f;
        size_t n_excl             = 0;
        if (gpu_ms_list.size() > 1) {
            for (size_t i = 1; i < gpu_ms_list.size(); ++i) {
                mean_excl_first_ms += gpu_ms_list[i];
            }
            n_excl = gpu_ms_list.size() - 1;
            mean_excl_first_ms /= (float)n_excl;
        }

        fprintf(stderr,
                "[evolve] kernel time (launch+sync): total=%g ms, first=%g ms, median_all=%g ms, "
                "mean_excl_first=%g ms over %zu evals (%zu excl first)\n",
                (double)total_ms, (double)first_ms, (double)median_ms,
                (double)mean_excl_first_ms, n, n_excl);
    }

    std::cout << "Alloc time (host wall): " << alloc_ms << " ms" << std::endl;
    std::cout << "Total H2D memcpy time (host wall): " << memcpy_h2d_ms_total << " ms" << std::endl;
    std::cout << "Total D2H memcpy time (host wall): " << memcpy_d2h_ms_total << " ms" << std::endl;
}


// Evolve a GP population for a single expression and write predictions of the
// best-found individual back into all_predictions[expr_idx].
//
// input_info      : Global metadata for all expressions (num_vars, num_dps, etc.).
// all_vars        : all_vars[expr][var][dp] holds the raw double data for each
//                   expression; for expr_idx we treat the first num_vars columns
//                   as inputs and the (num_vars)-th column as the target label.
// all_predictions : Output buffer; evolve fills all_predictions[expr_idx][dp]
//                   with the best individuals prediction for each datapoint.
// expr_idx        : Index of the expression to evolve (0 <= expr_idx < num_exprs).
// pop_size        : Number of individuals in the GP population.
// num_generations : How many evolutionary generations to run (currently used
//                   only to re-evaluate the same population each loop; no
//                   variation yet).
// constSamplesLen : Length of the constSamples array used during generation.
// outProb         : Probability of generating an OUT-node (only relevant for
//                   multi-output; typically 0 for single-output SR).
// constProb       : Probability that a leaf is a constant rather than a
//                   variable when trees are generated.
// h_keys          : Host-side RNG seed keys array of length 2 used by generate.
// h_depth2leaf    : Host-side array of length MAX_FULL_DEPTH giving the
//                   probability of generating a leaf at each depth.
// h_roulette      : Host-side array of length Function::END containing the
//                   cumulative roulette-wheel probabilities for function nodes.
// h_consts        : Host-side array of length constSamplesLen with the pool of
//                   constant values available during tree generation.
void evolve(InputInfo &input_info,
            double ***all_vars,
            double **all_predictions,
            int expr_idx,
            int pop_size,
            int num_generations,
            int maxGPLen,
            unsigned int constSamplesLen,
            float outProb,
            float constProb,
            const unsigned int* h_keys,
            const float* h_depth2leaf,
            const float* h_roulette,
            const float* h_consts,
            std::vector<float>* kernel_times_accum /* optional */)
{
    if (expr_idx < 0 || expr_idx >= input_info.num_exprs) {
        fprintf(stderr, "evolve: invalid expr_idx %d (num_exprs=%d)\n", expr_idx, input_info.num_exprs);
        return;
    }

    int num_vars = input_info.num_vars[expr_idx];
    int num_dps = input_info.num_dps[expr_idx];

    if (num_vars <= 0 || num_dps <= 0 || maxGPLen <= 0 || pop_size <= 0 || num_generations <= 0) {
        fprintf(stderr, "evolve: invalid parameters (num_vars=%d, num_dps=%d, maxGPLen=%d, pop=%d, gens=%d)\n",
                num_vars, num_dps, maxGPLen, pop_size, num_generations);
        return;
    }

    int varLen = num_vars;
    int outLen = 1;

    // Build input matrix X_inputs [num_dps, varLen] and labels [num_dps]
    std::vector<float> h_X_inputs((size_t)num_dps * (size_t)varLen);
    std::vector<float> h_labels(num_dps);
    for (int dp = 0; dp < num_dps; ++dp) {
        for (int v = 0; v < varLen; ++v) {
            h_X_inputs[(size_t)dp * (size_t)varLen + v] = (float)all_vars[expr_idx][v][dp];
        }
        // assume last column is output/label
        h_labels[dp] = (float)all_vars[expr_idx][varLen][dp];
    }

    // Device buffers for population
    float *d_val = nullptr;
    int16_t *d_type = nullptr;
    int16_t *d_size = nullptr;
    CUDA_CHECK(cudaMalloc(&d_val,  (size_t)pop_size * (size_t)maxGPLen * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_type, (size_t)pop_size * (size_t)maxGPLen * sizeof(int16_t)));
    CUDA_CHECK(cudaMalloc(&d_size, (size_t)pop_size * (size_t)maxGPLen * sizeof(int16_t)));

    unsigned int *d_keys = nullptr;
    float *d_depth2leaf = nullptr;
    float *d_roulette = nullptr;
    float *d_consts = nullptr;

    CUDA_CHECK(cudaMalloc(&d_keys, 2 * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_depth2leaf, MAX_FULL_DEPTH * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_roulette, Function::END * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_consts, constSamplesLen * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_keys, h_keys, 2 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_depth2leaf, h_depth2leaf, MAX_FULL_DEPTH * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_roulette, h_roulette, Function::END * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_consts, h_consts, constSamplesLen * sizeof(float), cudaMemcpyHostToDevice));

    // Generate initial population
    generate((unsigned int)pop_size,
             (unsigned int)maxGPLen,
             (unsigned int)varLen,
             (unsigned int)outLen,
             constSamplesLen,
             outProb,
             constProb,
             d_keys,
             d_depth2leaf,
             d_roulette,
             d_consts,
             d_val,
             d_type,
             d_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Device buffers for inputs and predictions
    float *d_X = nullptr;
    float *d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_X, (size_t)num_dps * (size_t)varLen * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, (size_t)num_dps * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_X, h_X_inputs.data(), h_X_inputs.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_labels.data(), h_labels.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Temporary buffers for evaluation of a single individual
    int *d_tokens = nullptr;
    float *d_values = nullptr;
    float *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tokens, maxGPLen * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values, maxGPLen * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)num_dps * sizeof(float)));

    float *s_val = nullptr;
    CUDA_CHECK(cudaMalloc(&s_val, (size_t)num_dps * (size_t)MAX_EVAL_STACK * sizeof(float)));

    std::vector<int> h_tokens(maxGPLen);
    std::vector<float> h_values(maxGPLen);
    // Remove 
    std::vector<double> h_values_d(maxGPLen); // for readable formula via format_formula
    std::vector<float> h_out(num_dps);

    std::vector<float> h_pop_val((size_t)pop_size * (size_t)maxGPLen);
    std::vector<int16_t> h_pop_type((size_t)pop_size * (size_t)maxGPLen);
    std::vector<int16_t> h_pop_size((size_t)pop_size * (size_t)maxGPLen);

    float best_fitness = std::numeric_limits<float>::infinity();
    int best_idx = -1;

    // Kernel timing (only eval_tree_gpu_batch calls)
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    std::vector<float> kernel_times_ms;
    kernel_times_ms.reserve((size_t)pop_size * (size_t)num_generations);

    for (int gen = 0; gen < num_generations; ++gen) {
        CUDA_CHECK(cudaMemcpy(h_pop_val.data(), d_val,
                             h_pop_val.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_pop_type.data(), d_type,
                             h_pop_type.size() * sizeof(int16_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_pop_size.data(), d_size,
                             h_pop_size.size() * sizeof(int16_t), cudaMemcpyDeviceToHost));

        for (int i = 0; i < pop_size; ++i) {
            const float *val_i = &h_pop_val[(size_t)i * (size_t)maxGPLen];
            const int16_t *type_i = &h_pop_type[(size_t)i * (size_t)maxGPLen];
            const int16_t *sub_i = &h_pop_size[(size_t)i * (size_t)maxGPLen];

            int len = (int)sub_i[0];
            if (len <= 0 || len > maxGPLen) continue;
            // printf("gen=%d, indv=%d, len=%d\n", gen, i, len); 

            for (int t = 0; t < len; ++t) {
                int16_t node_type = (int16_t)(type_i[t] & NodeType::TYPE_MASK);
                float node_val = val_i[t];
                if (node_type == NodeType::CONST) {
                    // constant leaf: token marks CONST, value is numeric constant
                    h_tokens[t] = TOK_CONST;
                    h_values[t] = node_val;
                } else if (node_type == NodeType::VAR) {
                    // variable leaf: token marks VAR, value is variable index
                    h_tokens[t] = TOK_VAR;
                    h_values[t] = node_val;
                } else {
                    // function node: token is opcode, value unused by format_formula
                    h_tokens[t] = static_cast<int>(node_val);
                    h_values[t] = 0.0f;
                }
                // keep a double copy for pretty-printing
                h_values_d[t] = static_cast<double>(h_values[t]);
            }

            // Debug: human-readable symbolic formula (infix) using existing helper
            // std::string formula = format_formula(h_tokens.data(), h_values_d.data(), len);
            // fprintf(stderr, "[evolve] gen=%d ind=%d len=%d formula=%s\n", gen, i, len, formula.c_str());

            CUDA_CHECK(cudaMemcpy(d_tokens, h_tokens.data(), len * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_values, h_values.data(), len * sizeof(float), cudaMemcpyHostToDevice));

            int threads = 128;
            int blocks = (num_dps + threads - 1) / threads;

            // Time only the GPU kernel execution for this individual
            CUDA_CHECK(cudaEventRecord(ev_start));
            eval_tree_gpu_batch(d_tokens, d_values, d_X,
                               len, varLen, num_dps,
                               d_out, blocks, threads, s_val);
            CUDA_CHECK(cudaEventRecord(ev_stop));
            CUDA_CHECK(cudaEventSynchronize(ev_stop));

            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
            kernel_times_ms.push_back(ms);

            CUDA_CHECK(cudaMemcpy(h_out.data(), d_out,
                                 (size_t)num_dps * sizeof(float), cudaMemcpyDeviceToHost));

            double mse = 0.0;
            for (int dp = 0; dp < num_dps; ++dp) {
                double diff = (double)h_out[dp] - (double)h_labels[dp];
                mse += diff * diff;
            }
            mse /= (double)num_dps;

            if ((float)mse < best_fitness) {
                best_fitness = (float)mse;
                best_idx = i;
            }
        }
        fprintf(stderr, "[evolve] generation %d best MSE=%g (idx=%d)\n", gen, best_fitness, best_idx);
    }

    if (!kernel_times_ms.empty()) {
        // Compute total, first, median, and mean(excluding first) kernel time over all evals
        float total_ms = 0.0f;
        for (float v : kernel_times_ms) total_ms += v;
        float first_ms = kernel_times_ms.front();

        // Median over all evals (including first)
        std::vector<float> sorted_all = kernel_times_ms;
        std::sort(sorted_all.begin(), sorted_all.end());
        float median_ms = 0.0f;
        size_t n = sorted_all.size();
        if (n % 2 == 1)
            median_ms = sorted_all[n / 2];
        else
            median_ms = 0.5f * (sorted_all[n / 2 - 1] + sorted_all[n / 2]);

        // Mean excluding the first eval (to avoid cold-start / JIT effects)
        float mean_excl_first_ms = 0.0f;
        size_t n_excl = 0;
        if (kernel_times_ms.size() > 1) {
            for (size_t i = 1; i < kernel_times_ms.size(); ++i) {
                mean_excl_first_ms += kernel_times_ms[i];
            }
            n_excl = kernel_times_ms.size() - 1;
            mean_excl_first_ms /= (float)n_excl;
        }

        fprintf(stderr,
                "[evolve] kernel time: total=%g ms, first=%g ms, median_all=%g ms, mean_excl_first=%g ms over %zu evals (%zu excl first)\n",
                (double)total_ms, (double)first_ms, (double)median_ms,
                (double)mean_excl_first_ms, n, n_excl);
    }

    // Optionally export per-kernel timings to an external accumulator
    if (kernel_times_accum && !kernel_times_ms.empty()) {
        kernel_times_accum->insert(kernel_times_accum->end(),
                                   kernel_times_ms.begin(),
                                   kernel_times_ms.end());
    }

    if (best_idx >= 0) {
        const float *val_i = &h_pop_val[(size_t)best_idx * (size_t)maxGPLen];
        const int16_t *type_i = &h_pop_type[(size_t)best_idx * (size_t)maxGPLen];
        const int16_t *sub_i = &h_pop_size[(size_t)best_idx * (size_t)maxGPLen];

        int len = (int)sub_i[0];
        if (len > 0 && len <= maxGPLen) {
            for (int t = 0; t < len; ++t) {
                int16_t node_type = (int16_t)(type_i[t] & NodeType::TYPE_MASK);
                float node_val = val_i[t];
                if (node_type == NodeType::CONST) {
                    h_tokens[t] = TOK_CONST;
                    h_values[t] = node_val;
                } else if (node_type == NodeType::VAR) {
                    h_tokens[t] = TOK_VAR;
                    h_values[t] = node_val;
                } else {
                    h_tokens[t] = (int)node_val;
                    h_values[t] = 0.0f;
                }
            }

            CUDA_CHECK(cudaMemcpy(d_tokens, h_tokens.data(), len * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_values, h_values.data(), len * sizeof(float), cudaMemcpyHostToDevice));

            int threads = 128;
            int blocks = (num_dps + threads - 1) / threads;
            eval_tree_gpu_batch(d_tokens, d_values, d_X,
                               len, varLen, num_dps,
                               d_out, blocks, threads, s_val);

            CUDA_CHECK(cudaMemcpy(h_out.data(), d_out,
                                 (size_t)num_dps * sizeof(float), cudaMemcpyDeviceToHost));

            for (int dp = 0; dp < num_dps; ++dp) {
                all_predictions[expr_idx][dp] = (double)h_out[dp];
            }
        }
    }

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    cudaFree(d_tokens);
    cudaFree(d_values);
    cudaFree(d_out);
    cudaFree(s_val);
    cudaFree(d_X);
    cudaFree(d_y);
    cudaFree(d_val);
    cudaFree(d_type);
    cudaFree(d_size);
    cudaFree(d_keys);
    cudaFree(d_depth2leaf);
    cudaFree(d_roulette);
    cudaFree(d_consts);
}


// Evolution-based batch entry point used when building with USE_GPU_EVOLVE_JINHA.
// It mirrors the eval_jinha_batch signature so it can be selected via eval_batch
// in evaluator.h and called from evaluate_and_save_results.
void eval_evolve_jinha_batch(InputInfo &input_info, double ***all_vars, double **all_predictions)
{
    // Simple, fixed evolution configuration for now. These can be exposed later.
    const int pop_size = 256;
    const int num_generations = 5;
    // const int maxGPLen = 1024; // configurable GP maximum length (independent of input_info.max_tokens)
    const int maxGPLen = 60; 
    const unsigned int constSamplesLen = 8;
    const float outProb = 0.0f;
    const float constProb = 0.01f;

    // Host-side tables mirroring evolution_tests.cpp
    unsigned int h_keys[2] = {42u, 1337u};
    float h_depth[MAX_FULL_DEPTH];
    for (int i = 0; i < MAX_FULL_DEPTH; ++i) h_depth[i] = 0.0001f;

    std::vector<float> h_roulette(Function::LOOSE_SQRT + 1);
    h_roulette[0] = 0.0f;
    for (int i = 1; i <= Function::LOOSE_SQRT; ++i)
        h_roulette[i] = static_cast<float>(i) / static_cast<float>(Function::LOOSE_SQRT);

    float h_consts[constSamplesLen] = {-1.f, -0.5f, 0.f, 0.5f, 1.f, 2.f, 3.f, 4.f};

    // Accumulate kernel timings across all expressions and generations
    std::vector<float> batch_kernel_times;
    batch_kernel_times.reserve((size_t)input_info.num_exprs * (size_t)pop_size * (size_t)num_generations);

    for (int expr_idx = 0; expr_idx < input_info.num_exprs; ++expr_idx) {
        evolve(input_info,
               all_vars,
               all_predictions,
               expr_idx,
               pop_size,
               num_generations,
               maxGPLen,
               constSamplesLen,
               outProb,
               constProb,
               h_keys,
               h_depth,
               h_roulette.data(),
               h_consts,
               &batch_kernel_times);
    }

    if (!batch_kernel_times.empty()) {
        float total_ms = 0.0f;
        for (float v : batch_kernel_times) total_ms += v;
        float first_ms = batch_kernel_times.front();

        std::vector<float> sorted_all = batch_kernel_times;
        std::sort(sorted_all.begin(), sorted_all.end());
        float  median_ms = 0.0f;
        size_t n         = sorted_all.size();
        if (n % 2 == 1)
            median_ms = sorted_all[n / 2];
        else
            median_ms = 0.5f * (sorted_all[n / 2 - 1] + sorted_all[n / 2]);

        float  mean_excl_first_ms = 0.0f;
        size_t n_excl             = 0;
        if (batch_kernel_times.size() > 1) {
            for (size_t i = 1; i < batch_kernel_times.size(); ++i) {
                mean_excl_first_ms += batch_kernel_times[i];
            }
            n_excl = batch_kernel_times.size() - 1;
            mean_excl_first_ms /= (float)n_excl;
        }

        fprintf(stderr,
                "[evolve_batch] kernel time (launch+sync): total=%g ms, first=%g ms, median_all=%g ms, "
                "mean_excl_first=%g ms over %zu evals (%zu excl first)\n",
                (double)total_ms, (double)first_ms, (double)median_ms,
                (double)mean_excl_first_ms, n, n_excl);
    }
}