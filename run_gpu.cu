#include <cstdio>
#include <vector>
#include <cstdlib>
#include <chrono>

extern "C" float eval_tree_cpu(const int* tokens,
                                const float* values,
                                const float* features,
                                int len,
                                int num_features);

extern "C" void eval_tree_gpu(const int* tokens,
                               const float* values,
                               const float* features,
                               int len,
                               int num_features,
                               float* out_host);

static void run_example(const char* name,
                        const std::vector<int>& tokens,
                        const std::vector<float>& values,
                        const std::vector<float>& x) {
    const int iters_cpu = 1000;
    const int iters_gpu = 100;

    // Single evaluations (also verify values)
    float cpu = eval_tree_cpu(tokens.data(), values.data(), x.data(), (int)tokens.size(), (int)x.size());
    float gpu = 0.0f;
    eval_tree_gpu(tokens.data(), values.data(), x.data(), (int)tokens.size(), (int)x.size(), &gpu);
    std::printf("%s -> cpu=%.7f gpu=%.7f\n", name, cpu, gpu);

    // CPU timing (average per call)
    auto t0 = std::chrono::high_resolution_clock::now();
    float acc_cpu = 0.0f;
    for (int i = 0; i < iters_cpu; ++i) {
        acc_cpu += eval_tree_cpu(tokens.data(), values.data(), x.data(), (int)tokens.size(), (int)x.size());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double us_cpu = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    double avg_us_cpu = us_cpu / iters_cpu;

    // GPU timing (average per call; includes allocations inside eval_tree_gpu)
    auto g0 = std::chrono::high_resolution_clock::now();
    float acc_gpu = 0.0f, tmp = 0.0f;
    for (int i = 0; i < iters_gpu; ++i) {
        tmp = 0.0f;
        eval_tree_gpu(tokens.data(), values.data(), x.data(), (int)tokens.size(), (int)x.size(), &tmp);
        acc_gpu += tmp;
    }
    auto g1 = std::chrono::high_resolution_clock::now();
    double us_gpu = std::chrono::duration_cast<std::chrono::microseconds>(g1 - g0).count();
    double avg_us_gpu = us_gpu / iters_gpu;

    std::printf("%s timings: cpu_avg_us=%.3f (iters=%d) gpu_avg_us=%.3f (iters=%d)\n",
                name, avg_us_cpu, iters_cpu, avg_us_gpu, iters_gpu);
}

int main(int argc, char** argv) {
    // Match run_single_point.jl example:
    // expr = sin(x1) + cos(x2) + 3  (0-based here: sin(x0) + cos(x1) + 3)
    // Prefix tree: ADD, ADD, SIN, VAR(0), COS, VAR(1), CONST(3)
    std::vector<int> tokens = {1, 1, 5, -1, 6, -1, 0};
    std::vector<float> values = {0, 0, 0, 0, 0, 1, 3};

    // Point 1: X = [0.5, 1.2]
    {
        std::vector<float> x = {0.5f, 1.2f};
        run_example("sin(x0)+cos(x1)+3 @ [0.5,1.2]", tokens, values, x);
    }
    // Point 2: Y = [0.5, 0.3]
    {
        std::vector<float> x = {0.5f, 0.3f};
        run_example("sin(x0)+cos(x1)+3 @ [0.5,0.3]", tokens, values, x);
    }

    return 0;
}
