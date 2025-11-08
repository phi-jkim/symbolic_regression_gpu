#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <cassert>
#include <functional>

#include "benchmark_utils.hpp"

// Forward declarations from eval_tree.cu
extern "C" {
    float eval_tree_cpu(const int* tokens, const float* values,
                        const float* features, int len, int num_features);
    void eval_tree_gpu(const int* tokens, const float* values,
                       const float* features, int len, int num_features,
                       float* out_host);
}

// Benchmark configuration
struct BenchConfig {
    int warmup_iterations;
    int benchmark_iterations;
    bool run_cpu;
    bool run_gpu;
    std::string output_file;

    BenchConfig()
        : warmup_iterations(10), benchmark_iterations(1000),
          run_cpu(true), run_gpu(true), output_file("") {}
};

// Run CPU benchmark
BenchmarkResult benchmark_cpu(const int* tokens, const float* values,
                              const float* features, int len, int num_features,
                              const BenchConfig& config) {
    std::vector<double> times_us;
    Timer timer;

    // Warmup
    for (int i = 0; i < config.warmup_iterations; i++) {
        eval_tree_cpu(tokens, values, features, len, num_features);
    }

    // Benchmark
    for (int i = 0; i < config.benchmark_iterations; i++) {
        timer.start();
        float result = eval_tree_cpu(tokens, values, features, len, num_features);
        timer.stop();
        times_us.push_back(timer.elapsed_us());
        (void)result; // Prevent optimization
    }

    return BenchmarkResult("CPU", config.benchmark_iterations, times_us);
}

// Run GPU benchmark
BenchmarkResult benchmark_gpu(const int* tokens, const float* values,
                              const float* features, int len, int num_features,
                              const BenchConfig& config) {
    std::vector<double> times_us;
    Timer timer;
    float result;

    // Warmup
    for (int i = 0; i < config.warmup_iterations; i++) {
        eval_tree_gpu(tokens, values, features, len, num_features, &result);
    }

    // Benchmark
    for (int i = 0; i < config.benchmark_iterations; i++) {
        timer.start();
        eval_tree_gpu(tokens, values, features, len, num_features, &result);
        timer.stop();
        times_us.push_back(timer.elapsed_us());
    }

    return BenchmarkResult("GPU", config.benchmark_iterations, times_us);
}

// Print benchmark results
void print_result(const BenchmarkResult& result) {
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n--- " << result.name << " Benchmark ---\n";
    std::cout << "Iterations:  " << result.iterations << "\n";
    std::cout << "Avg time:    " << result.avg_time_us << " us\n";
    std::cout << "Min time:    " << result.min_time_us << " us\n";
    std::cout << "Max time:    " << result.max_time_us << " us\n";
    std::cout << "Std dev:     " << result.stddev_us << " us\n";
    std::cout << "Throughput:  " << std::setprecision(0) << result.throughput_ops_per_sec() << " ops/sec\n";
}

// Write results to CSV
void write_csv(const std::vector<BenchmarkResult>& results, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open " << filename << " for writing\n";
        return;
    }

    file << "name,iterations,avg_time_us,min_time_us,max_time_us,stddev_us,throughput_ops_per_sec\n";
    for (const auto& result : results) {
        file << result.name << ","
             << result.iterations << ","
             << result.avg_time_us << ","
             << result.min_time_us << ","
             << result.max_time_us << ","
             << result.stddev_us << ","
             << result.throughput_ops_per_sec() << "\n";
    }
    file.close();
    std::cout << "\nResults written to " << filename << "\n";
}

int main(int argc, char** argv) {
    // Parse command-line arguments
    BenchConfig config;
    if (argc > 1) config.warmup_iterations = std::stoi(argv[1]);
    if (argc > 2) config.benchmark_iterations = std::stoi(argv[2]);
    if (argc > 3) config.output_file = argv[3];

    std::cout << "=== Expression Evaluator Benchmark ===\n";
    std::cout << "Warmup iterations:     " << config.warmup_iterations << "\n";
    std::cout << "Benchmark iterations:  " << config.benchmark_iterations << "\n";

    // Test expression: sin(x0) + cos(x1) + 3
    const int tokens[] = {1, 1, 5, -1, 6, -1, 0};
    const float values[] = {0, 0, 0, 0, 0, 1, 3};
    const float features[] = {0.5f, 1.2f};
    const int len = 7;
    const int num_features = 2;

    std::cout << "\nTest expression: sin(x0) + cos(x1) + 3\n";
    std::cout << "Test point: [" << features[0] << ", " << features[1] << "]\n";

    std::vector<BenchmarkResult> results;

    // Run CPU benchmark
    if (config.run_cpu) {
        auto cpu_result = benchmark_cpu(tokens, values, features, len, num_features, config);
        print_result(cpu_result);
        results.push_back(cpu_result);
    }

    // Run GPU benchmark
    if (config.run_gpu) {
        auto gpu_result = benchmark_gpu(tokens, values, features, len, num_features, config);
        print_result(gpu_result);
        results.push_back(gpu_result);
    }

    // Write CSV if requested
    if (!config.output_file.empty()) {
        write_csv(results, config.output_file);
    }

    std::cout << "\n=== Benchmark Complete ===\n";
    return 0;
}
