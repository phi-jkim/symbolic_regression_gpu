#ifndef BENCHMARK_UTILS_HPP
#define BENCHMARK_UTILS_HPP

#include <chrono>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

// High-resolution timer for benchmarking
class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    bool running;

public:
    Timer() : running(false) {}

    void start() {
        start_time = std::chrono::high_resolution_clock::now();
        running = true;
    }

    void stop() {
        end_time = std::chrono::high_resolution_clock::now();
        running = false;
    }

    // Returns elapsed time in microseconds
    double elapsed_us() const {
        auto end = running ? std::chrono::high_resolution_clock::now() : end_time;
        return std::chrono::duration<double, std::micro>(end - start_time).count();
    }

    // Returns elapsed time in milliseconds
    double elapsed_ms() const {
        return elapsed_us() / 1000.0;
    }
};

// Result structure for single benchmark run
struct BenchmarkResult {
    std::string name;
    int iterations;
    double total_time_us;
    double avg_time_us;
    double min_time_us;
    double max_time_us;
    double stddev_us;

    BenchmarkResult()
        : name(""), iterations(0), total_time_us(0.0),
          avg_time_us(0.0), min_time_us(0.0), max_time_us(0.0), stddev_us(0.0) {}

    BenchmarkResult(const std::string& n, int iters, const std::vector<double>& times_us)
        : name(n), iterations(iters) {
        total_time_us = 0.0;
        min_time_us = 1e18;
        max_time_us = 0.0;

        for (double t : times_us) {
            total_time_us += t;
            min_time_us = std::min(min_time_us, t);
            max_time_us = std::max(max_time_us, t);
        }

        avg_time_us = total_time_us / iterations;

        // Calculate standard deviation
        double sum_squared_diff = 0.0;
        for (double t : times_us) {
            double diff = t - avg_time_us;
            sum_squared_diff += diff * diff;
        }
        stddev_us = std::sqrt(sum_squared_diff / iterations);
    }

    double throughput_ops_per_sec() const {
        return avg_time_us > 0 ? (1e6 / avg_time_us) : 0.0;
    }
};

// Helper to run warmup iterations
inline void warmup_iterations(int count, std::function<void()> func) {
    for (int i = 0; i < count; i++) {
        func();
    }
}

#endif // BENCHMARK_UTILS_HPP
