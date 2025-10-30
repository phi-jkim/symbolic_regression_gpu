#include <iostream>
#include <cmath>
#include "test_utils.hpp"

// Forward declarations from eval_tree.cu
extern "C" {
    float eval_tree_cpu(const int* tokens, const float* values,
                        const float* features, int len, int num_features);
    void eval_tree_gpu(const int* tokens, const float* values,
                       const float* features, int len, int num_features,
                       float* out_host);
}

// Test: CPU vs GPU results match for simple expression
void test_cpu_gpu_match() {
    TEST_SECTION("CPU vs GPU Consistency");

    // Expression: sin(x0) + cos(x1) + 3
    const int tokens[] = {1, 1, 5, -1, 6, -1, 0};
    const float values[] = {0, 0, 0, 0, 0, 1, 3};
    const float features[] = {0.5f, 1.2f};
    const int len = 7;
    const int num_features = 2;

    float cpu_result = eval_tree_cpu(tokens, values, features, len, num_features);
    float gpu_result;
    eval_tree_gpu(tokens, values, features, len, num_features, &gpu_result);

    TEST_ASSERT_FLOAT_EQ(cpu_result, gpu_result, "CPU and GPU results match for sin(x0)+cos(x1)+3");
}

// Test: Basic arithmetic operations
void test_basic_arithmetic() {
    TEST_SECTION("Basic Arithmetic");

    const int num_features = 2;
    float gpu_result;

    // Test: ADD - x0 + x1
    {
        const int tokens[] = {1, -1, -1};
        const float values[] = {0, 0, 1};
        const float features[] = {3.0f, 4.0f};
        const int len = 3;

        float cpu_result = eval_tree_cpu(tokens, values, features, len, num_features);
        eval_tree_gpu(tokens, values, features, len, num_features, &gpu_result);

        TEST_ASSERT_FLOAT_EQ(cpu_result, 7.0f, "ADD: 3 + 4 = 7");
        TEST_ASSERT_FLOAT_EQ(cpu_result, gpu_result, "ADD: CPU and GPU match");
    }

    // Test: SUB - x0 - x1
    {
        const int tokens[] = {2, -1, -1};
        const float values[] = {0, 0, 1};
        const float features[] = {10.0f, 3.0f};
        const int len = 3;

        float cpu_result = eval_tree_cpu(tokens, values, features, len, num_features);
        eval_tree_gpu(tokens, values, features, len, num_features, &gpu_result);

        TEST_ASSERT_FLOAT_EQ(cpu_result, 7.0f, "SUB: 10 - 3 = 7");
        TEST_ASSERT_FLOAT_EQ(cpu_result, gpu_result, "SUB: CPU and GPU match");
    }

    // Test: MUL - x0 * x1
    {
        const int tokens[] = {3, -1, -1};
        const float values[] = {0, 0, 1};
        const float features[] = {3.0f, 4.0f};
        const int len = 3;

        float cpu_result = eval_tree_cpu(tokens, values, features, len, num_features);
        eval_tree_gpu(tokens, values, features, len, num_features, &gpu_result);

        TEST_ASSERT_FLOAT_EQ(cpu_result, 12.0f, "MUL: 3 * 4 = 12");
        TEST_ASSERT_FLOAT_EQ(cpu_result, gpu_result, "MUL: CPU and GPU match");
    }

    // Test: DIV - x0 / x1
    {
        const int tokens[] = {4, -1, -1};
        const float values[] = {0, 0, 1};
        const float features[] = {12.0f, 4.0f};
        const int len = 3;

        float cpu_result = eval_tree_cpu(tokens, values, features, len, num_features);
        eval_tree_gpu(tokens, values, features, len, num_features, &gpu_result);

        TEST_ASSERT_FLOAT_EQ(cpu_result, 3.0f, "DIV: 12 / 4 = 3");
        TEST_ASSERT_FLOAT_EQ(cpu_result, gpu_result, "DIV: CPU and GPU match");
    }
}

// Test: Unary operations
void test_unary_operations() {
    TEST_SECTION("Unary Operations");

    const int num_features = 1;
    float gpu_result;

    // Test: SIN
    {
        const int tokens[] = {5, -1};
        const float values[] = {0, 0};
        const float features[] = {0.0f};
        const int len = 2;

        float cpu_result = eval_tree_cpu(tokens, values, features, len, num_features);
        eval_tree_gpu(tokens, values, features, len, num_features, &gpu_result);

        TEST_ASSERT_FLOAT_EQ(cpu_result, std::sin(0.0f), "SIN: sin(0) = 0");
        TEST_ASSERT_FLOAT_EQ(cpu_result, gpu_result, "SIN: CPU and GPU match");
    }

    // Test: COS
    {
        const int tokens[] = {6, -1};
        const float values[] = {0, 0};
        const float features[] = {0.0f};
        const int len = 2;

        float cpu_result = eval_tree_cpu(tokens, values, features, len, num_features);
        eval_tree_gpu(tokens, values, features, len, num_features, &gpu_result);

        TEST_ASSERT_FLOAT_EQ(cpu_result, std::cos(0.0f), "COS: cos(0) = 1");
        TEST_ASSERT_FLOAT_EQ(cpu_result, gpu_result, "COS: CPU and GPU match");
    }

    // Test: EXP
    {
        const int tokens[] = {7, -1};
        const float values[] = {0, 0};
        const float features[] = {1.0f};
        const int len = 2;

        float cpu_result = eval_tree_cpu(tokens, values, features, len, num_features);
        eval_tree_gpu(tokens, values, features, len, num_features, &gpu_result);

        TEST_ASSERT_FLOAT_EQ(cpu_result, std::exp(1.0f), "EXP: exp(1) = e");
        TEST_ASSERT_FLOAT_EQ(cpu_result, gpu_result, "EXP: CPU and GPU match");
    }

    // Test: LOG
    {
        const int tokens[] = {8, -1};
        const float values[] = {0, 0};
        const float features[] = {2.718281828f};
        const int len = 2;

        float cpu_result = eval_tree_cpu(tokens, values, features, len, num_features);
        eval_tree_gpu(tokens, values, features, len, num_features, &gpu_result);

        TEST_ASSERT_FLOAT_EQ(cpu_result, 1.0f, "LOG: log(e) = 1");
        TEST_ASSERT_FLOAT_EQ(cpu_result, gpu_result, "LOG: CPU and GPU match");
    }
}

// Test: Edge cases
void test_edge_cases() {
    TEST_SECTION("Edge Cases");

    const int num_features = 2;
    float gpu_result;

    // Test: Division by small number (safe_div protection)
    {
        const int tokens[] = {4, -1, -1};
        const float values[] = {0, 0, 1};
        const float features[] = {1.0f, 0.0f};
        const int len = 3;

        float cpu_result = eval_tree_cpu(tokens, values, features, len, num_features);
        eval_tree_gpu(tokens, values, features, len, num_features, &gpu_result);

        // Should not produce NaN or Inf
        TEST_ASSERT(!std::isnan(cpu_result) && !std::isinf(cpu_result), "DIV by zero: CPU result is finite");
        TEST_ASSERT(!std::isnan(gpu_result) && !std::isinf(gpu_result), "DIV by zero: GPU result is finite");
        TEST_ASSERT_FLOAT_EQ(cpu_result, gpu_result, "DIV by zero: CPU and GPU match");
    }

    // Test: Log of negative (safe_log protection)
    {
        const int tokens[] = {8, -1};
        const float values[] = {0, 0};
        const float features[] = {-1.0f};
        const int len = 2;

        float cpu_result = eval_tree_cpu(tokens, values, features, len, num_features);
        eval_tree_gpu(tokens, values, features, len, num_features, &gpu_result);

        // Should not produce NaN
        TEST_ASSERT(!std::isnan(cpu_result), "LOG of negative: CPU result is not NaN");
        TEST_ASSERT(!std::isnan(gpu_result), "LOG of negative: GPU result is not NaN");
        TEST_ASSERT_FLOAT_EQ(cpu_result, gpu_result, "LOG of negative: CPU and GPU match");
    }

    // Test: Constant only
    {
        const int tokens[] = {0};
        const float values[] = {42.0f};
        const float features[] = {0.0f, 0.0f};
        const int len = 1;

        float cpu_result = eval_tree_cpu(tokens, values, features, len, num_features);
        eval_tree_gpu(tokens, values, features, len, num_features, &gpu_result);

        TEST_ASSERT_FLOAT_EQ(cpu_result, 42.0f, "Constant: CPU returns 42");
        TEST_ASSERT_FLOAT_EQ(gpu_result, 42.0f, "Constant: GPU returns 42");
    }
}

int main() {
    std::cout << "=== Expression Evaluator Tests ===\n";

    test_cpu_gpu_match();
    test_basic_arithmetic();
    test_unary_operations();
    test_edge_cases();

    g_test_stats.print_summary();

    return g_test_stats.all_passed() ? 0 : 1;
}
