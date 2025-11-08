#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP

#include <iostream>
#include <string>
#include <cmath>
#include <iomanip>

// Test result tracking
struct TestStats {
    int total_tests;
    int passed_tests;
    int failed_tests;

    TestStats() : total_tests(0), passed_tests(0), failed_tests(0) {}

    void record_pass() {
        total_tests++;
        passed_tests++;
    }

    void record_fail() {
        total_tests++;
        failed_tests++;
    }

    bool all_passed() const {
        return failed_tests == 0;
    }

    void print_summary() const {
        std::cout << "\n=== Test Summary ===\n";
        std::cout << "Total:  " << total_tests << "\n";
        std::cout << "Passed: " << passed_tests << "\n";
        std::cout << "Failed: " << failed_tests << "\n";
        if (all_passed()) {
            std::cout << "\nALL TESTS PASSED!\n";
        } else {
            std::cout << "\nSOME TESTS FAILED!\n";
        }
    }
};

// Global test stats
static TestStats g_test_stats;

// Assertion macros
#define TEST_ASSERT(condition, message) \
    do { \
        if (condition) { \
            g_test_stats.record_pass(); \
            std::cout << "[PASS] " << message << "\n"; \
        } else { \
            g_test_stats.record_fail(); \
            std::cout << "[FAIL] " << message << " (at " << __FILE__ << ":" << __LINE__ << ")\n"; \
        } \
    } while(0)

#define TEST_ASSERT_EQ(a, b, message) \
    TEST_ASSERT((a) == (b), message)

// Float comparison with tolerance
inline bool float_approx_equal(float a, float b, float tolerance = 1e-5f) {
    return std::fabs(a - b) < tolerance;
}

#define TEST_ASSERT_FLOAT_EQ(a, b, message) \
    do { \
        float _a = (a); \
        float _b = (b); \
        if (float_approx_equal(_a, _b)) { \
            g_test_stats.record_pass(); \
            std::cout << "[PASS] " << message << "\n"; \
        } else { \
            g_test_stats.record_fail(); \
            std::cout << std::fixed << std::setprecision(6); \
            std::cout << "[FAIL] " << message << " (expected " << _b << ", got " << _a << ") at " \
                      << __FILE__ << ":" << __LINE__ << "\n"; \
        } \
    } while(0)

// Test section header
#define TEST_SECTION(name) \
    std::cout << "\n--- " << name << " ---\n"

#endif // TEST_UTILS_HPP
