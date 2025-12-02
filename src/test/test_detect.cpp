#include "../utils/utils.h"
#include "../utils/detect.h"
#include <iostream>

int main() {
    std::cout << "Testing subtree detection..." << std::endl;
    
    // Create simple test case: 3 expressions with common subtree (x0 + x1)
    // Expression 0: MUL(ADD(x0, x1), 2.0)  -> [1, 2, -1, -2, -3]
    // Expression 1: ADD(x0, x1)            -> [2, -1, -2]
    // Expression 2: MUL(ADD(x0, x1), 3.0)  -> [1, 2, -1, -2, -4]
    
    int num_exprs = 3;
    int num_vars = 2;
    
    int num_tokens_arr[3] = {5, 3, 5};
    
    // Allocate tokens
    int* tokens_arr[3];
    tokens_arr[0] = new int[5]{1, 2, -1, -2, -3};  // MUL(ADD(x0, x1), 2.0)
    tokens_arr[1] = new int[3]{2, -1, -2};         // ADD(x0, x1)
    tokens_arr[2] = new int[5]{1, 2, -1, -2, -4};  // MUL(ADD(x0, x1), 3.0)
    
    // Allocate values (constants for negative tokens)
    double* values_arr[3];
    values_arr[0] = new double[5]{0.0, 0.0, 0.0, 0.0, 2.0};
    values_arr[1] = new double[3]{0.0, 0.0, 0.0};
    values_arr[2] = new double[5]{0.0, 0.0, 0.0, 0.0, 3.0};
    
    const int** tokens = const_cast<const int**>(tokens_arr);
    const double** values = const_cast<const double**>(values_arr);
    
    // Run detection
    SubtreeDetectionResult result = detect_common_subtrees(
        num_exprs, num_vars, num_tokens_arr, tokens, values,
        2,  // min_subtree_size
        2   // min_frequency
    );
    
    // Print results
    print_subtree_detection_summary(result, true);
    
    // Verify results
    std::cout << "\nVerification:" << std::endl;
    std::cout << "Expected: 1 common subtree (ADD(x0, x1) appearing 3 times)" << std::endl;
    std::cout << "Found: " << result.num_subs << " common subtrees" << std::endl;
    
    if (result.num_subs > 0) {
        std::cout << "\nSubtree 0 details:" << std::endl;
        std::cout << "  Size: " << result.num_sub_tokens[0] << " tokens" << std::endl;
        std::cout << "  Occurrences: " << result.num_occurrences[0] << std::endl;
        std::cout << "  Token sequence: [";
        for (int i = 0; i < result.num_sub_tokens[0]; i++) {
            if (i > 0) std::cout << ", ";
            std::cout << result.sub_tokens[0][i];
        }
        std::cout << "]" << std::endl;
    }
    
    // Check expr_sub_hints
    std::cout << "\nexpr_sub_hints:" << std::endl;
    for (int expr_id = 0; expr_id < num_exprs; expr_id++) {
        std::cout << "  Expression " << expr_id << ": [";
        for (int i = 0; i < num_tokens_arr[expr_id]; i++) {
            if (i > 0) std::cout << ", ";
            std::cout << result.expr_sub_hints[expr_id][i];
        }
        std::cout << "]" << std::endl;
    }
    
    // Clean up
    free_subtree_detection_result(result);
    for (int i = 0; i < num_exprs; i++) {
        delete[] tokens_arr[i];
        delete[] values_arr[i];
    }
    
    std::cout << "\nTest complete!" << std::endl;
    return 0;
}
