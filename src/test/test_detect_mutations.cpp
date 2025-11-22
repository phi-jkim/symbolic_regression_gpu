#include "../utils/utils.h"
#include "../utils/detect.h"
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <digest_file> [min_subtree_size] [min_frequency]" << std::endl;
        std::cerr << "Example: " << argv[0] << " data/ai_feyn/mutations/input_base056_100mut_1000k.txt 3 2" << std::endl;
        return 1;
    }
    
    std::string digest_file = argv[1];
    int min_subtree_size = (argc >= 3) ? atoi(argv[2]) : 3;
    int min_frequency = (argc >= 4) ? atoi(argv[3]) : 2;
    
    std::cout << "Testing subtree detection on: " << digest_file << std::endl;
    std::cout << "Parameters: min_subtree_size=" << min_subtree_size 
              << ", min_frequency=" << min_frequency << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    // Parse input file
    TimePoint parse_start = measure_clock();
    InputInfo input_info = parse_input_info(digest_file);
    double parse_time = clock_to_ms(parse_start, measure_clock());
    
    if (input_info.num_exprs == 0) {
        std::cerr << "Error: Failed to parse input file" << std::endl;
        return 1;
    }
    
    std::cout << "Parsed " << input_info.num_exprs << " expressions in " 
              << parse_time << " ms" << std::endl;
    
    // Compute total tokens
    int total_tokens = 0;
    for (int i = 0; i < input_info.num_exprs; i++) {
        total_tokens += input_info.num_tokens[i];
    }
    std::cout << "Total tokens: " << total_tokens << std::endl;
    
    // Get num_vars (all expressions share same data)
    int num_vars = input_info.num_vars[0];
    
    // Run subtree detection
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Running Common Subtree Detection..." << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    SubtreeDetectionResult result = detect_common_subtrees(
        input_info.num_exprs,
        num_vars,
        input_info.num_tokens,
        const_cast<const int**>(input_info.tokens),
        const_cast<const double**>(input_info.values),
        min_subtree_size,
        min_frequency
    );
    
    // Print summary
    print_subtree_detection_summary(result, false);
    
    // Print top 20 most frequent subtrees with details
    if (result.num_subs > 0) {
        int top_n = (result.num_subs < 20) ? result.num_subs : 20;
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "Top " << top_n << " Most Frequent Subtrees" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        for (int i = 0; i < top_n; i++) {
            std::cout << "\n" << (i + 1) << ". Subtree #" << i+1 << std::endl;
            std::cout << "   Size: " << result.num_sub_tokens[i] << " tokens" << std::endl;
            std::cout << "   Occurrences: " << result.num_occurrences[i] << std::endl;
            
            // Count unique expressions
            int unique_exprs = 0;
            bool* seen = new bool[input_info.num_exprs]();
            for (int j = 0; j < result.num_occurrences[i]; j++) {
                int expr_id = result.sub_occ_expr[i][j];
                if (!seen[expr_id]) {
                    seen[expr_id] = true;
                    unique_exprs++;
                }
            }
            delete[] seen;
            std::cout << "   Unique expressions: " << unique_exprs << std::endl;
            
            std::cout << "   Token sequence: [";
            for (int j = 0; j < result.num_sub_tokens[i]; j++) {
                if (j > 0) std::cout << ", ";
                std::cout << result.sub_tokens[i][j];
            }
            std::cout << "]" << std::endl;
            
            // Show first 5 occurrences
            int show_occ = (result.num_occurrences[i] < 5) ? result.num_occurrences[i] : 5;
            std::cout << "   First " << show_occ << " occurrences:" << std::endl;
            for (int j = 0; j < show_occ; j++) {
                std::cout << "     - Expression " << result.sub_occ_expr[i][j]
                         << " at position " << result.sub_occ_idx[i][j] << std::endl;
            }
            if (result.num_occurrences[i] > 5) {
                std::cout << "     ... and " << (result.num_occurrences[i] - 5) << " more" << std::endl;
            }
        }
        std::cout << "\n" << std::string(60, '=') << std::endl;
        
        // Compute statistics
        int total_occurrences = 0;
        int total_subtree_tokens = 0;
        for (int i = 0; i < result.num_subs; i++) {
            total_occurrences += result.num_occurrences[i];
            total_subtree_tokens += result.num_sub_tokens[i] * result.num_occurrences[i];
        }
        
        std::cout << "\nStatistics:" << std::endl;
        std::cout << "  Total common subtrees found: " << result.num_subs << std::endl;
        std::cout << "  Total occurrences: " << total_occurrences << std::endl;
        std::cout << "  Total tokens in common subtrees: " << total_subtree_tokens << std::endl;
        std::cout << "  Coverage: " << (100.0 * total_subtree_tokens / total_tokens) << "%" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
    }
    
    // Clean up
    free_subtree_detection_result(result);
    free_input_info(input_info);
    
    return 0;
}
