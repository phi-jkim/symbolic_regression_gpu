#ifndef DETECT_H
#define DETECT_H

// ============================================================================
// Common Subtree Detection
// ============================================================================

// Result of common subtree detection across multiple expressions
typedef struct {
    // Number of common subtrees found
    int num_subs;
    
    // Per-subtree information
    int* num_sub_tokens;        // Array[num_subs]: number of tokens in each subtree
    int** sub_tokens;           // Array[num_subs][num_sub_tokens[i]]: token sequence for each subtree
    double** sub_values;        // Array[num_subs][num_sub_tokens[i]]: value sequence for each subtree
    
    // Occurrence information
    int* num_occurrences;       // Array[num_subs]: how many times each subtree appears
    int** sub_occ_expr;         // Array[num_subs][num_occurrences[i]]: which expression each occurrence is in
    int** sub_occ_idx;          // Array[num_subs][num_occurrences[i]]: starting token index in that expression
    
    // Per-expression hints (same shape as input tokens/values)
    // expr_sub_hints[expr_id][token_idx] = subtree_id if a subtree starts at this position, -1 otherwise
    int** expr_sub_hints;       // Array[num_exprs][num_tokens[expr_id]]: subtree ID at each position or -1
    
    // Metadata
    int num_exprs;              // Number of expressions analyzed
    double analysis_time_ms;    // Time taken for detection
} SubtreeDetectionResult;

// Detect common subtrees across multiple expressions
// 
// Inputs:
//   num_exprs:    Number of expressions to analyze
//   num_vars:     Number of variables (same for all expressions)
//   num_tokens:   Array[num_exprs] - number of tokens per expression
//   tokens:       Array[num_exprs][num_tokens[i]] - token arrays
//   values:       Array[num_exprs][num_tokens[i]] - value arrays
//   min_subtree_size: Minimum number of tokens to consider as a subtree (default: 3)
//   min_frequency:    Minimum occurrences to report as common (default: 2)
//
// Returns:
//   SubtreeDetectionResult with all allocated memory
//   Caller is responsible for freeing using free_subtree_detection_result()
//
SubtreeDetectionResult detect_common_subtrees(
    int num_exprs,
    int num_vars,
    const int* num_tokens,
    const int** tokens,
    const double** values,
    int min_subtree_size = 3,
    int min_frequency = 2
);

// Free all memory allocated by detect_common_subtrees
void free_subtree_detection_result(SubtreeDetectionResult& result);

// Print summary of detection results
void print_subtree_detection_summary(const SubtreeDetectionResult& result, bool verbose = false);

#endif // DETECT_H
