#include "detect.h"
#include "utils.h"
#include <iostream>
#include <cstring>
#include <map>
#include <vector>
#include <algorithm>

// Simple arity function: token < 10 -> binary (arity 2), >= 10 -> unary (arity 1)
static int simple_op_arity(int token) {
    if (token < 10) return 2;  // Binary operators
    return 1;                   // Unary operators
}

// Compute size of subtree starting at position start_idx (prefix notation)
// Returns number of tokens in the subtree
static int compute_subtree_size(const int* tokens, int start_idx, int total_tokens) {
    if (start_idx >= total_tokens) return 0;
    
    int token = tokens[start_idx];
    
    // Variable or constant (negative token)
    if (token < 0) {
        return 1;
    }
    
    // Operator
    int arity = simple_op_arity(token);
    int size = 1;  // Count the operator itself
    int pos = start_idx + 1;
    
    // Add sizes of all operands
    for (int i = 0; i < arity; i++) {
        if (pos >= total_tokens) break;
        int operand_size = compute_subtree_size(tokens, pos, total_tokens);
        size += operand_size;
        pos += operand_size;
    }
    
    return size;
}

// Compute hash for token+value sequence
static uint64_t compute_hash(const int* tokens, const double* values, int length) {
    // Simple FNV-1a hash
    uint64_t hash = 14695981039346656037ULL;
    const uint64_t prime = 1099511628211ULL;
    
    for (int i = 0; i < length; i++) {
        // Hash token
        hash ^= (uint64_t)tokens[i];
        hash *= prime;
        
        // Hash value (convert double to uint64_t)
        uint64_t val_bits;
        memcpy(&val_bits, &values[i], sizeof(uint64_t));
        hash ^= val_bits;
        hash *= prime;
    }
    
    return hash;
}

// Check if two subtrees match exactly
static bool subtrees_match(
    const int* tokens1, const double* values1,
    const int* tokens2, const double* values2,
    int length)
{
    for (int i = 0; i < length; i++) {
        if (tokens1[i] != tokens2[i]) return false;
        if (values1[i] != values2[i]) return false;
    }
    return true;
}

// Candidate subtree occurrence
struct SubtreeCandidate {
    int expr_id;
    int start_idx;
    int size;
};

SubtreeDetectionResult detect_common_subtrees(
    int num_exprs,
    int num_vars,
    const int* num_tokens,
    const int** tokens,
    const double** values,
    int min_subtree_size,
    int min_frequency)
{
    TimePoint start = measure_clock();
    
    SubtreeDetectionResult result;
    result.num_exprs = num_exprs;
    
    // Allocate and initialize expr_sub_hints (same shape as tokens)
    result.expr_sub_hints = new int*[num_exprs];
    for (int expr_id = 0; expr_id < num_exprs; expr_id++) {
        result.expr_sub_hints[expr_id] = new int[num_tokens[expr_id]];
        for (int i = 0; i < num_tokens[expr_id]; i++) {
            result.expr_sub_hints[expr_id][i] = -1;
        }
    }
    
    // Step 1: Extract all subtrees and hash them
    // map: hash -> vector of candidates
    std::map<uint64_t, std::vector<SubtreeCandidate>> hash_map;
    
    for (int expr_id = 0; expr_id < num_exprs; expr_id++) {
        for (int start_idx = 0; start_idx < num_tokens[expr_id]; start_idx++) {
            // Compute subtree size
            int subtree_size = compute_subtree_size(tokens[expr_id], start_idx, num_tokens[expr_id]);
            
            // Skip singletons and subtrees smaller than minimum
            if (subtree_size < 2 || subtree_size < min_subtree_size) continue;
            
            // Compute hash
            uint64_t hash = compute_hash(
                tokens[expr_id] + start_idx,
                values[expr_id] + start_idx,
                subtree_size
            );
            
            // Add to hash map
            SubtreeCandidate cand;
            cand.expr_id = expr_id;
            cand.start_idx = start_idx;
            cand.size = subtree_size;
            hash_map[hash].push_back(cand);
        }
    }
    
    // Step 2: Sort hashes by occurrence count (most frequent first)
    std::vector<std::pair<uint64_t, std::vector<SubtreeCandidate>>> sorted_buckets;
    for (auto& kv : hash_map) {
        if ((int)kv.second.size() >= min_frequency) {
            sorted_buckets.push_back(kv);
        }
    }
    std::sort(sorted_buckets.begin(), sorted_buckets.end(),
              [](const std::pair<uint64_t, std::vector<SubtreeCandidate>>& a,
                 const std::pair<uint64_t, std::vector<SubtreeCandidate>>& b) {
                  return a.second.size() > b.second.size();  // Most frequent first
              });
    
    // Step 3: Verify matches and assign subtree IDs
    std::vector<std::vector<SubtreeCandidate>> verified_subtrees;
    
    for (auto& bucket : sorted_buckets) {
        std::vector<SubtreeCandidate>& candidates = bucket.second;
        
        // Verify all candidates actually match (compare to first candidate)
        std::vector<SubtreeCandidate> verified;
        
        const SubtreeCandidate& ref = candidates[0];
        const int* ref_tokens = tokens[ref.expr_id] + ref.start_idx;
        const double* ref_values = values[ref.expr_id] + ref.start_idx;
        
        for (const auto& cand : candidates) {
            // Skip if this position is already part of a larger common subtree
            if (result.expr_sub_hints[cand.expr_id][cand.start_idx] != -1) {
                continue;
            }
            
            // Verify match
            const int* cand_tokens = tokens[cand.expr_id] + cand.start_idx;
            const double* cand_values = values[cand.expr_id] + cand.start_idx;
            
            if (subtrees_match(ref_tokens, ref_values, cand_tokens, cand_values, cand.size)) {
                verified.push_back(cand);
            }
        }
        
        // Only keep if we still have enough occurrences
        if ((int)verified.size() >= min_frequency) {
            verified_subtrees.push_back(verified);
        }
    }
    
    // Step 4: Build result structure
    result.num_subs = verified_subtrees.size();
    
    if (result.num_subs > 0) {
        result.num_sub_tokens = new int[result.num_subs];
        result.sub_tokens = new int*[result.num_subs];
        result.sub_values = new double*[result.num_subs];
        result.num_occurrences = new int[result.num_subs];
        result.sub_occ_expr = new int*[result.num_subs];
        result.sub_occ_idx = new int*[result.num_subs];
        
        for (int sub_id = 0; sub_id < result.num_subs; sub_id++) {
            const auto& verified = verified_subtrees[sub_id];
            const SubtreeCandidate& ref = verified[0];
            
            // Store subtree tokens and values
            result.num_sub_tokens[sub_id] = ref.size;
            result.sub_tokens[sub_id] = new int[ref.size];
            result.sub_values[sub_id] = new double[ref.size];
            
            for (int i = 0; i < ref.size; i++) {
                result.sub_tokens[sub_id][i] = tokens[ref.expr_id][ref.start_idx + i];
                result.sub_values[sub_id][i] = values[ref.expr_id][ref.start_idx + i];
            }
            
            // Store occurrences
            result.num_occurrences[sub_id] = verified.size();
            result.sub_occ_expr[sub_id] = new int[verified.size()];
            result.sub_occ_idx[sub_id] = new int[verified.size()];
            
            for (size_t occ = 0; occ < verified.size(); occ++) {
                result.sub_occ_expr[sub_id][occ] = verified[occ].expr_id;
                result.sub_occ_idx[sub_id][occ] = verified[occ].start_idx;
                
                // Mark in expr_sub_hints (1-based subtree ID)
                int subtree_id_1based = sub_id + 1;
                result.expr_sub_hints[verified[occ].expr_id][verified[occ].start_idx] = subtree_id_1based;
                
                // Mark all positions in this subtree to skip deeper subtrees
                for (int i = 1; i < verified[occ].size; i++) {
                    result.expr_sub_hints[verified[occ].expr_id][verified[occ].start_idx + i] = subtree_id_1based;
                }
            }
        }
    } else {
        result.num_sub_tokens = nullptr;
        result.sub_tokens = nullptr;
        result.sub_values = nullptr;
        result.num_occurrences = nullptr;
        result.sub_occ_expr = nullptr;
        result.sub_occ_idx = nullptr;
    }
    
    result.analysis_time_ms = clock_to_ms(start, measure_clock());
    
    return result;
}

void free_subtree_detection_result(SubtreeDetectionResult& result)
{
    if (result.num_sub_tokens != nullptr) {
        delete[] result.num_sub_tokens;
        result.num_sub_tokens = nullptr;
    }
    
    if (result.sub_tokens != nullptr) {
        for (int i = 0; i < result.num_subs; i++) {
            if (result.sub_tokens[i] != nullptr) {
                delete[] result.sub_tokens[i];
            }
        }
        delete[] result.sub_tokens;
        result.sub_tokens = nullptr;
    }
    
    if (result.sub_values != nullptr) {
        for (int i = 0; i < result.num_subs; i++) {
            if (result.sub_values[i] != nullptr) {
                delete[] result.sub_values[i];
            }
        }
        delete[] result.sub_values;
        result.sub_values = nullptr;
    }
    
    if (result.num_occurrences != nullptr) {
        delete[] result.num_occurrences;
        result.num_occurrences = nullptr;
    }
    
    if (result.sub_occ_expr != nullptr) {
        for (int i = 0; i < result.num_subs; i++) {
            if (result.sub_occ_expr[i] != nullptr) {
                delete[] result.sub_occ_expr[i];
            }
        }
        delete[] result.sub_occ_expr;
        result.sub_occ_expr = nullptr;
    }
    
    if (result.sub_occ_idx != nullptr) {
        for (int i = 0; i < result.num_subs; i++) {
            if (result.sub_occ_idx[i] != nullptr) {
                delete[] result.sub_occ_idx[i];
            }
        }
        delete[] result.sub_occ_idx;
        result.sub_occ_idx = nullptr;
    }
    
    if (result.expr_sub_hints != nullptr) {
        for (int i = 0; i < result.num_exprs; i++) {
            if (result.expr_sub_hints[i] != nullptr) {
                delete[] result.expr_sub_hints[i];
            }
        }
        delete[] result.expr_sub_hints;
        result.expr_sub_hints = nullptr;
    }
    
    result.num_subs = 0;
    result.num_exprs = 0;
}

void print_subtree_detection_summary(const SubtreeDetectionResult& result, bool verbose)
{
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Subtree Detection Summary" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "Expressions analyzed: " << result.num_exprs << std::endl;
    std::cout << "Common subtrees found: " << result.num_subs << std::endl;
    std::cout << "Analysis time: " << result.analysis_time_ms << " ms" << std::endl;
    
    if (result.num_subs > 0 && verbose) {
        std::cout << "\n" << std::string(60, '-') << std::endl;
        std::cout << "Subtree Details:" << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        
        for (int sub_id = 0; sub_id < result.num_subs; sub_id++) {
            std::cout << "\nSubtree #" << sub_id << ":" << std::endl;
            std::cout << "  Tokens: " << result.num_sub_tokens[sub_id] << std::endl;
            std::cout << "  Occurrences: " << result.num_occurrences[sub_id] << std::endl;
            
            // Print token sequence
            std::cout << "  Token sequence: [";
            for (int i = 0; i < result.num_sub_tokens[sub_id]; i++) {
                if (i > 0) std::cout << ", ";
                std::cout << result.sub_tokens[sub_id][i];
            }
            std::cout << "]" << std::endl;
            
            // Print occurrences
            std::cout << "  Found in:" << std::endl;
            for (int occ = 0; occ < result.num_occurrences[sub_id]; occ++) {
                std::cout << "    Expression " << result.sub_occ_expr[sub_id][occ]
                         << " at position " << result.sub_occ_idx[sub_id][occ] << std::endl;
            }
        }
    }
    
    std::cout << std::string(60, '=') << std::endl;
}
