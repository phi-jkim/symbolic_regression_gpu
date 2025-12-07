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
    
    // Variable or constant (token <= 0)
    if (token <= 0) {
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
    uint64_t hash;
};

// Merkle-style hash combination
static uint64_t combine_hashes(uint64_t h1, uint64_t h2) {
    // Boost-like hash combine
    return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
}

static uint64_t hash_node(int token, double value, const std::vector<uint64_t>& child_hashes) {
    uint64_t h = 14695981039346656037ULL;
    const uint64_t prime = 1099511628211ULL;
    
    h ^= (uint64_t)token;
    h *= prime;
    
    uint64_t val_bits;
    memcpy(&val_bits, &value, sizeof(uint64_t));
    h ^= val_bits;
    h *= prime;
    
    for (uint64_t ch : child_hashes) {
        h = combine_hashes(h, ch);
    }
    return h;
}

SubtreeDetectionResult detect_common_subtrees(
    int num_exprs,
    int num_vars,
    int* num_tokens,
    const int** tokens,
    const double** values,
    int min_subtree_size,
    int min_frequency)
{
    TimePoint start = measure_clock();
    
    SubtreeDetectionResult result;
    result.num_exprs = num_exprs;
    
    // Allocate and initialize expr_sub_hints
    result.expr_sub_hints = new int*[num_exprs];
    for (int expr_id = 0; expr_id < num_exprs; expr_id++) {
        result.expr_sub_hints[expr_id] = new int[num_tokens[expr_id]];
        for (int i = 0; i < num_tokens[expr_id]; i++) {
            result.expr_sub_hints[expr_id][i] = -1;
        }
    }
    
    // Step 1: Linear pass to compute sizes and hashes
    // map: hash -> vector of candidates
    std::map<uint64_t, std::vector<SubtreeCandidate>> hash_map;
    
    // Pre-allocate vectors for per-expression info to avoid repeated allocs
    std::vector<int> sizes;
    std::vector<uint64_t> hashes;
    
    for (int expr_id = 0; expr_id < num_exprs; expr_id++) {
        int len = num_tokens[expr_id];
        const int* toks = tokens[expr_id];
        const double* vals = values[expr_id];
        
        sizes.resize(len);
        hashes.resize(len);
        
        // Reverse pass
        for (int i = len - 1; i >= 0; i--) {
            int token = toks[i];
            if (token <= 0) {
                // Terminal
                sizes[i] = 1;
                hashes[i] = hash_node(token, vals[i], {});
            } else {
                // Operator
                int arity = simple_op_arity(token);
                int size = 1;
                std::vector<uint64_t> child_hashes;
                int pos = i + 1;
                
                for (int k = 0; k < arity; k++) {
                    if (pos >= len) break; // Should not happen for valid prefix
                    size += sizes[pos];
                    child_hashes.push_back(hashes[pos]);
                    pos += sizes[pos];
                }
                sizes[i] = size;
                hashes[i] = hash_node(token, vals[i], child_hashes);
            }
            
            // Collect candidate
            if (sizes[i] >= min_subtree_size) {
                SubtreeCandidate cand;
                cand.expr_id = expr_id;
                cand.start_idx = i;
                cand.size = sizes[i];
                cand.hash = hashes[i];
                hash_map[cand.hash].push_back(cand);
            }
        }
    }
    
    // Step 2: Filter and Verify
    std::vector<std::vector<SubtreeCandidate>> verified_subtrees;
    
    for (auto& kv : hash_map) {
        if ((int)kv.second.size() < min_frequency) continue;
        
        std::vector<SubtreeCandidate>& candidates = kv.second;
        
        // Verify against first candidate (to handle hash collisions)
        // Note: Merkle hash is strong, but we still verify token sequence match
        // which is O(Size).
        
        const SubtreeCandidate& ref = candidates[0];
        const int* ref_tokens = tokens[ref.expr_id] + ref.start_idx;
        const double* ref_values = values[ref.expr_id] + ref.start_idx;
        
        std::vector<SubtreeCandidate> verified;
        verified.push_back(ref);
        
        for (size_t k = 1; k < candidates.size(); k++) {
            const auto& cand = candidates[k];
            // Quick size check (should match if hash matches, but good sanity)
            if (cand.size != ref.size) continue;
            
            const int* cand_tokens = tokens[cand.expr_id] + cand.start_idx;
            const double* cand_values = values[cand.expr_id] + cand.start_idx;
            
            if (subtrees_match(ref_tokens, ref_values, cand_tokens, cand_values, cand.size)) {
                verified.push_back(cand);
            }
        }
        
        if ((int)verified.size() >= min_frequency) {
            verified_subtrees.push_back(verified);
        }
    }
    
    // Step 3: Sort by size descending
    std::sort(verified_subtrees.begin(), verified_subtrees.end(),
              [](const std::vector<SubtreeCandidate>& a,
                 const std::vector<SubtreeCandidate>& b) {
                  return a[0].size > b[0].size;
              });
    
    // Step 4: Build result
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
                
                // Check overlap
                bool overlap = false;
                for (int k = 0; k < ref.size; k++) {
                    if (result.expr_sub_hints[verified[occ].expr_id][verified[occ].start_idx + k] != -1) {
                        overlap = true;
                        break;
                    }
                }
                
                if (!overlap) {
                    // Mark root
                    result.expr_sub_hints[verified[occ].expr_id][verified[occ].start_idx] = sub_id;
                    // Mark body
                    for (int k = 1; k < ref.size; k++) {
                        result.expr_sub_hints[verified[occ].expr_id][verified[occ].start_idx + k] = -2;
                    }
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

// Stateful detection: checks cache first, adds new common subtrees
SubtreeDetectionResult detect_and_update_cache(
    SubtreeCache& cache,
    int num_exprs,
    int num_features,
    int* num_tokens,
    const int** tokens,
    const double** values,
    int min_size,
    int min_freq
) {
    SubtreeDetectionResult result;
    result.num_exprs = num_exprs;
    result.num_subs = 0; // Will be updated
    result.sub_tokens = nullptr; // Not used in result for stateful (cache holds them)
    result.sub_values = nullptr;
    result.num_sub_tokens = nullptr;
    result.sub_occ_expr = nullptr; // Not strictly needed for eval, but could be useful
    result.sub_occ_idx = nullptr;
    result.num_occurrences = nullptr;

    // Allocate hints
    result.expr_sub_hints = new int*[num_exprs];
    for (int i = 0; i < num_exprs; i++) {
        result.expr_sub_hints[i] = new int[num_tokens[i]];
        // Initialize with -1 (no hint)
        for (int j = 0; j < num_tokens[i]; j++) result.expr_sub_hints[i][j] = -1;
    }

    // Step 1: Hash all subtrees in current batch
    // map: hash -> vector of candidates
    std::map<uint64_t, std::vector<SubtreeCandidate>> hash_map;

    for (int i = 0; i < num_exprs; i++) {
        for (int j = 0; j < num_tokens[i]; j++) {
            int size = compute_subtree_size(tokens[i], j, num_tokens[i]);
            if (size >= min_size) {
                uint64_t h = compute_hash(tokens[i], values[i], size);
                hash_map[h].push_back({i, j, size, h});
            }
        }
    }

    // Step 2: Identify subtrees (both cached and new)
    // We need to map local sub_id (for this batch) to cache_id
    // But wait, the evaluator needs to know which cache index to use.
    // So hints should store the CACHE ID directly.
    
    // Check existing cache hits
    // int cache_hits = 0;
    // int new_subs = 0;

    // For new candidates, we need to filter by frequency
    for (auto& pair : hash_map) {
        uint64_t h = pair.first;
        std::vector<SubtreeCandidate>& candidates = pair.second;

        int cache_id = -1;
        bool in_cache = false;

        // Check if in cache
        auto it = cache.hash_to_id.find(h);
        if (it != cache.hash_to_id.end()) {
            cache_id = it->second;
            in_cache = true;
            // cache_hits++;
        } else {
            // Not in cache, check frequency
            if ((int)candidates.size() >= min_freq) {
                // Add to cache!
                cache_id = (int)cache.results.size();
                cache.hash_to_id[h] = cache_id;
                cache.results.push_back(nullptr); // Placeholder, will be allocated by evaluator
                cache.ref_counts.push_back(0);
                
                // Store structure for verification/debugging
                int size = candidates[0].size;
                cache.sub_sizes.push_back(size);
                
                int* toks = new int[size];
                double* vals = new double[size];
                int expr_id = candidates[0].expr_id;
                int start_idx = candidates[0].start_idx;
                
                for(int k=0; k<size; k++) {
                    toks[k] = tokens[expr_id][start_idx + k];
                    vals[k] = values[expr_id][start_idx + k];
                }
                cache.sub_tokens.push_back(toks);
                cache.sub_values.push_back(vals);
                
                in_cache = true;
                // new_subs++;
            }
        }

        if (in_cache) {
            // Mark hints for all occurrences
            // We need to be careful about overlapping subtrees.
            // Larger subtrees should take precedence? Or maybe just first found?
            // The original logic sorted by size descending. We should do the same.
            // But we are iterating by hash...
            
            // Let's defer marking until we have processed all hashes?
            // Or just mark now and handle overlaps?
            // The original logic: "Sort verified subtrees by size (descending)"
            
            // Current approach: We are iterating map, order is by hash (randomish).
            // Better: Collect all valid (cached or new) candidates, sort by size, then mark.
        }
    }
    
    // Re-scan to collect all valid candidates (cached or new-high-freq)
    std::vector<std::vector<SubtreeCandidate>> valid_groups;
    
    for (auto& pair : hash_map) {
        uint64_t h = pair.first;
        if (cache.hash_to_id.count(h)) {
             valid_groups.push_back(pair.second);
        }
    }
    
    // Sort groups by subtree size (descending)
    std::sort(valid_groups.begin(), valid_groups.end(), [](const std::vector<SubtreeCandidate>& a, const std::vector<SubtreeCandidate>& b) {
        return a[0].size > b[0].size;
    });
    
    // Mark hints
    for (const auto& group : valid_groups) {
        uint64_t h = group[0].hash;
        int cache_id = cache.hash_to_id[h];
        
        for (const auto& cand : group) {
            // Check if already marked (part of a larger subtree)
            bool overlap = false;
            for (int k = 0; k < cand.size; k++) {
                if (result.expr_sub_hints[cand.expr_id][cand.start_idx + k] != -1) {
                    overlap = true;
                    break;
                }
            }
            
            if (!overlap) {
                // Mark root with cache_id
                result.expr_sub_hints[cand.expr_id][cand.start_idx] = cache_id;
                // Mark body with -2
                for (int k = 1; k < cand.size; k++) {
                    result.expr_sub_hints[cand.expr_id][cand.start_idx + k] = -2;
                }
                
                // Update ref count
                cache.ref_counts[cache_id]++;
            }
        }
    }
    
    // We don't populate result.sub_tokens etc because the evaluator will use the cache directly.
    // But we need to tell the evaluator which cache entries are NEW and need computation.
    // Actually, the evaluator can check if cache.results[id] is nullptr.
    
    result.num_subs = (int)cache.results.size(); // Total subtrees in cache
    
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
