#include <vector>
#include <cmath>
#include <algorithm>

#include "transformer.hpp"

using namespace std;
using namespace Eigen;

// Helper to generate combinations
void generateCombinations(std::vector<std::vector<int>>& result, std::vector<int>& current, 
                          int start, int n, int remaining) {
    if (remaining == 0) {
        result.push_back(current);
        return;
    }
    
    for (int i = start; i < n; i++) {
        current.push_back(i);
        generateCombinations(result, current, i, n, remaining - 1);
        current.pop_back();
    }
}

Tensor PolynomialTransformer::transform(const Tensor& X) const {
    int n_samples = X.rows();
    int n_features = X.cols();
    
    // Calculate output features and combinations
    std::vector<std::vector<std::vector<int>>> combinations_by_degree;
    int output_features = 0;
    
    for (int d = 0; d <= degree; d++) {
        std::vector<std::vector<int>> combinations;
        std::vector<int> current;
        generateCombinations(combinations, current, 0, n_features, d);
        combinations_by_degree.push_back(combinations);
        output_features += combinations.size();
    }
    
    // Skip bias if not included
    int start_idx = include_bias ? 0 : 1;
    if (!include_bias) output_features--;
    
    // Create output tensor
    Tensor X_poly(n_samples, output_features);
    
    // Fill output tensor with transformed features
    int col_idx = 0;
    for (int d = start_idx; d <= degree; d++) {
        for (const auto& combo : combinations_by_degree[d]) {
            if (combo.empty() && !include_bias) continue;
            
            if (combo.empty()) {
                // Bias term (all 1s)
                X_poly.data.col(col_idx).setOnes();
            } else {
                // Initialize with 1s
                X_poly.data.col(col_idx).setOnes();
                
                // Multiply features according to combination
                for (int feat_idx : combo) {
                    for (int i = 0; i < n_samples; i++) {
                        X_poly.data(i, col_idx) *= X.data(i, feat_idx);
                    }
                }
            }
            col_idx++;
        }
    }
    
    return X_poly;
}

int suggestPolynomialDegree(int n_samples, int n_features, int max_degree) {
    // Limit features to <= ~1/3 of samples
    const int max_allowed_features = n_samples / 3;

    // Precompute factorials to avoid repeated computation
    vector<double> factorial(max_degree + 2, 1.0);
    for (int i = 1; i <= max_degree + 1; ++i)
        factorial[i] = factorial[i - 1] * i;

    for (int d = 1; d <= max_degree; ++d) {
        // Estimated number of polynomial features: binomial(n_features + d, d)
        double estimated_features = 1.0;
        for (int i = 0; i < d; ++i)
            estimated_features *= (n_features + i + 1) / static_cast<double>(i + 1);

        if (estimated_features > max_allowed_features)
            return max(1, d - 1);
    }

    return max_degree; // Safe if all degrees acceptable
}
