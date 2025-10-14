#pragma once
#include "tensor.hpp"
#include <vector>
#include <string>
#include <cmath>
#include <utility>

using namespace std;

// Batch type enum for clarity
enum class BatchType {
    FULL,
    MINI,
    STOCHASTIC
};

// Returns a pair of (X_batch, y_batch)
pair<Tensor, vector<Tensor*>> getBatch(const Tensor& X, const vector<Tensor*>& y, BatchType mode, int& batchStart, int batchSize);

// Helper to compute appropriate batch size
int computeBatchSize(int n, BatchType mode);
