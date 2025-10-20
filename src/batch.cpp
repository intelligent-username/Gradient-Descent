// Batching

#include "batch.hpp"

#include <algorithm>

using namespace std;

// Compute batch size dynamically based on mode
int computeBatchSize(int n, BatchType mode) {
    switch (mode) {
        case BatchType::FULL:
            return n; // Use the entire dataset
        case BatchType::STOCHASTIC:
            return 1; // Single sample per iteration
        case BatchType::MINI:
        default:
            // Compute based on log or a fraction of total size
            return max(1, n / 100 > 0 ? n / 100 : static_cast<int>(log(n)));
    }
}

// Get the next batch based on mode
pair<Tensor, vector<Tensor*>> getBatch(const Tensor& X, const vector<Tensor*>& y, BatchType mode, int& batchStart, int batchSize) {
    int n = X.rows();

    if (mode == BatchType::FULL) {
        return {X, y};
    }

    int start = batchStart;
    int end = min(start + batchSize, n);

    Tensor X_batch = X.block(start, 0, end - start, X.cols());
    vector<Tensor*> y_batch(y.begin() + start, y.begin() + end);

    // Update batchStart for next iteration
    batchStart = end;
    if (batchStart >= n) {
        batchStart = 0; // Reset after one full pass (epoch)
    }

    return {X_batch, y_batch};
}
