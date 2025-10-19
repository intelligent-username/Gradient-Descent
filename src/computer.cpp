// Helper for computing gradients
// i.e. plug in weights into a given gradient of a loss function

#include "computer.hpp"
#include <stdexcept>
#include <string>

using std::invalid_argument;
using std::string;

// Compute gradient application given a gradient Tensor and input points Tensor
// Supports:
// - scalar gradient (1x1) -> scales points
// - matrix-matrix product when inner dims match
// - element-wise product when shapes match exactly
Tensor computeGradient(const Tensor& gradient, const Tensor& points) {
    // scalar scaling
    if (gradient.rows() == 1 && gradient.cols() == 1) {
        return points * gradient.data(0, 0);
    }

    // matrix multiplication when feasible
    if (gradient.cols() == points.rows()) {
        return Tensor(gradient.data * points.data);
    }

    // element-wise when same shape
    if (gradient.rows() == points.rows() && gradient.cols() == points.cols()) {
        return Tensor(gradient.data.cwiseProduct(points.data));
    }

    throw invalid_argument("Incompatible tensor shapes for gradient application");
}


// Gradient to use for parameter updates
// Will plug weights into this gradient
// Might extend to finding custom loss functions in the future

Tensor gradientFinder(const string& lossType) {
    if (lossType == "MSE") {
        // Placeholder: return scalar 2.0 which often appears in d/dw of MSE
        Tensor g(1, 1);
        g.data(0, 0) = 2.0;
        return g;
    } else if (lossType == "MAE") {
        // Placeholder: sign-like scalar 1.0, actual depends on residual sign
        Tensor g(1, 1);
        g.data(0, 0) = 1.0;
        return g;
    } else if (lossType == "Hinge") {
        // Placeholder scalar; true gradient depends on margin condition
        Tensor g(1, 1);
        g.data(0, 0) = -1.0;
        return g;
    } else if (lossType == "NLL") {
        // Placeholder scalar for negative log-likelihood derivative scale
        Tensor g(1, 1);
        g.data(0, 0) = 1.0;
        return g;
    } else if (lossType == "cos") {
        // Placeholder scalar for cosine similarity loss derivative scale
        Tensor g(1, 1);
        g.data(0, 0) = -1.0;
        return g;
    } else {
        throw invalid_argument("Unknown loss type: " + lossType);
    }
}


