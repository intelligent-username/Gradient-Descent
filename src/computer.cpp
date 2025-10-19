#include "computer.hpp"
#include <stdexcept>
#include <string>
#include <functional>

using std::invalid_argument;
using std::string;
using std::function;

// computeGradient now evaluates the precomputed gradient at a specific point
Tensor computeGradient(const function<Tensor(const Tensor&, const Tensor&)>& gradFunc,
                       const Tensor& predictions,
                       const Tensor& targets) {
    return gradFunc(predictions, targets);
}


// gradientFinder returns a callable that computes ∂L/∂ŷ for given predictions, targets
function<Tensor(const Tensor&, const Tensor&)> gradientFinder(const string& lossType) {
    if (lossType == "MSE") {
        // dL/dŷ = (2/n) * (ŷ - y)
        return [](const Tensor& y_hat, const Tensor& y_true) {
            Tensor diff = y_hat - y_true;
            double scale = 2.0 / static_cast<double>(y_hat.rows());
            return diff * scale;
        };
    }

    if (lossType == "MAE") {
        // dL/dŷ = sign(ŷ - y) / n
        return [](const Tensor& y_hat, const Tensor& y_true) {
            Tensor diff = y_hat - y_true;
            Tensor sign = diff.sign();  // assumes Tensor::sign() exists
            double scale = 1.0 / static_cast<double>(y_hat.rows());
            return sign * scale;
        };
    }

    if (lossType == "Hinge") {
        // dL/dŷ = -y if (1 - y*ŷ) > 0 else 0
        return [](const Tensor& y_hat, const Tensor& y_true) {
            Tensor margin = Tensor::Ones(y_true.rows(), y_true.cols()) - y_true.cwiseProduct(y_hat);
            Tensor grad = Tensor::ZeroLike(y_hat);
            for (int i = 0; i < grad.rows(); ++i)
                for (int j = 0; j < grad.cols(); ++j)
                    grad.data(i, j) = (margin.data(i, j) > 0.0) ? -y_true.data(i, j) : 0.0;
            return grad;
        };
    }

    if (lossType == "NLL") {
        // dL/dŷ = -1 / ŷ
        return [](const Tensor& y_hat, const Tensor& /*y_true*/) {
            Tensor grad = Tensor::ZeroLike(y_hat);
            grad.data = (-1.0 * y_hat.data.array().inverse()).matrix();
            return grad;
        };
    }

    if (lossType == "cos") {
        // dL/dŷ = derivative of cosine similarity loss L = 1 - (ŷ·y)/(||ŷ|| ||y||)
        return [](const Tensor& y_hat, const Tensor& y_true) {
            double dot = (y_hat.data.cwiseProduct(y_true.data)).sum();
            double norm_yhat = y_hat.data.norm();
            double norm_ytrue = y_true.data.norm();
            Tensor term1 = y_hat * (-dot / (norm_yhat * norm_yhat * norm_ytrue));
            Tensor term2 = y_true * (-1.0 / (norm_yhat * norm_ytrue));
            return term1 + term2;
        };
    }

    throw invalid_argument("Unknown loss type: " + lossType);
}
