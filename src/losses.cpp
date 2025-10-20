#include "losses.hpp"
#include <algorithm>
#include <cmath>

using std::function;
using std::map;
using std::string;

// Helper: build y_true (n x 1) from vector<Tensor*>
static Tensor buildYTrue(const std::vector<Tensor*>& y_vec) {
    Tensor y_true(static_cast<int>(y_vec.size()), 1);
    for (int i = 0; i < static_cast<int>(y_vec.size()); ++i)
        y_true.data(i, 0) = y_vec[i]->data(0, 0);
    return y_true;
}

// All losses expect first arg as predictions y_hat (n x 1)

double mseLoss(const Tensor& y_hat, const std::vector<Tensor*>& y, const Tensor& /*w*/) {
    Tensor y_true = buildYTrue(y);
    Tensor diff = y_hat - y_true;
    double sse = diff.data.array().square().sum();
    return sse / static_cast<double>(y_true.rows());
}

double maeLoss(const Tensor& y_hat, const std::vector<Tensor*>& y, const Tensor& /*w*/) {
    Tensor y_true = buildYTrue(y);
    Tensor diff = y_hat - y_true;
    double sae = diff.data.array().abs().sum();
    return sae / static_cast<double>(y_true.rows());
}

double hingeLoss(const Tensor& y_hat, const std::vector<Tensor*>& y, const Tensor& /*w*/) {
    // y in {-1, +1}
    Tensor y_true = buildYTrue(y);
    Eigen::MatrixXd margin = Eigen::MatrixXd::Ones(y_true.rows(), y_true.cols()) - y_true.data.cwiseProduct(y_hat.data);
    double sum = margin.array().max(0.0).sum();
    return sum / static_cast<double>(y_true.rows());
}

double negativeLogLikelihoodLoss(const Tensor& y_hat, const std::vector<Tensor*>& y, const Tensor& /*w*/) {
    // Binary cross-entropy on logits: p = sigmoid(y_hat)
    Tensor y_true = buildYTrue(y);
    const double eps = 1e-15;
    Eigen::ArrayXXd p = 1.0 / (1.0 + (-y_hat.data.array()).exp());
    p = p.min(1.0 - eps).max(eps);
    double loss = -((y_true.data.array() * p.log()) + ((1.0 - y_true.data.array()) * (1.0 - p).log())).sum();
    return loss / static_cast<double>(y_true.rows());
}

double cosineLoss(const Tensor& y_hat, const std::vector<Tensor*>& y, const Tensor& /*w*/) {
    Tensor y_true = buildYTrue(y);
    double dot = (y_hat.data.cwiseProduct(y_true.data)).sum();
    double n1 = y_hat.data.norm();
    double n2 = y_true.data.norm();
    if (n1 == 0.0 || n2 == 0.0) return 1.0;
    return 1.0 - (dot / (n1 * n2));
}

map<string, function<double(const Tensor&, const std::vector<Tensor*>&, const Tensor&)>> getLossFunctions() {
    return {
        {"MSE", mseLoss},
        {"MAE", maeLoss},
        {"Hinge", hingeLoss},
        {"NLL", negativeLogLikelihoodLoss},
        {"cos", cosineLoss}
    };
}
