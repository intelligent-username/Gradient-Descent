#include "losses.hpp"

double mseLoss(const Tensor& X, const vector<Tensor*>& y, const Tensor& w) {
    return 0.0; 
}

double maeLoss(const Tensor& X, const vector<Tensor*>& y, const Tensor& w) {
    return 0.0;
}

double hingeLoss(const Tensor& X, const vector<Tensor*>& y, const Tensor& w) {
    return 0.0;
}

double negativeLogLikelihoodLoss(const Tensor& X, const vector<Tensor*>& y, const Tensor& w) {
    return 0.0;
}

double cosineLoss(const Tensor& X, const vector<Tensor*>& y, const Tensor& w) {
    return 0.0;
}

map<string, function<double(const Tensor&, const vector<Tensor*>&, const Tensor&)>> getLossFunctions() {
    return {
        {"MSE", mseLoss},
        {"MAE", maeLoss},
        {"Hinge", hingeLoss},
        {"NLL", negativeLogLikelihoodLoss},
        {"cos", cosineLoss}
    };
}
