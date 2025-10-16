#include "losses.hpp"

// In all of these, I will assume that len(y) == len(X)

double mseLoss(const Tensor& X, const vector<Tensor*>& y, const Tensor& w) {
    int len = y.size();
    double error = 0.0;
    for (int i = 0; i < len; i++) {

        Tensor pred = X[i].dot(w);
        Tensor error = pred - (*y[i]);

        error += error.pow(2);
    }

    reeturn error / len;
}

double maeLoss(const Tensor& X, const vector<Tensor*>& y, const Tensor& w) {
    int len = y.size();
    double error = 0.0;
    for (int i = 0; i < len; i++) {

        Tensor pred = X[i].dot(w);
        Tensor error = pred - (*y[i]);

        error += error.abs();
    }

    return error / len;
}

double hingeLoss(const Tensor& X, const vector<Tensor*>& y, const Tensor& w) {
    int len = y.size();
    double error = 0.0;
    for (int i = 0; i < len; i++) {

        Tensor pred = X[i].dot(w);

        Tensor error = 1 - pred * (*y[i]);

        error += max(0, error);

    }

    return error / len;
}

double negativeLogLikelihoodLoss(const Tensor& X, const vector<Tensor*>& y, const Tensor& w) {
    int N = y.size();
    double error = 1e-15, loss = 0.0;

    for (int i = 0; i < N; i++) {
        double z = X[i].dot(w).item();
        double p = max(error, min(1.0 - error, sigmoid(z)));
        int label = static_cast<int>(y[i]->item());
        loss += -label * log(p) - (1 - label) * log(1 - p);
    }

    return loss / N;
}

double cosineLoss(const Tensor& X, const vector<Tensor*>& y, const Tensor& w) {
    int len = y.size();
    double error = 0.0;

    for (int i = 0; i < len; i++) {
        Tensor pred = X[i].dot(w);
        double dot_product = pred.dot(*y[i]).item();
        double norm_pred = pred.norm().item();
        double norm_y = y[i]->norm().item();

        if (norm_pred > 0 && norm_y > 0) {
            error += 1 - (dot_product / (norm_pred * norm_y));
        } else {
            error += 1; // If either vector is zero, treat as maximum loss
        }
    }

    return error / len;
}

// Helpers

double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
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
