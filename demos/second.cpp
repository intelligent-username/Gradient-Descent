// Minimal cubic polynomial regression demo that prints ONLY the learned weights

#include <cstdio>
#include <random>
#include <vector>
#include <Eigen/Dense>

#include "../include/gradientDescent.hpp"
#include "../include/tensor.hpp"
#include "../include/regression.hpp"

using namespace std;
using namespace Eigen;

int main() {
    printf("--------------------\n");
    printf("SECOND DEMO\n");
    printf("--------------------\n");

    const int n = 200;
    VectorXd x = VectorXd::LinSpaced(n, -3, 3);
    
    // Generate y = 2 + 3x - 2x² + 0.5x³ + noise
    VectorXd yv = 2 + 3*x.array() - 2*x.array().square() + 0.5*x.array().cube();

    mt19937 gen(42);
    normal_distribution<> noise(0.0, 0.3);
    for (int i = 0; i < n; ++i) yv(i) += noise(gen);

    Tensor X(n, 1);
    X.data.col(0) = x;

    vector<Tensor> y_storage; y_storage.reserve(n);
    vector<Tensor*> y_ptrs; y_ptrs.reserve(n);
    for (int i = 0; i < n; ++i) {
        Tensor t(1, 1);
        t.data(0, 0) = yv(i);
        y_storage.push_back(t);
        y_ptrs.push_back(&y_storage.back());
    }

    Result res = fitRegression(
        X, y_ptrs,
        RegressionType::POLYNOMIAL,
        3,
        "STOCHASTIC",
        "MSE",
        "Adam",
        1e-6,
        1000,
        8000,
        1e-7,
        0.0
    );

    printf("currently, the results are COMPLETELY OFF, that's because I need to tweak the hyperparameters. DO THIS (later).\n");
    // Print true weights then learned weights
    printf("True:   2, 3, -2, 0.5\n");
    printf("Learned: %.6f %.6f %.6f %.6f\n", 
           res.weights.data(0), res.weights.data(1),
           res.weights.data(2), res.weights.data(3));

    printf("--------------------\n");
    return 0;
}
