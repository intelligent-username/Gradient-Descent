// Minimal cubic polynomial regression demo that prints ONLY the learned weights

#include <cstdio>
#include <random>
#include <vector>
#include <Eigen/Dense>
#include <chrono>

#include "../include/gradientDescent.hpp"
#include "../include/tensor.hpp"
#include "../include/regression.hpp"

using namespace std;
using namespace Eigen;

int main() {
    printf("\n\n");
    printf("SECOND DEMO: one feature but degree 3\n");
    printf("--------------------");
    const int n = 200;
    VectorXd x = VectorXd::LinSpaced(n, -3, 3);
    
    // Generate y = 2 + 3x - 2x² + 0.5x³ + noise
    VectorXd yv = 2 + 3*x.array() - 2*x.array().square() + 0.5*x.array().cube();

    unsigned int seed = static_cast<unsigned int>(chrono::system_clock::now().time_since_epoch().count());
    mt19937 gen(seed);
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
        RegressionType::POLYNOMIAL,       // regression type
        3,                                // degree
        "STOCHASTIC",                     // BatchMode
        "MSE",                            // Loss
        "Adam",                           // Optimization
        1e-8,                             // Minimum gradient
        2000,                              // Max epochs
        20000,                             // Max iterations
        1e-7,                             // Loss difference
        0.0                               // Minimum loss (how 'accurate' we want it to current data)
    );

    printf("\n\nStill need to tune functions & hyperparams.\n");
    // Print true weights then learned weights
    printf("Final loss: %.6f\n", res.loss);
    printf("Epochs: %d\n", res.epochs);
    printf("True:   2, 3, -2, 0.5\n");
    printf("Learned: %.6f %.6f %.6f %.6f\n", 
           res.weights.data(0), res.weights.data(1),
           res.weights.data(2), res.weights.data(3));

    return 0;
}
