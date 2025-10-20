// This one just has random, synthetic data.
// Upcoming ones will use real data and compare to expected outputs

#include <iostream>
#include <random>
#include <vector>
#include <Eigen/Dense>
#include "../include/gradientDescent.hpp"
#include "../include/tensor.hpp"

int main() {
    // Synthetic linear data: y = X w_true + noise
    const int n = 300;     // samples
    const int d = 3;       // features

    Eigen::MatrixXd Xm = Eigen::MatrixXd::Random(n, d);
    Eigen::VectorXd w_true(d);
    w_true << 2.0, -3.0, 0.5;

    Eigen::VectorXd yv = Xm * w_true;
    yv.array() += 0.05 * Eigen::VectorXd::Random(n).array(); // small noise

    // Wrap in Tensor
    Tensor X(n, d); 
    X.data = Xm;

    // y as vector<Tensor*> with shape (1,1) each
    std::vector<Tensor> y_storage; 
    y_storage.reserve(n);
    std::vector<Tensor*> y_ptrs;   
    y_ptrs.reserve(n);
    
    for (int i = 0; i < n; ++i) {
        Tensor t(1, 1);
        t.data(0, 0) = yv(i);
        y_storage.push_back(t);
        y_ptrs.push_back(&y_storage.back());
    }

    // Initial weights
    Tensor w0(d, 1);
    w0.data.setZero();

    // Run GD
    Result res = gradientDescent(
        &w0, X, y_ptrs,
        "MINI",      // BatchMode
        "MSE",       // lossType
        "constant",  // learningRateType (safer for first test)
        1e-6,        // minGrad
        200,         // maxEpochs
        2000,        // maxIterations
        1e-7,        // lossDif
        0.0          // minLoss
    );

    std::cout << "Final loss: " << res.loss << "\n";
    std::cout << "Epochs: " << res.epochs << "\n";
    std::cout << "Learned weights:\n" << res.weights.data << "\n";
    std::cout << "True weights:\n" << w_true << "\n";
    
    return 0;
}