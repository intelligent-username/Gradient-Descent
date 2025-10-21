// This one just has random, synthetic data.
// Upcoming ones will use real data and compare to expected outputs

#include <cstdio>
#include <random>
#include <vector>
#include <Eigen/Dense>
#include "../include/gradientDescent.hpp"
#include "../include/tensor.hpp"

using namespace std;
using namespace Eigen;

int main() {

    printf("\n\n");
    printf("FIRST DEMO: plane through 3D points\n");
    printf("--------------------\n");

    // Synthetic linear data: y = X w_true + noise
    const int n = 300;     // samples
    const int d = 3;       // features

    MatrixXd Xm = MatrixXd::Random(n, d);
    VectorXd w_true(d);
    w_true << 2.0, -3.0, 0.5;

    VectorXd yv = Xm * w_true;
    yv.array() += 0.05 * VectorXd::Random(n).array(); // small noise

    // Wrap in Tensor
    Tensor X(n, d); 
    X.data = Xm;

    // y as vector<Tensor*> with shape (1,1) each
    vector<Tensor> y_storage; 
    y_storage.reserve(n);
    vector<Tensor*> y_ptrs;   
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
        800,         // maxEpochs
        2000,        // maxIterations
        1e-7,        // lossDif
        0.0          // minLoss
    );

    printf("Final loss: %.6f\n", res.loss);
    printf("Epochs: %d\n", res.epochs);
    printf("True:    2, -3, 0.5\n");
    printf("Learned: %.6f %.6f %.6f\n",
           res.weights.data(0), res.weights.data(1), res.weights.data(2));

}
