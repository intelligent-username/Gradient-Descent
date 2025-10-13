#include "gradientDescent.hpp"
#include "tensor.hpp"
using namespace Eigen;
using namespace std;

/**
* Gradient Descent Algorithm

* NOTE: Remember to make the logic for validation & test sets.
**/

struct Result {
    Tensors weights;
    double loss;
    int epochs;
}

/**
*
Inputs:
- @param w0: Initial weights, default values or otherwise
- @param X: Input data (Tensor)
- @param y: Target outputs (vector<Tensor*>)
- @param mode: Batch mode (Batch Vs. MiniBatch. Vs  Stochastic)

    Stopping conditions:
    - @param minGrad: Minimum gradient for continuing iterations
    - @param maxEpochs: Maximum number of epochs
    - @param lossDif: Minimum difference before we conclude convergence
    - @param minLoss: Minimum loss value for early stopping
    - @param maxGrad: Maximum gradient for continuing iterations
    - @param minWeight: Minimum weight value for early stopping

Output:
- Result: custom class for returning the final model
  Components:
    - Updated weights in w0 (Tensor*)
    - Final loss value (double)
    - Epochs run (int)
*
**/
Result gradientDescent(Tensor* w0,
    const Tensor& X,
    const vector<Tensor*>& y,
    BatchMode mode,
    double minGrad = 1e-3,
    int maxEpochs = 1000,
    double lossDif = 1e-5,
    double minLoss = 1e-4
){
    Tensor w = *w0; // Current weights
    int epoch = 0;
    double prevLoss = std::numeric_limits<double>::max();
    double currLoss = 0.0;
    double curGrad = std::numeric_limits<double>::max();
    double lossDiff = 0.0;

    while (!(epoch < maxEpochs || curGrad < minGrad || currLoss > minLoss || lossDiff < lossDif)) {
        
        
        if (mode == BatchMode::Stochastic) {
            // Stochastic Gradient Descent
            

        } else if (mode == BatchMode::MiniBatch) {
            // Mini-Batch Gradient Descent
            
            
        } else {
            // Batch Gradient Descent
            
            
        }        
    
    }
};
