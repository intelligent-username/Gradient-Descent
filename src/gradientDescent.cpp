#include "gradientDescent.hpp"
#include "tensor.hpp"
#include "losses.hpp"
#include "batch.hpp"

using namespace Eigen;
using namespace std;

/**
* Gradient Descent Algorithm

// Mini helpers

// Universal object to return
struct Result {
    Tensor weights;
    double loss;
    int epochs;
};

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

                        string BatchMode,
                        const string& lossType,
                        string learningRateType,
                        
                        double minGrad = 1e-3,
                        int maxEpochs = 2000,
                        int maxIterations = 2000,
                        double lossDif = 1e-5,
                        double minLoss = 1e-4)
{
    // Initiate the actual data to train/on

    Tensor w = *w0; 

    int n = X.rows();
    int trainEnd = n * 0.7;       // 70% training
    int valEnd = n * 0.85;        // 15% validation
    int testEnd = n;               // 15% test

    Tensor X_train = X.block(0, 0, trainEnd, X.cols());
    Tensor X_val   = X.block(trainEnd, 0, valEnd - trainEnd, X.cols());
    Tensor X_test  = X.block(valEnd, 0, testEnd - valEnd, X.cols());

    vector<Tensor*> y_train(y.begin(), y.begin() + trainEnd);
    vector<Tensor*> y_val(y.begin() + trainEnd, y.begin() + valEnd);
    vector<Tensor*> y_test(y.begin() + valEnd, y.end());


    int epoch = 0;
    int iteration = 0;
    double prevLoss = std::numeric_limits<double>::max();
    double currLoss = 0.0;
    double curGrad = std::numeric_limits<double>::max();

    auto losses = getLossFunctions();
    auto loss = losses[lossType];

    // Store the gradient of the loss functions
    // Will always be used

    


    bool running = true;

    while (running) {

        /* Steps for each iteration:

        1) Take the current weights
        2) Find the 'term' for the current iteration
                (Whether that be the gradient of loss with points substituted in, momentum
                RMS/Adam/ADA, or otherwise)
        3) Multiply it by the learning rate 
           Whether its dyanmic, deterministic, etc.
        4) Subtract the learning rate * the gradients from current weights
        5) Check if the updated weights/conditions should trigger the stop
        6) If they do, return. If not, continue iterating

        */
        
        // First pick out the batch

        Tensor X_batch = X_train; // Placeholder until batching implemented
        vector<Tensor*> y_batch = y_train; // Placeholder

        // Run it through the current weights
        Tensor predictions = X_batch * w;

        // Compute loss
        currLoss = loss(predictions, y_batch, w);

        // Find how much to subtract (gradient placeholder)
        Tensor grad = X_batch.transpose() * (predictions - Tensor(y_batch.size(), 1)); // Placeholder for gradient logic

        // Update the weights (simple GD step)

        if learningRateType != "StepDecay" {
            // Setting gamma to 0.96
            // i.e. learning rate shrinks by 5% every 20 epochs.
            // This is a hyperparameter that can be tuned

            learningRate = stepDecay(learningRate, 0.96, 20, epoch);

        } else if learningRateType != "ExponentialDecay" {
            learningRate = 0.01; // Default static learning rate
        } else if learningRateType != "NR" {
            learningRate = NR(learningRate, epoch, 0.1);
        }




        w = w - (grad * learningRate);

        // Update gradient magnitude for stopping condition
        curGrad = grad.norm();

        // And update the stop condition trackers

        // Now that I think about it, this process is very simple.

        if (iteration % 10 == 0) {
            running = (
                curGrad > minGrad &&
                fabs(currLoss - prevLoss) > lossDif &&
                currLoss > minLoss &&
                epoch < maxEpochs &&
                iteration < maxIterations
            );
        } else {
            running = (
                epoch < maxEpochs &&
                iteration < maxIterations
            );
        }
    
        prevLoss = currLoss;
        iteration++;
        if (iteration % 100 == 0) epoch++;
    }

    Result final = {w, currLoss, epoch};
    
    return final;
    // QED :)
};
