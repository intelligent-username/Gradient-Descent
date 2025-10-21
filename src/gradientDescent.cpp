// Gradient Descent Algorithm
// Implemented Beautifully

#include <limits>
#include <cmath>
#include <map>
#include <algorithm>
#include <stdexcept>
#include <utility>
#include <Eigen/Dense>
#include <random>

#include "gradientDescent.hpp"
#include "tensor.hpp"
#include "losses.hpp"
#include "batch.hpp"
#include "computer.hpp"
#include "lr.hpp"

using namespace Eigen;
using namespace std;

static BatchType parseBatchType(const std::string& s) {
    if (s == "FULL" || s == "Batch" || s == "BATCH") return BatchType::FULL;
    if (s == "STOCHASTIC" || s == "SGD") return BatchType::STOCHASTIC;
    return BatchType::MINI; // default
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
                        string BatchMode,
                        const string& lossType,
                        string learningRateType,
                        double minGrad,
                        int maxEpochs,
                        int maxIterations,
                        double lossDif,
                        double minLoss)
{
    // Initiate the actual data to train on
    Result final{*w0, 0.0, 0};

    // Working copy of weights
    Tensor w = *w0;

    // Train/Val/Test split
    int n = X.rows();
    if (n == 0) {
        throw std::invalid_argument("X has zero rows");
    }
    int trainEnd = static_cast<int>(n * 0.7);
    int valEnd   = static_cast<int>(n * 0.85);
    int testEnd  = n;

    // Ensure at least one training sample and non-decreasing boundaries
    if (trainEnd == 0) trainEnd = 1;
    if (valEnd < trainEnd) valEnd = trainEnd;
    if (testEnd < valEnd) testEnd = valEnd;

    Tensor X_train = X.block(0, 0, trainEnd, X.cols());
    Tensor X_val   = X.block(trainEnd, 0, valEnd - trainEnd, X.cols());
    Tensor X_test  = X.block(valEnd, 0, testEnd - valEnd, X.cols());

    vector<Tensor*> y_train(y.begin(), y.begin() + trainEnd);
    vector<Tensor*> y_val(y.begin() + trainEnd, y.begin() + valEnd);
    vector<Tensor*> y_test(y.begin() + valEnd, y.end());

    // Basic sanity checks (cheap, helpful for demos)
    if (static_cast<int>(y.size()) != n) {
        throw std::invalid_argument("y.size() must equal X.rows()");
    }
    if (X.cols() != w.rows()) {
        throw std::invalid_argument("X.cols() must match w0->rows()");
    }

    auto losses = getLossFunctions();

    auto it = losses.find(lossType);
    if (it == losses.end()) {
        throw std::invalid_argument("Unknown loss type: " + lossType);
    }
    auto loss = it->second;

    // Store the gradient of the loss functions
    // Will always be used
    auto gradFunc = gradientFinder(lossType);
    // So far, valid args are ['MSE', 'MAE', 'Hinge', 'NLL', 'cos']

    // Batching
    BatchType mode = parseBatchType(BatchMode);
    int batchStart = 0;
    int batchSize = computeBatchSize(X_train.rows(), mode);
    batchSize = std::clamp(batchSize, 1, std::max(1, X_train.rows()));

    // Prepare shuffled epoch copies for true stochastic/mini-batch behavior.
    // We shuffle indices and permute X_train / y_train together at the start
    // of training and after each epoch completes.
    std::random_device rd;
    std::mt19937 rng(rd());
    Tensor X_epoch = X_train;
    vector<Tensor*> y_epoch = y_train;
    auto shuffle_epoch = [&](void) {
        int m = X_train.rows();
        vector<int> idx(m);
        for (int i = 0; i < m; ++i) idx[i] = i;
        std::shuffle(idx.begin(), idx.end(), rng);
        Tensor X_tmp(m, X_train.cols());
        for (int i = 0; i < m; ++i) {
            X_tmp.data.row(i) = X_train.data.row(idx[i]);
        }
        X_epoch = X_tmp;
        vector<Tensor*> y_tmp; y_tmp.reserve(m);
        for (int i = 0; i < m; ++i) y_tmp.push_back(y_train[idx[i]]);
        y_epoch.swap(y_tmp);
    };
    // Initial shuffle before training loop
    shuffle_epoch();

    double baseLR = 0.1; // initial
    double learningRate = baseLR;
    VectorXd accumulatedSquares = VectorXd::Zero(w.rows()); // kept for API compatibility
    // Adam moments
    VectorXd m = VectorXd::Zero(w.rows());
    VectorXd v = VectorXd::Zero(w.rows());

    // Loop state
    int epoch = 0;
    int iteration = 0;
    double prevLoss = numeric_limits<double>::infinity();
    double currLoss = 0.0;
    double curGrad = numeric_limits<double>::infinity();

    double prevValLoss = numeric_limits<double>::infinity();
    int patience = 10;
    int patienceCounter = 0;

    bool running = true;

    while (running) {
        /*
        Steps for each iteration:
        
            1) Take the current weights
            2) Find the 'term' for the current iteration
                 (Whether that be the gradient of loss with points 
                 substituted in, momentum RMS/Adam/ADA, or otherwise
                 doesn't matter we're 'generalizing' the concept of
                 learning rate schedules here)
            3) Multiply it by the learning rate
            4) Update step (current weights - (learning ratae * term))
            5) Check stop conditions
            6) If stop conditions are met, return, else keep looping
        
        
        */
        
    // Get batch (from shuffled epoch data)
    auto [X_batch, y_batch] = getBatch(X_epoch, y_epoch, mode, batchStart, batchSize);

        // Predictions
        Tensor predictions = X_batch * w;

        // Compute loss for current batch
        currLoss = loss(predictions, y_batch, w);

        // Build y_true Tensor from y_batch
        Tensor y_true(static_cast<int>(y_batch.size()), 1);
        for (int i = 0; i < static_cast<int>(y_batch.size()); ++i)
            y_true.data(i, 0) = y_batch[i]->data(0, 0);

        // dL/dŷ
        Tensor gradPred = gradFunc(predictions, y_true);

    // ∂L/∂w = Xᵀ * dL/dŷ
    Tensor grad = X_batch.transpose() * gradPred;

        // Update step based on optimizer selection
        if (learningRateType == "Adam") {
            // Adam with persistent state and consistent per-iteration timestep
            // Use t = iteration + 1 for correct bias correction at each step.
            int t = iteration + 1;
            VectorXd stepVec = Adam(grad.data.col(0), m, v,
                                    0.9, 0.999, 1e-8,
                                    learningRate, t);
            Tensor step(stepVec.size(), 1);
            step.data.col(0) = stepVec;
            w = w - step; // Adam already scales by lr
        } else {
            // Update LR from schedule using the configured base learning rate
            learningRate = learningRateCaller(learningRateType, accumulatedSquares, learningRate, epoch, baseLR);
            // Standard SGD-style step
            w = w - (learningRate * grad);
        }

        // Update gradient magnitude for stopping
        curGrad = grad.norm();

        // If we completed an epoch (batchStart wrapped around), do validation
        if (batchStart == 0 && X_val.rows() > 0) {
            Tensor valPred = X_val * w;
            double valLoss = loss(valPred, y_val, w);
            if (valLoss > prevValLoss) {
                patienceCounter++;
                // printf("Validation loss increased (patience %d/%d) at iteration %d, epoch %d:\n", 
                //        patienceCounter, patience, iteration, epoch);
                // printf("  valLoss=%.6e > prevValLoss=%.6e\n", valLoss, prevValLoss);
                if (patienceCounter >= patience) {
                    // printf("Training stopped due to %d consecutive validation loss increases\n", patience);
                    break;
                }
            } else {
                patienceCounter = 0; // reset counter on improvement
            }
            prevValLoss = valLoss;
            epoch++;
            // printf("Epoch %d: learningRate = %.6e\n", epoch, learningRate);

            // Prepare/shuffle data for the next epoch
            shuffle_epoch();
        }

        iteration++;

        // Compute improvement BEFORE updating prevLoss
        double improvement = fabs(currLoss - prevLoss);

        // Stopping checks (every 10 iterations, full set; otherwise budget only)
        if (iteration % 10 == 0) {
            bool gradCondition = curGrad > minGrad;
            bool improvementCondition = improvement > lossDif;
            bool lossCondition = currLoss > minLoss;
            bool epochCondition = epoch < maxEpochs;
            bool iterationCondition = iteration < maxIterations;
            
            running = (gradCondition && improvementCondition && lossCondition && epochCondition && iterationCondition);
            
            // if (!running) {
            //     printf("\nTraining stopped at iteration %d, epoch %d:\n", iteration, epoch);
            //     printf("  curGrad=%.6e > minGrad=%.6e: %s\n", curGrad, minGrad, gradCondition ? "PASS" : "FAIL");
            //     printf("  improvement=%.6e > lossDif=%.6e: %s\n", improvement, lossDif, improvementCondition ? "PASS" : "FAIL");
            //     printf("  currLoss=%.6e > minLoss=%.6e: %s\n", currLoss, minLoss, lossCondition ? "PASS" : "FAIL");
            //     printf("  epoch=%d < maxEpochs=%d: %s\n", epoch, maxEpochs, epochCondition ? "PASS" : "FAIL");
            //     printf("  iteration=%d < maxIterations=%d: %s\n", iteration, maxIterations, iterationCondition ? "PASS" : "FAIL");
            // }
        } else {
            running = (
                epoch < maxEpochs &&
                iteration < maxIterations
            );
        }

        // Now update prevLoss for the next iteration
        prevLoss = currLoss;
    }

    final.weights = w;
    final.loss = currLoss;
    final.epochs = epoch;
    return final;
    // QED :)
};
