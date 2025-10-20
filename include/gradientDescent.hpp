#ifndef GRADIENT_DESCENT_HPP
#define GRADIENT_DESCENT_HPP

#include "tensor.hpp"
#include <vector>
#include <limits>
#include <string>

// Result object returned by gradientDescent
struct Result {
    Tensor weights;
    double loss;
    int epochs;
};

/**
 * Gradient Descent Algorithm
 * 
 * NOTE: Remember to make the logic for validation & test sets.
 * 
 * @param w0 Initial weights (Tensor*)
 * @param X Input data (Tensor)
 * @param y Target outputs (vector<Tensor*>)
 * @param mode Batch mode (Batch vs. MiniBatch vs. Stochastic)
 * @param minGrad Minimum gradient for continuing iterations (default: 1e-3)
 * @param maxEpochs Maximum number of epochs (default: 1000)
 * @param lossDif Minimum difference before we conclude convergence (default: 1e-5)
 * @param minLoss Minimum loss value for early stopping (default: 1e-4)
 *
 *
 * Splits data into train/val/test, iterates updates with a selected loss and
 * learning rate schedule, and returns the final weights and stats.
 */
Result gradientDescent(Tensor* w0,
                       const Tensor& X,
                       const std::vector<Tensor*>& y,
                       std::string BatchMode,
                       const std::string& lossType,
                       std::string learningRateType,
                       double minGrad = 1e-3,
                       int maxEpochs = 2000,
                       int maxIterations = 2000,
                       double lossDif = 1e-5,
                       double minLoss = 1e-4);

#endif // GRADIENT_DESCENT_HPP
