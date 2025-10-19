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
