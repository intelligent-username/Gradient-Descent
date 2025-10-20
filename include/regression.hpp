#ifndef REGRESSION_HPP
#define REGRESSION_HPP

#include "gradientDescent.hpp"
#include "transformer.hpp"
#include <memory>
#include <string>

enum class RegressionType {
    LINEAR,
    POLYNOMIAL,
    AUTO
};

/**
 * Enhanced regression that handles feature transformation
 * 
 * @param X Input data
 * @param y Target values
 * @param type Regression type (LINEAR, POLYNOMIAL, AUTO)
 * @param degree Polynomial degree (if POLYNOMIAL), or max degree (if AUTO)
 * @param batchMode Batch mode for gradient descent
 * @param lossType Loss function to use
 * @param learningRateType Learning rate schedule
 * @param other_params Other gradient descent parameters
 * @return Result object with fitted model
 */
Result fitRegression(
    const Tensor& X,
    const std::vector<Tensor*>& y,
    RegressionType type = RegressionType::LINEAR,
    int degree = 2,
    const std::string& batchMode = "MINI",
    const std::string& lossType = "MSE",
    const std::string& learningRateType = "constant",
    double minGrad = 1e-6,
    int maxEpochs = 200,
    int maxIterations = 2000,
    double lossDif = 1e-7,
    double minLoss = 0.0
);

#endif
