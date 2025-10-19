#ifndef COMPUTER_HPP
#define COMPUTER_HPP

#include "tensor.hpp"
#include <string>

/**
 * Compute gradient application given a gradient Tensor and input points Tensor
 * 
 * Supports multiple operation modes:
 * - scalar gradient (1x1) -> scales points
 * - matrix multiplication when inner dimensions match
 * - element-wise product when shapes match exactly
 * 
 * @param gradient The gradient tensor to apply
 * @param points The input points tensor
 * @return Tensor Result of gradient application
 * @throws std::invalid_argument if tensor shapes are incompatible
 */
Tensor computeGradient(const Tensor& gradient, const Tensor& points);

/**
 * Gradient finder for different loss functions
 * 
 * Returns a gradient coefficient tensor for the specified loss type.
 * Currently returns placeholder scalar values that can be extended
 * to full gradient implementations in the future.
 * 
 * @param lossType The type of loss function ("MSE", "MAE", "Hinge", "NLL", "cos")
 * @return Tensor A 1x1 tensor containing the gradient coefficient
 * @throws std::invalid_argument if loss type is unknown
 */
Tensor gradientFinder(const std::string& lossType);

#endif // COMPUTER_HPP
