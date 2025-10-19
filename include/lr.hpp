// Learning Rates

#pragma once
#include <string>
#include <Eigen/Dense>

// Learning rate schedules
double stepDecay(double initialLR, double gamma, double t, double k);
double exponentialDecay(double initialLR, int epoch, double decayRate);
double InverseTimeDecay(double initialR, double gamma, double k);

// Optimizer helpers
Eigen::MatrixXd NewtonRaphson(const Eigen::MatrixXd& hessian);
Eigen::VectorXd AdaGrad(const Eigen::VectorXd& grad,
						Eigen::VectorXd& accumulatedSquares,
						double initialLR,
						double epsilon = 1e-8);

// Learning rate / optimizer dispatcher used in gradientDescent
double learningRateCaller(const std::string& type,
						  Eigen::VectorXd& accumulatedSquares,
						  double initialLR,
						  int epoch,
						  double param);
