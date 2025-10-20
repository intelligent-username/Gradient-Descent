// Learning Rates

#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>
#include <string>
#include "lr.hpp"

using namespace Eigen;
using std::invalid_argument;
using std::runtime_error;
using std::string;

double learningRateCaller(const string& type, VectorXd& /*accumulatedSquares*/, double initialLR, int epoch, double param) {
    if (type == "StepDecay") {
        // step-decay: eta = eta0 * gamma^{floor(t/k)}, with t=epoch, k=20
        return stepDecay(initialLR, 0.96, static_cast<double>(epoch), 20.0);
    } else if (type == "ExponentialDecay") {
        return exponentialDecay(initialLR, epoch, param);
    } else if (type == "InverseTimeDecay") {
        return InverseTimeDecay(initialLR, param, static_cast<double>(epoch));
    } else {
        // Unknown or "AG" for now: fall back to constant LR
        return initialLR;
    }
}

// Made in order of appearance in the README
// k represents epochs.

double stepDecay(double initialLR, double gamma, double t, double k) {
    return initialLR * std::pow(gamma, std::floor(t / k));
}

double exponentialDecay(double initialLR, int epoch, double decayRate) {
    return initialLR * std::exp(-decayRate * epoch);
}

double InverseTimeDecay(double initialR, double gamma, double k) {
    return initialR / (1 + gamma * k);
}

// Inverse Hessian helper
MatrixXd NewtonRaphson(const MatrixXd& hessian) {
    if (hessian.rows() != hessian.cols()) {
        throw invalid_argument("Hessian must be square.");
    }

    LLT<MatrixXd> solver(hessian);
    if (solver.info() != Success) {
        throw runtime_error("Hessian decomposition failed.");
    }

    return solver.solve(MatrixXd::Identity(hessian.rows(), hessian.cols()));
}

VectorXd AdaGrad(const VectorXd& grad,
                 VectorXd& accumulatedSquares,
                 double initialLR,
                 double epsilon) {
    accumulatedSquares.array() += grad.array().square();
    VectorXd adjusted = grad.array() / (accumulatedSquares.array().sqrt() + epsilon);
    return initialLR * adjusted;
}
