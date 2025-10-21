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
    } else if (type == "Adam") {
        return initialLR; // Adam handles its own LR scaling internally
    } else if (type == "Nadam") {
        return initialLR; // Nadam handles its own LR scaling internally
    } else if (type == "AMSGrad") {
        return initialLR; // AMSGrad handles its own LR scaling internally
    } else {
        // Unknown or constant: fall back to constant LR
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

// Adam optimizer
VectorXd Adam(const VectorXd& grad,
              VectorXd& m,
              VectorXd& v,
              double beta1,
              double beta2,
              double epsilon,
              double lr,
              int t)
{
    m = beta1 * m + (1.0 - beta1) * grad;
    v = beta2 * v + (1.0 - beta2) * grad.array().square().matrix();
    VectorXd m_hat = m / (1.0 - std::pow(beta1, t));
    VectorXd v_hat = v / (1.0 - std::pow(beta2, t));
    return lr * m_hat.array() / (v_hat.array().sqrt() + epsilon);
}

// Nadam optimizer
VectorXd Nadam(const VectorXd& grad,
               VectorXd& m,
               VectorXd& v,
               double beta1,
               double beta2,
               double epsilon,
               double lr,
               int t)
{
    m = beta1 * m + (1.0 - beta1) * grad;
    v = beta2 * v + (1.0 - beta2) * grad.array().square().matrix();
    VectorXd m_hat = (beta1 * m / (1.0 - std::pow(beta1, t))) + ((1.0 - beta1) * grad / (1.0 - std::pow(beta1, t)));
    VectorXd v_hat = v / (1.0 - std::pow(beta2, t));
    return lr * m_hat.array() / (v_hat.array().sqrt() + epsilon);
}

// AMSGrad optimizer
VectorXd AMSGrad(const VectorXd& grad,
                 VectorXd& m,
                 VectorXd& v,
                 VectorXd& v_hat,
                 double beta1,
                 double beta2,
                 double epsilon,
                 double lr,
                 int t)
{
    m = beta1 * m + (1.0 - beta1) * grad;
    v = beta2 * v + (1.0 - beta2) * grad.array().square().matrix();
    v_hat = v_hat.array().max(v.array());
    VectorXd m_hat = m / (1.0 - std::pow(beta1, t));
    return lr * m_hat.array() / (v_hat.array().sqrt() + epsilon);
}
