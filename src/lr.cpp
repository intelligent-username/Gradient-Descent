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

VectorXd learningRateCaller(const string& type,
                           const VectorXd& grad,
                           VectorXd& accumulatedSquares,
                           VectorXd& m,
                           VectorXd& v,
                           VectorXd& v_hat,
                           double& learningRate,
                           int epoch,
                           int iteration,
                           double baseLR)
{
    if (type == "Adam") {
        int t = iteration + 1;
        return Adam(grad, m, v, 0.9, 0.999, 1e-8, learningRate, t);
    } else if (type == "Nadam") {
        int t = iteration + 1;
        return Nadam(grad, m, v, 0.9, 0.999, 1e-8, learningRate, t);
    } else if (type == "AMSGrad") {
        int t = iteration + 1;
        return AMSGrad(grad, m, v, v_hat, 0.9, 0.999, 1e-8, learningRate, t);
    } else if (type == "AdaGrad") {
        return AdaGrad(grad, accumulatedSquares, learningRate, 1e-8);
    }

    double scheduleParam = 0.0;
    if (type == "ExponentialDecay") scheduleParam = 0.01;
    if (type == "InverseTimeDecay") scheduleParam = 0.1;
    if (type == "StepDecay") {
        learningRate = stepDecay(baseLR, 0.96, static_cast<double>(epoch), 20.0);
    } else if (type == "ExponentialDecay") {
        learningRate = exponentialDecay(baseLR, epoch, scheduleParam);
    } else if (type == "InverseTimeDecay") {
        learningRate = InverseTimeDecay(baseLR, scheduleParam, static_cast<double>(epoch));
    } else {
        learningRate = baseLR;
    }
    return learningRate * grad;
}

// Made in order of appearance in the README
// Some are neglected because I don't have 10 years to perrfect this project.
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

// AdaGrad
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

// Nadam
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

// AMSGrad
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

