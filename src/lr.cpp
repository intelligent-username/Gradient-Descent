// Learning Rates

double learningRateCaller(const string& type, double initialLR, VectorXd& accumulatedSquares, int epoch, double param) {
    if (type == "StepDecay") {
        return stepDecay(initialLR, 0.96, 20, epoch);
    } else if (type == "ExponentialDecay") {
        return exponentialDecay(initialLR, epoch, param);
    } else if (type == "InverseTimeDecay") {
        return InverseTimeDecay(initialLR, param, epoch);
    } else if (type == "NR") {
        MatrixXd H_inv = NewtonRaphson(hessian);
        return H_inv;
    } else if (type == "AG") {
        return AdaGrad(gradients, accumulatedSquares, initialLR, epoch);
    } else {
        return initialLR; // Default static learning rate
    }
}

// Made in order of appearance in the README
// k represents epochs.

double stepDecay(double initialLR, double gamma, double t, double k) {
    return initialLR * pow(gamma, floor(t / k));
}

double exponentialDecay(double initialLR, int epoch, double decayRate) {
    return initialLR * exp(-decayRate * epoch);
}

double InverseTimeDecay(double initialR, double gamma, double k) {
    return initialR/(1 + gamma * k);
}

// Newton-Raphson decay (Newton's Method)
// The learning rate "is" the inverse Hessian

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
                 double epsilon = 1e-8) {
    accumulatedSquares.array() += grad.data.array().square();

    grad.data.array() /= (accumulatedSquares.array().sqrt() + 1e-8);
    grad.data *= initialLR;
    return grad;
}
