// Learning rates

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
// The learning rate is basically the Hessian, it's then multiplied by the gradient of the loss functionw with those weights to find the next weights

double NR(double initialLR, int epoch, double decayFactor) {
    return initialLR / (1 + decayFactor * epoch);
}



