// Learning Rates

#pragma once
#include "tensor.hpp"
#include <vector>
#include <string>
#include <functional>
using namespace std;

// Learning rate functions
double stepDecay(double initialLR, double gamma, double t, double k);
double exponentialDecay(double initialLR, int epoch, double decayRate);
double InverseTimeDecay(double initialR, double gamma, double k);
double NR(double initialLR, int epoch, double decayFactor);
