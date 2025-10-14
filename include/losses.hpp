#pragma once
#include "tensor.hpp"
#include <vector>
#include <string>
#include <functional>
using namespace std;

double mseLoss(const Tensor&, const vector<Tensor*>&, const Tensor&);
double maeLoss(const Tensor&, const vector<Tensor*>&, const Tensor&);
double hingeLoss(const Tensor&, const vector<Tensor*>&, const Tensor&);
double negativeLogLikelihoodLoss(const Tensor&, const vector<Tensor*>&, const Tensor&);
double cosineLoss(const Tensor&, const vector<Tensor*>&, const Tensor&);

map<string, function<double(const Tensor&, const vector<Tensor*>&, const Tensor&)>> getLossFunctions();
