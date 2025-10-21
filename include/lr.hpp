// Learning Rates

#pragma once
#include <string>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

// Learning rate schedules
double stepDecay(double initialLR, double gamma, double t, double k);
double exponentialDecay(double initialLR, int epoch, double decayRate);
double InverseTimeDecay(double initialR, double gamma, double k);
double Adam(double initialLR, int epoch, double param = 0.0);
double Nadam(double initialLR, int epoch, double param = 0.0);
double AMSGrad(double initialLR, int epoch, double param = 0.0);

// Optimizer helpers
MatrixXd NewtonRaphson(const MatrixXd& hessian);
VectorXd AdaGrad(const VectorXd& grad,
               VectorXd& accumulatedSquares,
               double initialLR,
               double epsilon = 1e-8);

// Learning rate / optimizer dispatcher used in gradientDescent
double learningRateCaller(const string& type,
                       VectorXd& accumulatedSquares,
                       double initialLR,
                       int epoch,
                       double param);

VectorXd Adam(const VectorXd& grad,
              VectorXd& m,
              VectorXd& v,
              double beta1,
              double beta2,
              double epsilon,
              double lr,
              int t);

VectorXd Nadam(const VectorXd& grad,
               VectorXd& m,
               VectorXd& v,
               double beta1,
               double beta2,
               double epsilon,
               double lr,
               int t);

VectorXd AMSGrad(const VectorXd& grad,
                 VectorXd& m,
                 VectorXd& v,
                 VectorXd& v_hat,
                 double beta1,
                 double beta2,
                 double epsilon,
                 double lr,
                 int t);

