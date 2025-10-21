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

MatrixXd NewtonRaphson(const MatrixXd& hessian);
VectorXd AdaGrad(const VectorXd& grad,
               VectorXd& accumulatedSquares,
               double initialLR,
               double epsilon = 1e-8);

VectorXd learningRateCaller(const string& type,
                           const VectorXd& grad,
                           VectorXd& accumulatedSquares,
                           VectorXd& m,
                           VectorXd& v,
                           VectorXd& v_hat,
                           double& learningRate,
                           int epoch,
                           int iteration,
                           double baseLR);

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

