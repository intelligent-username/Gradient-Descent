#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <Eigen/Dense>

using namespace Eigen;

class Tensor {
public:
    MatrixXd data;

    Tensor(int rows, int cols);
    Tensor(const MatrixXd& m);

    int rows() const;
    int cols() const;

    Tensor operator-(const Tensor& other) const;
    Tensor operator*(double scalar) const;
};

#endif
