#include "tensor.hpp"
using namespace Eigen;

// Constructors
Tensor::Tensor(int rows, int cols) {
    data = MatrixXd::Zero(rows, cols);
}

Tensor::Tensor(const MatrixXd& m) {
    data = m;
}

// Accessors
int Tensor::rows() const { return data.rows(); }
int Tensor::cols() const { return data.cols(); }

// Operators
Tensor Tensor::operator-(const Tensor& other) const {
    return Tensor(data - other.data);
}

Tensor Tensor::operator*(double scalar) const {
    return Tensor(data * scalar);
}
