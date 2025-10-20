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

Tensor Tensor::operator+(const Tensor& other) const {
    return Tensor(data + other.data);
}

Tensor Tensor::operator*(double scalar) const {
    return Tensor(data * scalar);
}

Tensor Tensor::operator*(const Tensor& other) const {
    return Tensor(data * other.data);
}

// Elementwise ops
Tensor Tensor::cwiseProduct(const Tensor& other) const {
    return Tensor(data.cwiseProduct(other.data));
}

Tensor Tensor::sign() const {
    Tensor out(rows(), cols());
    
    out.data = ((data.array() > 0.0).cast<double>() * 1.0) - 
               ((data.array() < 0.0).cast<double>() * 1.0);
    
    return out;
}

// Linear algebra helpers
Tensor Tensor::transpose() const {
    return Tensor(data.transpose());
}

Tensor Tensor::block(int startRow, int startCol, int blockRows, int blockCols) const {
    return Tensor(data.block(startRow, startCol, blockRows, blockCols));
}

double Tensor::norm() const {
    return data.norm();
}

// Factories
Tensor Tensor::ZeroLike(const Tensor& other) {
    return Tensor(MatrixXd::Zero(other.rows(), other.cols()));
}

Tensor Tensor::Ones(int rows, int cols) {
    return Tensor(MatrixXd::Ones(rows, cols));
}
