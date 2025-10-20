#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <Eigen/Dense>

using namespace Eigen;

class Tensor {
public:
    MatrixXd data;

    Tensor(int rows, int cols);
    Tensor(const MatrixXd& m);
    // Optional default
    Tensor() : data() {}

    // Accessors
    int rows() const;
    int cols() const;

    // Basic ops
    Tensor operator-(const Tensor& other) const;
    Tensor operator+(const Tensor& other) const;
    Tensor operator*(double scalar) const;
    Tensor operator*(const Tensor& other) const; // multiplication

    Tensor cwiseProduct(const Tensor& other) const;
    Tensor sign() const;

    // L.A. helpers
    Tensor transpose() const;
    Tensor block(int startRow, int startCol, int blockRows, int blockCols) const;
    double norm() const;

    // Helpers used by losses/gradients
    static Tensor ZeroLike(const Tensor& other);
    static Tensor Ones(int rows, int cols);

    friend Tensor operator*(double scalar, const Tensor& t) {
        return t * scalar;
    }
};

#endif
