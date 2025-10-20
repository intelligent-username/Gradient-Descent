#ifndef TRANSFORMER_HPP
#define TRANSFORMER_HPP

#include "tensor.hpp"

// Base class for feature transformations
class Transformer {
public:
    virtual ~Transformer() = default;
    virtual Tensor transform(const Tensor& X) const = 0;
};

// Identity transformer (no transformation)
class IdentityTransformer : public Transformer {
public:
    Tensor transform(const Tensor& X) const override {
        return X; // Return as-is
    }
};

// Polynomial feature transformer
class PolynomialTransformer : public Transformer {
private:
    int degree;
    bool include_bias;
    
public:
    PolynomialTransformer(int degree = 2, bool include_bias = true) 
        : degree(degree), include_bias(include_bias) {}
        
    Tensor transform(const Tensor& X) const override;
};

// Automatically select degree based on dataset size
int suggestPolynomialDegree(int n_samples, int n_features, int max_degree = 5);

#endif // TRANSFORMER_HPP
