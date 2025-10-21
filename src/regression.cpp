#include <memory>

#include "regression.hpp"

using namespace std;
using namespace Eigen;

Result fitRegression(
    const Tensor& X,
    const vector<Tensor*>& y,
    RegressionType type,
    int degree,
    const string& batchMode,
    const string& lossType,
    const string& learningRateType,
    double minGrad,
    int maxEpochs,
    int maxIterations,
    double lossDif,
    double minLoss)
{
    unique_ptr<Transformer> transformer;
    
    switch (type) {
        case RegressionType::LINEAR:
            transformer = make_unique<IdentityTransformer>();
            break;
            
        case RegressionType::POLYNOMIAL:
            transformer = make_unique<PolynomialTransformer>(degree);
            break;
            
        case RegressionType::AUTO:
            int auto_degree = suggestPolynomialDegree(X.rows(), X.cols(), degree);
            transformer = make_unique<PolynomialTransformer>(auto_degree);
            break;
    }
    
    Tensor X_transformed = transformer->transform(X);
    
    Tensor w0(X_transformed.cols(), 1);
    w0.data.setZero();
    
    // Run gradient descent on transformed features
    return gradientDescent(
        &w0, 
        X_transformed, 
        y,
        batchMode,
        lossType,
        learningRateType,
        minGrad,
        maxEpochs,
        maxIterations,
        lossDif,
        minLoss
    );
}
