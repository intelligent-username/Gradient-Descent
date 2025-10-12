#pragma once

#include <cstddef>

namespace gr {

using Scalar = double;

struct OptimizationState {
    Scalar value{0.0};
    Scalar gradient_norm{0.0};
};

struct GradientDescentConfig {
    Scalar learning_rate{0.01};
};

struct StopCriteria {
    Scalar tolerance{1e-6};
    Scalar gradient_tolerance{1e-6};
    std::size_t max_iterations{1000};
};

} // namespace gr
