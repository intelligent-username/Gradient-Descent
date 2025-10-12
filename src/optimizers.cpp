#include "gr/optimizers.hpp"

namespace gr {

OptimizationState gradient_descent_step(const OptimizationState& current,
                                        const GradientDescentConfig& config) {
    (void)config;
    return current;
}

} // namespace gr
