#pragma once

#include "types.hpp"

namespace gr {

OptimizationState gradient_descent_step(const OptimizationState& current,
                                        const GradientDescentConfig& config);

} // namespace gr
