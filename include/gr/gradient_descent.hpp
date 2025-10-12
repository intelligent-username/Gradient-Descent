#pragma once

#include "optimizers.hpp"
#include "stop_conditions.hpp"

namespace gr {

OptimizationState optimize(const GradientDescentConfig& config,
                           const StopCriteria& criteria);

} // namespace gr
