#pragma once

#include <cstddef>

#include "types.hpp"

namespace gr {

bool reached_value_tolerance(const OptimizationState& previous,
                             const OptimizationState& current,
                             const StopCriteria& criteria);

bool reached_gradient_tolerance(const OptimizationState& current,
                                const StopCriteria& criteria);

bool reached_iteration_limit(std::size_t iteration,
                             const StopCriteria& criteria);

} // namespace gr
