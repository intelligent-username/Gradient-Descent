#include "gr/stop_conditions.hpp"

#include <cmath>

namespace gr {

bool reached_value_tolerance(const OptimizationState& previous,
                             const OptimizationState& current,
                             const StopCriteria& criteria) {
    return std::abs(current.value - previous.value) < criteria.tolerance;
}

bool reached_gradient_tolerance(const OptimizationState& current,
                                const StopCriteria& criteria) {
    return current.gradient_norm < criteria.gradient_tolerance;
}

bool reached_iteration_limit(std::size_t iteration,
                             const StopCriteria& criteria) {
    return iteration >= criteria.max_iterations;
}

} // namespace gr
