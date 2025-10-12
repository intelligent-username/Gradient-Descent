#include <iostream>

#include "gr/optimizers.hpp"
#include "gr/types.hpp"

int main() {
    gr::OptimizationState state;
    gr::GradientDescentConfig config;
    auto updated = gr::gradient_descent_step(state, config);
    std::cout << "RMSProp demo placeholder value: " << updated.value << '\n';
    return 0;
}
