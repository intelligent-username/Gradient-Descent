#include <iostream>

#include "gr/gradient_descent.hpp"

int main() {
    gr::GradientDescentConfig config;
    gr::StopCriteria criteria;
    auto result = gr::optimize(config, criteria);
    std::cout << "Gradient Descent demo result value: " << result.value << '\n';
    return 0;
}
