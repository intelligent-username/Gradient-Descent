#include <cassert>

#include "gr/utils.hpp"

int main() {
    gr::OptimizationState state;
    state.value = 42.0;
    gr::reset_state(state);
    assert(state.value == 0.0);
    return 0;
}
