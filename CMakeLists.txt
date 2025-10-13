cmake_minimum_required(VERSION 3.10)
project(GradientDescent LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(${CMAKE_SOURCE_DIR}/include)

# Find Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Library
add_library(gradient_descent STATIC
    src/tensor.cpp
    src/gradient_descent.cpp
    src/loss.cpp
    src/stop_criteria.cpp
)
target_include_directories(gradient_descent PUBLIC include)
target_link_libraries(gradient_descent Eigen3::Eigen)

# Demo executable
add_executable(demo src/main.cpp)
target_link_libraries(demo gradient_descent)
