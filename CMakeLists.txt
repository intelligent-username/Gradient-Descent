cmake_minimum_required(VERSION 3.10)
project(GradientDescent LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(${CMAKE_SOURCE_DIR}/include)

# Library
add_library(gradient_descent STATIC
    src/tensor.cpp
    src/gradientDescent.cpp
    src/losses.cpp
    src/lr.cpp
    src/computer.cpp
    src/batch.cpp
    src/transformer.cpp
    src/regression.cpp
)
target_include_directories(gradient_descent PUBLIC include)

# Executables

# First demo
add_executable(first demos/first.cpp)
target_link_libraries(first gradient_descent)

# Second demo
add_executable(second demos/second.cpp)
target_link_libraries(second gradient_descent)