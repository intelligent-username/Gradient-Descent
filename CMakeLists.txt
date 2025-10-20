cmake_minimum_required(VERSION 3.10)
project(GradientDescent LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(${CMAKE_SOURCE_DIR}/include)

# Use vendored Eigen headers from include/ to avoid external dependency
# If you prefer system Eigen, remove the next line and uncomment find_package below.
# find_package(Eigen3 REQUIRED)
# include_directories(${EIGEN3_INCLUDE_DIR})

# Library
add_library(gradient_descent STATIC
    src/tensor.cpp
    src/gradientDescent.cpp
    src/losses.cpp
    src/lr.cpp
    src/computer.cpp
    src/batch.cpp
)
target_include_directories(gradient_descent PUBLIC include)
## If using system Eigen, link if needed
# target_link_libraries(gradient_descent Eigen3::Eigen)

# Executables, but not for demos

add_executable(first demos/first.cpp)
target_link_libraries(first gradient_descent)
