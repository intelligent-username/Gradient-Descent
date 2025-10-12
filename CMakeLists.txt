# CMake configuration for Gradient Descent project

cmake_minimum_required(VERSION 3.16)

project(GradientDescent LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(gr_lib
	src/gradient_descent.cpp
	src/optimizers.cpp
	src/stop_conditions.cpp
	src/utils.cpp
)

target_include_directories(gr_lib
	PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(demo_gd demos/demo_gd.cpp)
target_link_libraries(demo_gd PRIVATE gr_lib)

add_executable(demo_adagrad demos/demo_adagrad.cpp)
target_link_libraries(demo_adagrad PRIVATE gr_lib)

add_executable(demo_rmsprop demos/demo_rmsprop.cpp)
target_link_libraries(demo_rmsprop PRIVATE gr_lib)

add_executable(demo_adam demos/demo_adam.cpp)
target_link_libraries(demo_adam PRIVATE gr_lib)

add_executable(test_all tests/test_all.cpp)
target_link_libraries(test_all PRIVATE gr_lib)
