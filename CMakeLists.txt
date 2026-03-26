cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Setup CUDA Architectures before project() so it becomes the default
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

project(LearnCuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find CUDA Toolkit to link against libraries like cublas, cudart
find_package(CUDAToolkit REQUIRED)

# Common utility library
add_subdirectory(0_utils)

# Sample executables
add_subdirectory(1_print_cuda_info)
add_subdirectory(1_5_matrix_transposition)
add_subdirectory(2_vector_add)
add_subdirectory(3_vector_reduction)
add_subdirectory(4_matrix_mul)
add_subdirectory(100_float_precision)
add_subdirectory(101_test_shm_bank)
