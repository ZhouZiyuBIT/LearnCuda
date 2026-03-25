add_executable(matrix_mul
    matrix_mul_main.cpp
    matrix_mul.cu
)

# Link against utils library and CUBLAS
target_link_libraries(matrix_mul PRIVATE utils CUDA::cublas CUDA::cudart)
