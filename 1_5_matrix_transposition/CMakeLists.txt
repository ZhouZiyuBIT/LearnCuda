add_executable(matrix_transposition_tests
    matrix_transposition.cu
    matrix_transposition_tests.cpp
)

# Link against the utils library which brings in its include directory
target_link_libraries(matrix_transposition_tests PRIVATE utils CUDA::cudart)
