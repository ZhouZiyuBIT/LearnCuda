add_executable(float_precision_test
    float_precision_main.cpp
)

# Link against utils library and CUBLAS
target_link_libraries(float_precision_test PRIVATE utils CUDA::cublas CUDA::cudart)
