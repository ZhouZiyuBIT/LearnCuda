add_executable(vector_reduction
    vector_reduction_main.cpp
    reduce.cu
)

# Link against utils library and CUBLAS
target_link_libraries(vector_reduction PRIVATE utils CUDA::cublas CUDA::cudart)
