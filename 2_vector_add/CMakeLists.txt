add_executable(vector_add
    vector_add_main.cpp
    vector_add.cu
)

# Link against the utils library which brings in its include directory
target_link_libraries(vector_add PRIVATE utils CUDA::cudart)
