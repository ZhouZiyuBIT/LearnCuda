add_executable(print_cuda_info
    print_cuda_info.cpp
)

target_link_libraries(print_cuda_info PRIVATE CUDA::cudart)
