add_executable(test_shm_bank
    test_shm_bank_main.cpp
    test_shm_bank_kernel.cu
)

# Link against utils library and CUBLAS
target_link_libraries(test_shm_bank PRIVATE utils CUDA::cublas CUDA::cudart)
