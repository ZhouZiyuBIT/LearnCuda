add_library(utils SHARED
    device_data.cpp
    host_data.cpp
    random_generator_int.cu
    warmup.cu
)

# Export the include directory for anything linking against this library
target_include_directories(utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Link against CUDA runtime
target_link_libraries(utils PRIVATE CUDA::cudart)
