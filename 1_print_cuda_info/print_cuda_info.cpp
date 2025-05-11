#include <cstdlib>
#include <iostream>

#include <cuda_runtime.h>


int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: "
                  << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    std::cout << "CUDA device count: " << deviceCount << std::endl;

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaError_t err = cudaGetDeviceProperties(&prop, i);
        if (err != cudaSuccess) {
            std::cerr << "cudaGetDeviceProperties failed: "
                      << cudaGetErrorString(err) << std::endl;
            continue;
        }
        std::cout << "Device " << i << ": " << prop.name << std::endl;
        std::cout << "  totalGlobalMem: " << prop.totalGlobalMem / (1024 * 1024) << "MB" << std::endl;
        std::cout << "  sharedMemPerBlock: " << prop.sharedMemPerBlock / 1024 << "KB" << std::endl;
        std::cout << "  regsPerBlock: " << prop.regsPerBlock << std::endl;
        std::cout << "  maxThreadsPerBlock: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  warpSize: " << prop.warpSize << std::endl;
        std::cout << "  multiProcessorCount: " << prop.multiProcessorCount << std::endl;
        std::cout << "  sharedMemPerMultiprocessor: " << prop.sharedMemPerMultiprocessor / (1024) << "KB" << std::endl;
        std::cout << "  regsPerMultiprocessor: " << prop.regsPerMultiprocessor << std::endl;
        std::cout << "  maxThreadsPerMultiProcessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "  maxBlocksPerMultiProcessor: " << prop.maxBlocksPerMultiProcessor << std::endl;
        std::cout << "  compute capability: " << prop.major << "." << prop.minor << std::endl;
    }

    return 0;
}

