#include <cuda_runtime.h>
#include <cstdio>

__global__ void warmup_kernel() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (int i = 0; i < 1000; ++i) {
        sum += (float)(idx + i);
    }
    if (sum > 1e20f) {
        printf("Warmup: %f\n", sum);
    }
}

void warmup_gpu() {
    // 启动足够多的线程，跑1000次空转计算，彻底唤醒GPU进入P0状态
    warmup_kernel<<<1024, 1024>>>();
    cudaDeviceSynchronize();
}
