#include <iostream>

__global__ void data_load_to_kernel(const float* data, size_t n, float* output) {
    size_t tid_x = threadIdx.x;
    size_t tid_y = threadIdx.y;
    size_t tid = tid_x + tid_y * blockDim.x;

    size_t block_base = blockIdx.x * (blockDim.x * blockDim.y);

    __shared__ float s_data[16 * 10][32];

    size_t data_idx = block_base + tid;
    if (data_idx > n) {
        return;
    }

    for (int i = 0; i < 10; ++i) {
        int x_base = tid_x * 4;
        if (x_base < 32) {
            *reinterpret_cast<float4*>(&s_data[16 * i + tid_y][x_base]) = *reinterpret_cast<const float4*>(&(data[block_base + tid_y * blockDim.x + x_base]));
        }
    }
    __syncthreads();

    float a = 0;
    int cnt = 0;
    for (int i = 0; i < 1000; ++i) {
        cnt = cnt < 10 ? cnt : 0;
        int sd_base = cnt * 32 * 16;
        a += reinterpret_cast<float*>(s_data)[sd_base + tid_y * blockDim.x + tid_x];
        // reinterpret_cast<float*>(s_data)[sd_base + tid_y * blockDim.x + tid_x] = a;
        // a += reinterpret_cast<float*>(s_data)[sd_base + tid_x * blockDim.y + tid_y];
        // reinterpret_cast<float*>(s_data)[sd_base + tid_x * blockDim.y + tid_y] = a;
        ++cnt;
    }

    output[data_idx] = a;
}

void data_load_to(const float* data, size_t n, float* output) {
    dim3 block_size(32, 16);
    size_t grid_size = (n + 32 * 16 - 1) / (32 * 16);
    data_load_to_kernel<<<grid_size, block_size>>>(data, n, output);
    cudaDeviceSynchronize();
}

