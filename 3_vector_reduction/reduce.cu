
#include <__clang_cuda_builtin_vars.h>
__global__ void int_sum_kernel(const int* v, const size_t N, const size_t sd_size, int* out) {
    int bid = blockIdx.x;
    int block_base_idx = bid * blockDim.x;
    int tid = threadIdx.x;
    int idx = block_base_idx + tid;
    __shared__ extern int s_data[];

    size_t m = sd_size / blockDim.x;
    for (size_t i = 0; i < m; ++i) {
        s_data[tid * m + i] = v[idx * m + i];
    }
    __syncthreads();

    for (size_t ds = sd_size - blockDim.x; ds >= 1; ds -= blockDim.x) {
        if (tid < ds) {
            s_data[tid] += s_data[tid + ds];
        }
        __syncthreads();
    }
    out[bid] = s_data[0];
}


int64_t int_sum_gpu(const int* v, const size_t N) {
    const size_t block_size = 512;
    const size_t sd_size = 1024;

    size_t block_num = N / sd_size; // TODO: for now, SD | N
    int* d_out = nullptr;
    cudaMalloc(&d_out, block_num * sizeof(int));
    int_sum_kernel<<<block_num, block_size, sd_size * sizeof(int)>>>(v, N, sd_size, d_out);
    int* h_out = new int[block_num * sizeof(int)];
    cudaMemcpy(h_out, d_out, block_num * sizeof(int), cudaMemcpyDeviceToHost);
    int64_t sum = 0;
    for (size_t i = 0; i < block_num; ++i) {
        sum += h_out[i];
    }
    return sum;
}

