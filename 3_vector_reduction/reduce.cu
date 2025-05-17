
template<int BLOCK_DIM_X>
__global__ void int_sum_kernel(const int* v, const size_t N, const size_t m, int* out) {
    __shared__ int s_data[BLOCK_DIM_X * sizeof(int)];
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int block_base_idx = bid * blockDim.x;

    int data_block_base_idx = block_base_idx * m;

    s_data[tid] = 0;
    for (size_t i = 0; i < m; ++i) {
        s_data[tid] += v[data_block_base_idx + tid * m + i];
    }
    __syncthreads();

    for (size_t ds = (BLOCK_DIM_X >> 1); ds >= 1; ds >>= 1) {
        if (tid < ds) {
            s_data[tid] += s_data[tid + ds];
        }
        __syncthreads();
    }
    out[bid] = s_data[0];
}


int* d_out = nullptr;
int* h_out = nullptr;
int64_t int_sum_gpu(const int* v, const size_t N) {
    const size_t BLOCK_DIM_X = 512;
    const size_t m = 8;

    size_t block_num = N / (BLOCK_DIM_X * m); // TODO: for now, SD | N
    if (d_out == nullptr) {
        cudaMalloc(&d_out, block_num * sizeof(int));
    }
    int_sum_kernel<BLOCK_DIM_X><<<block_num, BLOCK_DIM_X>>>(v, N, m, d_out);
    if (h_out == nullptr) {
        h_out = new int[block_num * sizeof(int)];
    }
    cudaMemcpy(h_out, d_out, block_num * sizeof(int), cudaMemcpyDeviceToHost);
    int64_t sum = 0;
    for (size_t i = 0; i < block_num; ++i) {
        sum += h_out[i];
    }
    return sum;
}

