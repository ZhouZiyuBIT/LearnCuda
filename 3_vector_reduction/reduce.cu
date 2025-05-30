
template<int BLOCK_DIM_X, int DATA_PER_THREAD>
__global__ void sum_kernel(const float* v, const size_t N, float* out) {
    __shared__ float s_data[BLOCK_DIM_X];
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int block_base_idx = bid * blockDim.x;

    int data_block_base_idx = block_base_idx * DATA_PER_THREAD;

    float sum = 0;
    // #pragma unroll
    for (int i = 0; i < DATA_PER_THREAD; ++i) {
        int data_idx = data_block_base_idx + i * blockDim.x + tid;
        if (data_idx < N) {
            sum += v[data_idx];
        }
    }
    s_data[tid] = sum;
    __syncthreads();

    #pragma unroll
    for (int ds = (BLOCK_DIM_X / 2); ds > 0; ds >>= 1) {
        if (tid < ds) {
            s_data[tid] += s_data[tid + ds];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[bid] = s_data[0];
    }
}


const size_t MultiProcessorCount = 70;

float* d_out = nullptr;
float* h_out = nullptr;
float sum_gpu(const float* v, const size_t N) {
    const size_t BLOCK_DIM_X = 512;
    const size_t m = 1024;

    size_t block_num = (N + BLOCK_DIM_X * m - 1) / (BLOCK_DIM_X * m);

    if (d_out == nullptr) {
        cudaMalloc(&d_out, block_num * sizeof(float));
    }
    sum_kernel<BLOCK_DIM_X, m><<<block_num, BLOCK_DIM_X>>>(v, N, d_out);
    if (h_out == nullptr) {
        h_out = new float[block_num * sizeof(float)];
    }
    cudaMemcpy(h_out, d_out, block_num * sizeof(float), cudaMemcpyDeviceToHost);
    double sum = 0;
    for (int i = 0; i < block_num; ++i) {
        sum += h_out[i];
    }
    return sum;
}

