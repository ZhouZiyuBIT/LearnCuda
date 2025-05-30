
#include <curand.h>
#include <curand_kernel.h>

__global__ void random_uniform_int_kernel(const int min, const int max, int* x, const size_t tb, const size_t N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * tb;
    if (idx >= N)
        return;

    curandState rnd_state;
    curand_init(clock64(), idx, 0, &rnd_state);
    for (size_t i = idx; i < idx + tb; ++i) {
        x[i] = curand_uniform(&rnd_state) * (max - min) + min;
    }
}

__global__ void random_uniform_float_kernel(const float min, const float max, float* x, const size_t tb, const size_t N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * tb;
    if (idx >= N) {
        return;
    }

    curandState rnd_state;
    curand_init(clock64(), idx, 0, &rnd_state);
    for (int i = idx; i < idx + tb; ++i) {
        x[i] = curand_uniform(&rnd_state) * (max - min) + min;
    }
}

void random_uniform(const int min, const int max, int* x, const size_t N) {
    const size_t block_size = 512;
    const size_t tb = 128;
    const size_t grid_size = N / (block_size * tb) + 1;
    random_uniform_int_kernel<<<grid_size, block_size>>>(min, max, x, tb, N);
}

void random_uniform(const float min, const float max, float* x, const size_t N) {
    const size_t block_size = 1024;
    const size_t tb = 128;
    const size_t grid_size = N / (block_size * tb) + 1;
    random_uniform_float_kernel<<<grid_size, block_size>>>(min, max, x, tb, N);
}

