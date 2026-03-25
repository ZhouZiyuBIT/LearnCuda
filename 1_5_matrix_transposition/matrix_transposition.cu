
#include <cassert>
// copy
// max memory bound

__global__ void copy_kernel(float* A, float* B, int N) {

    int global_index = (blockIdx.x * blockDim.x + threadIdx.x);
    B[global_index] = A[global_index];
}
void copy(float* A, float* B, int N) {
    const int block_size = 512;
    assert(N % block_size == 0);

    const int grid_size = N / block_size;
    copy_kernel<<<grid_size, block_size>>>(A, B, N);
}

__global__ void copy_float4_kernel(float* A, float* B, int N) {

    int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    reinterpret_cast<float4*>(B)[global_index] = reinterpret_cast<float4*>(A)[global_index];

    // int global_index = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    // *reinterpret_cast<float4*>(B + global_index) = *reinterpret_cast<float4*>(A + global_index);
}
void copy_float4(float* A, float* B, int N) {
    const int block_size = 512;
    assert(N % (block_size * 4) == 0);
    const int grid_size = N / (block_size * 4);
    copy_float4_kernel<<<grid_size, block_size>>>(A, B, N);
}


// naive
__global__ void matrix_transposition_naive_kernel(float* A, float* B, int M, int N) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;

    B[n + m * N] = A[m + n * M];
}
void matrix_transposition_naive(float* A, float* B, int M, int N) {
    assert(M % 32 == 0);
    assert(N % 16 == 0);

    dim3 block_dim(32, 16, 1);
    dim3 grid_dim(M / 32, N / 16, 1);
    matrix_transposition_naive_kernel<<<grid_dim, block_dim>>>(A, B, M, N);
}

__global__ void matrix_transposition_shared_32_32_kernel(float* A, float* B, int M, int N) {
    __shared__ float s_mem[32][32];
    int A_m = blockIdx.x * blockDim.x + threadIdx.x;
    int A_n = blockIdx.y * blockDim.y + threadIdx.y;

    s_mem[threadIdx.y][threadIdx.x] = A[A_m + A_n * M];

    __syncthreads();

    int B_m = blockIdx.y * blockDim.y + threadIdx.x;
    int B_n = blockIdx.x * blockDim.x + threadIdx.y;
    B[B_m + B_n * N] = s_mem[threadIdx.x][threadIdx.y];
}
void matrix_transpostion_shared_32_32(float* A, float* B, int M, int N) {
    assert(M % 32 == 0);
    assert(N % 32 == 0);

    dim3 block_dim(32, 32);
    dim3 grid_dim(M / 32, N / 32);

    matrix_transposition_shared_32_32_kernel<<<grid_dim, block_dim>>>(A, B, M, N);
}

__global__ void matrix_transposition_shared_no_bank_conflict_32_32_kernel(float* A, float* B, int M, int N) {
    __shared__ float s_mem[32][33];
    int A_m = blockIdx.x * blockDim.x + threadIdx.x;
    int A_n = blockIdx.y * blockDim.y + threadIdx.y;

    s_mem[threadIdx.y][threadIdx.x] = A[A_m + A_n * M];

    __syncthreads();

    int B_m = blockIdx.y * blockDim.y + threadIdx.x;
    int B_n = blockIdx.x * blockDim.x + threadIdx.y;
    B[B_m + B_n * N] = s_mem[threadIdx.x][threadIdx.y];
}
void matrix_transpostion_shared_no_bank_conflict_32_32(float* A, float* B, int M, int N) {
    assert(M % 32 == 0);
    assert(N % 32 == 0);

    dim3 block_dim(32, 32);
    dim3 grid_dim(M / 32, N / 32);

    matrix_transposition_shared_no_bank_conflict_32_32_kernel<<<grid_dim, block_dim>>>(A, B, M, N);
}

__global__ void matrix_transposition_shared_32_8_kernel(float* A, float* B, int M, int N) {
    __shared__ float s_mem[32][32];
    int A_m = blockIdx.x * blockDim.x + threadIdx.x;
    int A_n = (blockIdx.y * blockDim.y + threadIdx.y) * 4;

    for (int i = 0; i < 4; i++) {
        s_mem[threadIdx.y * 4 + i][threadIdx.x] = A[A_m + (A_n + i) * M];
    }

    __syncthreads();

    int B_m = blockIdx.y * blockDim.y * 4 + threadIdx.x;
    int B_n = blockIdx.x * blockDim.x + threadIdx.y * 4;
    for (int i = 0; i < 4; i++) {
        B[B_m + (B_n + i) * N] = s_mem[threadIdx.x][threadIdx.y * 4 + i];
    }
}
void matrix_transpostion_shared_32_8(float* A, float* B, int M, int N) {
    assert(M % 32 == 0);
    assert(N % 8 == 0);

    dim3 block_dim(32, 8);
    dim3 grid_dim(M / 32, N / 32);

    matrix_transposition_shared_32_8_kernel<<<grid_dim, block_dim>>>(A, B, M, N);
}

__global__ void matrix_transposition_shared_no_bank_conflict_32_8_kernel(float* A, float* B, int M, int N) {
    __shared__ float s_mem[32][33];
    int A_m = blockIdx.x * blockDim.x + threadIdx.x;
    int A_n = (blockIdx.y * blockDim.y + threadIdx.y) * 4;

    for (int i = 0; i < 4; i++) {
        s_mem[threadIdx.y * 4 + i][threadIdx.x] = A[A_m + (A_n + i) * M];
    }

    __syncthreads();

    int B_m = blockIdx.y * blockDim.y * 4 + threadIdx.x;
    int B_n = blockIdx.x * blockDim.x + threadIdx.y * 4;
    for (int i = 0; i < 4; i++) {
        B[B_m + (B_n + i) * N] = s_mem[threadIdx.x][threadIdx.y * 4 + i];
    }
}
void matrix_transpostion_shared_no_bank_conflict_32_8(float* A, float* B, int M, int N) {
    assert(M % 32 == 0);
    assert(N % 8 == 0);

    dim3 block_dim(32, 8);
    dim3 grid_dim(M / 32, N / 32);

    matrix_transposition_shared_no_bank_conflict_32_8_kernel<<<grid_dim, block_dim>>>(A, B, M, N);
}
