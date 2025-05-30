
#include <iostream>

#define FLOAT4(d) (*(reinterpret_cast<float4*>(&(d))))

#define WARP_SIZE 32

template<int BLOCK_X, int BLOCK_Y, int TM, int TN, int SUB_K>
__global__ void sgemm_kernel(float* A, float* B, float* C,
                      size_t M, size_t N, size_t K) {
    __shared__ float a_block[SUB_K][BLOCK_X * TM];
    __shared__ float b_block[BLOCK_Y * TN][SUB_K];
    const size_t a_shared_size = SUB_K * BLOCK_X * TM;
    const size_t b_shared_size = SUB_K * BLOCK_Y * TN;

    const size_t block_size = BLOCK_X * BLOCK_Y;
    size_t tid_x = threadIdx.x;
    size_t tid_y = threadIdx.y;
    size_t tid = tid_x + tid_y * BLOCK_X;

    size_t C_block_base_m = BLOCK_X * TM * blockIdx.x;
    size_t C_block_base_n = BLOCK_Y * TN * blockIdx.y;

    size_t t_m_base = C_block_base_m + tid_x * TM;
    size_t t_n_base = C_block_base_n + tid_y * TN;

    float c[TM][TN] = {0};
    float a_r[TM];
    float b_r[TN];
    for (int k_base = 0; k_base < K; k_base += SUB_K) {
        size_t m_offset, n_offset, k_offset;
        size_t m, n, k;
        size_t idx;

        size_t warp_id = tid / WARP_SIZE;
        size_t warp_offset = tid % WARP_SIZE;
        if (warp_offset < WARP_SIZE / 4) {
            idx = warp_id * WARP_SIZE + warp_offset * 4;
            // load_a
            if (idx < a_shared_size) {
                m_offset = idx % (BLOCK_X * TM);
                k_offset = idx / (BLOCK_X * TM);
                // load_a
                k = k_base + k_offset;
                m = C_block_base_m + m_offset;
                if (m < M && k < K) {
                    float4 temp = FLOAT4(A[m + k * M]);
                    a_block[k_offset][m_offset] = temp.x;
                    a_block[k_offset][m_offset + 1] = temp.y;
                    a_block[k_offset][m_offset + 2] = temp.z;
                    a_block[k_offset][m_offset + 3] = temp.w;

                } else {
                    a_block[k_offset][m_offset] = 0.f;
                    a_block[k_offset][m_offset + 1] = 0.f;
                    a_block[k_offset][m_offset + 2] = 0.f;
                    a_block[k_offset][m_offset + 3] = 0.f;
                }
            }

            //load_b
            if (idx < b_shared_size) {
                k_offset = idx % SUB_K;
                n_offset = idx / SUB_K;
                k = k_base + k_offset;
                n = C_block_base_n + n_offset;
                if (n < N && k < K) {
                    float4 temp = FLOAT4(B[k + n * K]);
                    b_block[n_offset][k_offset] = temp.x;
                    b_block[n_offset][k_offset + 1] = temp.y;
                    b_block[n_offset][k_offset + 2] = temp.z;
                    b_block[n_offset][k_offset + 3] = temp.w;
                } else {
                    b_block[n_offset][k_offset] = 0.f;
                    b_block[n_offset][k_offset + 1] = 0.f;
                    b_block[n_offset][k_offset + 2] = 0.f;
                    b_block[n_offset][k_offset + 3] = 0.f;
                }
            }

        }

        __syncthreads();


        // compute
        for (int local_k = 0; local_k < SUB_K; ++local_k) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                a_r[i] = a_block[local_k][tid_x * TM + i];
            }
            #pragma unroll
            for (int i = 0; i < TN; ++i) {
                b_r[i] = b_block[tid_y * TN + i][local_k];
            }
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    c[j][i] += a_r[i] * b_r[j];
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            C[t_m_base + i + (t_n_base + j) * M] = c[j][i];
        }
    }
}

const size_t MAX_SHARED_PER_BLOCK = ((1 << 10) * 100) / 6;
const size_t BLOCK_X = 32;
const size_t BLOCK_Y = 16;
const size_t SUB_K = 4;
const size_t TM = 2;
const size_t TN = 2;

void sgemm_gpu(float* A, float* B, float* C,
               size_t M, size_t K, size_t N) {
    dim3 block_size(BLOCK_X, BLOCK_Y);
    dim3 grid_size((M + BLOCK_X * TM - 1) / (BLOCK_X * TM), (N + BLOCK_Y * TN - 1) / (BLOCK_Y * TN));

    sgemm_kernel<BLOCK_X, BLOCK_Y, TM, TN, SUB_K><<<grid_size, block_size>>>
        (A, B, C, M, N, K);
}

