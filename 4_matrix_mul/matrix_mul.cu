
#include <cassert>
#include <iostream>

#define FLOAT4(d) (*(reinterpret_cast<float4*>(&(d))))

#define WARP_SIZE 32

template<int BK_M, int BK_N, int BK_K,
         int WM, int WN,
         int TM, int TN>
__global__ void sgemm_kernel_256(float* A, float* B, float* C,
                             int M, int N, int K) {
    __shared__ float smem_a[BK_K][BK_M];
    __shared__ float smem_b[BK_K][BK_N];

    int block_base_m = blockIdx.x * BK_M;
    int block_base_n = blockIdx.y * BK_N;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int warp_m = warp_id % (16 / WM);
    int warp_n = warp_id / (16 / WM);
    int warp_tid = tid % 32;
    int warp_tm = warp_tid % WM;
    int warp_tn = warp_tid / WM;

    int tm = warp_m * WM * TM + warp_tm * TM;
    int tn = warp_n * WN * TN + warp_tn * TN;

    float a_reg[TM];
    float b_reg[TN];
    float res[TN][TM] = {0.0};
    for (int k_base = 0; k_base < K; k_base += BK_K) {
         // load a;
         int local_m = (tid * 4) % BK_M;
         int local_k = (tid * 4) / BK_M;
         float4 tmp = FLOAT4(A[block_base_m + local_m + (k_base + local_k) * M]);
         smem_a[local_k][local_m] = tmp.x;
         smem_a[local_k][local_m + 1] = tmp.y;
         smem_a[local_k][local_m + 2] = tmp.z;
         smem_a[local_k][local_m + 3] = tmp.w;
 
         // local_b
         local_k = (tid * 4) % BK_K;
         int local_n = (tid * 4) / BK_K;
         tmp = FLOAT4(B[k_base + local_k + (block_base_n + local_n) * K]);
         smem_b[local_k][local_n] = tmp.x;
         smem_b[local_k + 1][local_n] = tmp.y;
         smem_b[local_k + 2][local_n] = tmp.z;
         smem_b[local_k + 3][local_n] = tmp.w;


        __syncthreads();

        // block: (128, 128)
        // warp: (4, 8)
        for (int k = 0; k < BK_K; k++) {
            for (int a = 0; a < 8; a++) {
                a_reg[a] = smem_a[k][tm + a];
            }
            for (int a = 0; a < 8; a++) {
                b_reg[a] = smem_b[k][tn + a];
            }
            __syncthreads();
            for (int a = 0; a < 8; a++) {
                for (int b = 0; b < 8; b++) {
                    res[a][b] += b_reg[a] * a_reg[b];
                }
            }
        }

        __syncthreads();
    }

    for(int a = 0; a < 8; a++) {
        int out_n = block_base_n + tn + a;
        for (int b = 0; b < 8; b+=4) {
            float4 tmp;
            tmp.x = res[a][b];
            tmp.y = res[a][b + 1];
            tmp.z = res[a][b + 2];
            tmp.w = res[a][b + 3];
            FLOAT4(C[block_base_m + tm + b + out_n * M]) = tmp;
        }
    }
}

void my_sgemm(float* A, float* B, float* C,
           int M, int N, int K) {
    assert(M % 128 == 0);
    assert(N % 128 == 0);
    assert(K % 8 == 0);
    dim3 grid_dim(M / 128, N / 128, 1);
    int block_dim = 256;
    sgemm_kernel_256<128, 128, 8,
                 2, 16,
                 8, 8><<<grid_dim, block_dim>>>(A, B, C, M, N, K);
}



template<int BLOCK_X, int BLOCK_Y, int TM, int TN, int SUB_K>
__global__ void sgemm_kernel_0(float* A, float* B, float* C,
                      int M, int N, int K) {
    __shared__ float a_block[SUB_K][BLOCK_X * TM];
    __shared__ float b_block[BLOCK_Y * TN][SUB_K];
    constexpr int a_shared_size = SUB_K * BLOCK_X * TM;
    constexpr int b_shared_size = SUB_K * BLOCK_Y * TN;

    constexpr int block_size = BLOCK_X * BLOCK_Y;
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int tid = tid_x + tid_y * BLOCK_X;

    int C_block_base_m = BLOCK_X * TM * blockIdx.x;
    int C_block_base_n = BLOCK_Y * TN * blockIdx.y;

    float c[TM][TN] = {0};
    float a_r[TM];
    float b_r[TN];
    for (int k_base = 0; k_base < K; k_base += SUB_K) {
        int m_offset, n_offset, k_offset;
        int m, n, k;
        int idx;

        // load A
        for (int base = 0; base < a_shared_size; base += block_size) {
            m_offset = (base + tid) % (BLOCK_X * TM);
            k_offset = (base + tid) / (BLOCK_X * TM);
            if (k_offset < SUB_K) {
                k = k_base + k_offset;
                m = C_block_base_m + m_offset;
                if (k < K && m < M) {
                    a_block[k_offset][m_offset] = A[k * M + m];
                } else {
                    a_block[k_offset][m_offset] = 0.f;
                }
            }
        }

        // load B
        for (int base = 0; base < b_shared_size; base += block_size) {
            k_offset = (base + tid) % SUB_K;
            n_offset = (base + tid) / SUB_K;
            if (n_offset < BLOCK_Y * TN) {
                k = k_base + k_offset;
                n = C_block_base_n + n_offset;
                if (k < K && n < N) {
                    b_block[n_offset][k_offset] = B[n * K + k];
                } else {
                    b_block[n][k] = 0.f;
                }
            }
        }
        __syncthreads();


        // compute
        for (int local_k = 0; local_k < SUB_K; ++local_k) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                a_r[i] = a_block[local_k][tid_x + i * BLOCK_X];
            }
            #pragma unroll
            for (int i = 0; i < TN; ++i) {
                b_r[i] = b_block[tid_y + i * BLOCK_Y][local_k];
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
            int m = C_block_base_m + BLOCK_X * i + tid_x;
            int n = C_block_base_n + BLOCK_Y * j + tid_y;
            C[m + n * M] = c[j][i];
        }
    }
}

template<int BLOCK_X, int BLOCK_Y, int TM, int TN, int SUB_K>
__global__ void sgemm_kernel_1(float* A, float* B, float* C,
                      int M, int N, int K) {
    __shared__ float a_block[SUB_K][BLOCK_X * TM];
    __shared__ float b_block[BLOCK_Y * TN][SUB_K];
    constexpr int a_shared_size = SUB_K * BLOCK_X * TM;
    constexpr int b_shared_size = SUB_K * BLOCK_Y * TN;

    constexpr int block_size = BLOCK_X * BLOCK_Y;
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int tid = tid_x + tid_y * BLOCK_X;

    int C_block_base_m = BLOCK_X * TM * blockIdx.x;
    int C_block_base_n = BLOCK_Y * TN * blockIdx.y;

    int warp_id = tid / WARP_SIZE;
    int warp_offset = tid % WARP_SIZE;

    float c[TM][TN] = {0};
    float a_r[TM];
    float b_r[TN];
    for (int k_base = 0; k_base < K; k_base += SUB_K) {
        int m_offset, n_offset, k_offset;
        int m, n, k;
        int offset;
        int idx;

        if (warp_offset < WARP_SIZE / 4) {
            offset = warp_id * WARP_SIZE + warp_offset * 4;
            // load A
            for (int base = 0; base < a_shared_size; base += block_size) {
                idx = base + offset;
                m_offset = idx % (BLOCK_X * TM);
                k_offset = idx / (BLOCK_X * TM);
                if (idx < a_shared_size) {
                    k = k_base + k_offset;
                    m = C_block_base_m + m_offset;
                    if (m < M && k < K) {
                        FLOAT4(a_block[k_offset][m_offset]) = FLOAT4(A[k * M + m]);
                    } else {
                        a_block[k_offset][m_offset] = 0.f;
                        a_block[k_offset][m_offset + 1] = 0.f;
                        a_block[k_offset][m_offset + 2] = 0.f;
                        a_block[k_offset][m_offset + 3] = 0.f;
                    }
                }
            }

            //load B
            for (int base = 0; base < b_shared_size; base += block_size) {
                idx = base + offset;
                k_offset = idx % SUB_K;
                n_offset = idx / SUB_K;
                if (idx < b_shared_size) {
                    k = k_base + k_offset;
                    n = C_block_base_n + n_offset;
                    if (n < N && k < K) {
                        FLOAT4(b_block[n_offset][k_offset]) = FLOAT4(B[k + n * K]);
                    } else {
                        b_block[n_offset][k_offset] = 0.f;
                        b_block[n_offset][k_offset + 1] = 0.f;
                        b_block[n_offset][k_offset + 2] = 0.f;
                        b_block[n_offset][k_offset + 3] = 0.f;
                    }
                }
            }

        }

        __syncthreads();


        // compute
        for (int local_k = 0; local_k < SUB_K; ++local_k) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                a_r[i] = a_block[local_k][tid_x + i * BLOCK_X];
            }
            #pragma unroll
            for (int i = 0; i < TN; ++i) {
                b_r[i] = b_block[tid_y + i * BLOCK_Y][local_k];
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
            int m = C_block_base_m + BLOCK_X * i + tid_x;
            int n = C_block_base_n + BLOCK_Y * j + tid_y;
            C[m + n * M] = c[j][i];
        }
    }
}

template<int BLOCK_X, int BLOCK_Y, int TM, int TN, int SUB_K>
__global__ void sgemm_kernel_2(float* A, float* B, float* C,
                      int M, int N, int K) {
    __shared__ float a_block[SUB_K][BLOCK_X * TM];
    __shared__ float b_block[BLOCK_Y * TN][SUB_K];
    constexpr int a_shared_size = SUB_K * BLOCK_X * TM;
    constexpr int b_shared_size = SUB_K * BLOCK_Y * TN;

    constexpr int block_size = BLOCK_X * BLOCK_Y;
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int tid = tid_x + tid_y * BLOCK_X;

    int C_block_base_m = BLOCK_X * TM * blockIdx.x;
    int C_block_base_n = BLOCK_Y * TN * blockIdx.y;

    float c[TM][TN] = {0};
    float a_r[TM];
    float b_r[TN];
    for (int k_base = 0; k_base < K; k_base += SUB_K) {
        int m_offset, n_offset, k_offset;
        int m, n, k;
        int idx;
        const int offset = tid * 4;

        // load A
        for (int base = 0; base < a_shared_size; base += block_size * 4) {
            idx = base + offset;
            m_offset = idx % (BLOCK_X * TM);
            k_offset = idx / (BLOCK_X * TM);
            if (k_offset < SUB_K) {
                k = k_base + k_offset;
                m = C_block_base_m + m_offset;
                if (k < K && m < M) {
                    FLOAT4(a_block[k_offset][m_offset]) = FLOAT4(A[k * M + m]);
                } else {
                    a_block[k_offset][m_offset] = 0.f;
                    a_block[k_offset][m_offset + 1] = 0.f;
                    a_block[k_offset][m_offset + 2] = 0.f;
                    a_block[k_offset][m_offset + 3] = 0.f;
                }
            }
        }

        // load B
        for (int base = 0; base < b_shared_size; base += block_size * 4) {
            idx = base + offset;
            k_offset = idx % SUB_K;
            n_offset = idx / SUB_K;
            if (n_offset < BLOCK_Y * TN) {
                k = k_base + k_offset;
                n = C_block_base_n + n_offset;
                if (k < K && n < N) {
                    FLOAT4(b_block[n_offset][k_offset]) = FLOAT4(B[n * K + k]);
                } else {
                    b_block[n][k] = 0.f;
                    b_block[n][k + 1] = 0.f;
                    b_block[n][k + 2] = 0.f;
                    b_block[n][k + 3] = 0.f;
                }
            }
        }
        __syncthreads();


        // compute
        for (int local_k = 0; local_k < SUB_K; ++local_k) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                a_r[i] = a_block[local_k][tid_x + i * BLOCK_X];
            }
            #pragma unroll
            for (int i = 0; i < TN; ++i) {
                b_r[i] = b_block[tid_y + i * BLOCK_Y][local_k];
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
            int m = C_block_base_m + BLOCK_X * i + tid_x;
            int n = C_block_base_n + BLOCK_Y * j + tid_y;
            C[m + n * M] = c[j][i];
        }
    }
}

template<int BLOCK_X, int BLOCK_Y, int TM, int TN, int SUB_K>
__global__ void sgemm_kernel_3(float* A, float* B, float* C,
                      int M, int N, int K) {
    __shared__ float a_block[2][SUB_K][BLOCK_X * TM];
    __shared__ float b_block[2][BLOCK_Y * TN][SUB_K];
    constexpr int a_shared_size = SUB_K * BLOCK_X * TM;
    constexpr int b_shared_size = SUB_K * BLOCK_Y * TN;
    constexpr int block_size = BLOCK_X * BLOCK_Y;
    const int tid_x = threadIdx.x;
    const int tid_y = threadIdx.y;
    const int tid = tid_x + tid_y * BLOCK_X;
    const int offset = tid * 4;

    int C_block_base_m = BLOCK_X * TM * blockIdx.x;
    int C_block_base_n = BLOCK_Y * TN * blockIdx.y;

    float c[TM][TN] = {0};
    float a_r[TM];
    float b_r[TN];

    int k_base = 0;
    int m_offset, n_offset, k_offset;
    int m, n, k;
    int idx;
    // pre load a b;
    // load A
    for (int base = 0; base < a_shared_size; base += block_size * 4) {
        idx = base + offset;
        m_offset = idx % (BLOCK_X * TM);
        k_offset = idx / (BLOCK_X * TM);
        if (k_offset < SUB_K) {
            k = k_base + k_offset;
            m = C_block_base_m + m_offset;
            if (k < K && m < M) {
                FLOAT4(a_block[0][k_offset][m_offset]) = FLOAT4(A[k * M + m]);
            } else {
                a_block[0][k_offset][m_offset] = 0.f;
                a_block[0][k_offset][m_offset + 1] = 0.f;
                a_block[0][k_offset][m_offset + 2] = 0.f;
                a_block[0][k_offset][m_offset + 3] = 0.f;
            }
        }
    }

    // load B
    for (int base = 0; base < b_shared_size; base += block_size * 4) {
        idx = base + offset;
        k_offset = idx % SUB_K;
        n_offset = idx / SUB_K;
        if (n_offset < BLOCK_Y * TN) {
            k = k_base + k_offset;
            n = C_block_base_n + n_offset;
            if (k < K && n < N) {
                FLOAT4(b_block[0][n_offset][k_offset]) = FLOAT4(B[n * K + k]);
            } else {
                b_block[0][n][k] = 0.f;
                b_block[0][n][k + 1] = 0.f;
                b_block[0][n][k + 2] = 0.f;
                b_block[0][n][k + 3] = 0.f;
            }
        }
    }
    __syncthreads();

    int cnt = 0;
    int buf_id;
    for (k_base = SUB_K; k_base < K; k_base += SUB_K) {

        ++cnt;
        buf_id = (cnt & 1);
        // load A
        for (int base = 0; base < a_shared_size; base += block_size * 4) {
            idx = base + offset;
            m_offset = idx % (BLOCK_X * TM);
            k_offset = idx / (BLOCK_X * TM);
            if (k_offset < SUB_K) {
                k = k_base + k_offset;
                m = C_block_base_m + m_offset;
                if (k < K && m < M) {
                    FLOAT4(a_block[buf_id][k_offset][m_offset]) = FLOAT4(A[k * M + m]);
                } else {
                    a_block[buf_id][k_offset][m_offset] = 0.f;
                    a_block[buf_id][k_offset][m_offset + 1] = 0.f;
                    a_block[buf_id][k_offset][m_offset + 2] = 0.f;
                    a_block[buf_id][k_offset][m_offset + 3] = 0.f;
                }
            }
        }

        // load B
        for (int base = 0; base < b_shared_size; base += block_size * 4) {
            idx = base + offset;
            k_offset = idx % SUB_K;
            n_offset = idx / SUB_K;
            if (n_offset < BLOCK_Y * TN) {
                k = k_base + k_offset;
                n = C_block_base_n + n_offset;
                if (k < K && n < N) {
                    FLOAT4(b_block[buf_id][n_offset][k_offset]) = FLOAT4(B[n * K + k]);
                } else {
                    b_block[buf_id][n][k] = 0.f;
                    b_block[buf_id][n][k + 1] = 0.f;
                    b_block[buf_id][n][k + 2] = 0.f;
                    b_block[buf_id][n][k + 3] = 0.f;
                }
            }

        }

        buf_id = ((cnt - 1) & 1);
        // compute
        for (int local_k = 0; local_k < SUB_K; ++local_k) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                a_r[i] = a_block[buf_id][local_k][tid_x + i * BLOCK_X];
            }
            #pragma unroll
            for (int i = 0; i < TN; ++i) {
                b_r[i] = b_block[buf_id][tid_y + i * BLOCK_Y][local_k];
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
    ++cnt;
    buf_id = ((cnt - 1) & 1);
    // compute
    for (int local_k = 0; local_k < SUB_K; ++local_k) {
        #pragma unroll
        for (int i = 0; i < TM; ++i) {
            a_r[i] = a_block[buf_id][local_k][tid_x + i * BLOCK_X];
        }
        #pragma unroll
        for (int i = 0; i < TN; ++i) {
            b_r[i] = b_block[buf_id][tid_y + i * BLOCK_Y][local_k];
        }
        #pragma unroll
        for (int i = 0; i < TM; ++i) {
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                c[j][i] += a_r[i] * b_r[j];
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            int m = C_block_base_m + BLOCK_X * i + tid_x;
            int n = C_block_base_n + BLOCK_Y * j + tid_y;
            C[m + n * M] = c[j][i];
        }
    }
}

__device__ constexpr int MAX(int a, int b) {
    return a > b ? a : b;
}

template<int BLOCK_X, int BLOCK_Y, int TM, int TN, int SUB_K>
__global__ void sgemm_kernel_4(float* A, float* B, float* C,
                      int M, int N, int K) {
    constexpr int shared_mem_size = 
        MAX(BLOCK_X * BLOCK_Y * TM * TN, 2 * (BLOCK_X * TM * SUB_K + BLOCK_Y * TN * SUB_K));
    __shared__ float _smem[shared_mem_size];

    // __shared__ float a_block[2][SUB_K][BLOCK_X * TM];
    // __shared__ float b_block[2][BLOCK_Y * TN][SUB_K];
    float (*a_block_ptr)[2][SUB_K][BLOCK_X * TM] =
        reinterpret_cast<float (*)[2][SUB_K][TM * BLOCK_X]>(_smem);
    float (&a_block)[2][SUB_K][BLOCK_X * TM] = *a_block_ptr;
    float (*b_block_ptr)[2][BLOCK_Y * TN][SUB_K] = 
        reinterpret_cast<float (*)[2][BLOCK_Y * TN][SUB_K]>(_smem + 2 * SUB_K * BLOCK_X * TM);
    float (&b_block)[2][BLOCK_Y * TN][SUB_K] = *b_block_ptr;


    constexpr int a_shared_size = SUB_K * BLOCK_X * TM;
    constexpr int b_shared_size = SUB_K * BLOCK_Y * TN;
    constexpr int block_size = BLOCK_X * BLOCK_Y;
    const int tid_x = threadIdx.x;
    const int tid_y = threadIdx.y;
    const int tid = tid_x + tid_y * BLOCK_X;
    const int offset = tid * 4;

    int C_block_base_m = BLOCK_X * TM * blockIdx.x;
    int C_block_base_n = BLOCK_Y * TN * blockIdx.y;

    float c[TM][TN] = {0};
    float a_r[TM];
    float b_r[TN];

    int k_base = 0;
    int m_offset, n_offset, k_offset;
    int m, n, k;
    int idx;
    // pre load a b;
    // load A
    for (int base = 0; base < a_shared_size; base += block_size * 4) {
        idx = base + offset;
        m_offset = idx % (BLOCK_X * TM);
        k_offset = idx / (BLOCK_X * TM);
        if (k_offset < SUB_K) {
            k = k_base + k_offset;
            m = C_block_base_m + m_offset;
            if (k < K && m < M) {
                FLOAT4(a_block[0][k_offset][m_offset]) = FLOAT4(A[k * M + m]);
            } else {
                a_block[0][k_offset][m_offset] = 0.f;
                a_block[0][k_offset][m_offset + 1] = 0.f;
                a_block[0][k_offset][m_offset + 2] = 0.f;
                a_block[0][k_offset][m_offset + 3] = 0.f;
            }
        }
    }

    // load B
    for (int base = 0; base < b_shared_size; base += block_size * 4) {
        idx = base + offset;
        k_offset = idx % SUB_K;
        n_offset = idx / SUB_K;
        if (n_offset < BLOCK_Y * TN) {
            k = k_base + k_offset;
            n = C_block_base_n + n_offset;
            if (k < K && n < N) {
                FLOAT4(b_block[0][n_offset][k_offset]) = FLOAT4(B[n * K + k]);
            } else {
                b_block[0][n][k] = 0.f;
                b_block[0][n][k + 1] = 0.f;
                b_block[0][n][k + 2] = 0.f;
                b_block[0][n][k + 3] = 0.f;
            }
        }
    }
    __syncthreads();

    int cnt = 0;
    int buf_id;
    for (k_base = SUB_K; k_base < K; k_base += SUB_K) {

        ++cnt;
        buf_id = (cnt & 1);
        // load A
        for (int base = 0; base < a_shared_size; base += block_size * 4) {
            idx = base + offset;
            m_offset = idx % (BLOCK_X * TM);
            k_offset = idx / (BLOCK_X * TM);
            if (k_offset < SUB_K) {
                k = k_base + k_offset;
                m = C_block_base_m + m_offset;
                if (k < K && m < M) {
                    FLOAT4(a_block[buf_id][k_offset][m_offset]) = FLOAT4(A[k * M + m]);
                } else {
                    a_block[buf_id][k_offset][m_offset] = 0.f;
                    a_block[buf_id][k_offset][m_offset + 1] = 0.f;
                    a_block[buf_id][k_offset][m_offset + 2] = 0.f;
                    a_block[buf_id][k_offset][m_offset + 3] = 0.f;
                }
            }
        }

        // load B
        for (int base = 0; base < b_shared_size; base += block_size * 4) {
            idx = base + offset;
            k_offset = idx % SUB_K;
            n_offset = idx / SUB_K;
            if (n_offset < BLOCK_Y * TN) {
                k = k_base + k_offset;
                n = C_block_base_n + n_offset;
                if (k < K && n < N) {
                    FLOAT4(b_block[buf_id][n_offset][k_offset]) = FLOAT4(B[n * K + k]);
                } else {
                    b_block[buf_id][n][k] = 0.f;
                    b_block[buf_id][n][k + 1] = 0.f;
                    b_block[buf_id][n][k + 2] = 0.f;
                    b_block[buf_id][n][k + 3] = 0.f;
                }
            }

        }

        buf_id = ((cnt - 1) & 1);
        // compute
        for (int local_k = 0; local_k < SUB_K; ++local_k) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                a_r[i] = a_block[buf_id][local_k][tid_x + i * BLOCK_X];
            }
            #pragma unroll
            for (int i = 0; i < TN; ++i) {
                b_r[i] = b_block[buf_id][tid_y + i * BLOCK_Y][local_k];
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
    ++cnt;
    buf_id = ((cnt - 1) & 1);
    // compute
    for (int local_k = 0; local_k < SUB_K; ++local_k) {
        #pragma unroll
        for (int i = 0; i < TM; ++i) {
            a_r[i] = a_block[buf_id][local_k][tid_x + i * BLOCK_X];
        }
        #pragma unroll
        for (int i = 0; i < TN; ++i) {
            b_r[i] = b_block[buf_id][tid_y + i * BLOCK_Y][local_k];
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

    float (*res_block_ptr)[BLOCK_Y * TN][BLOCK_X * TM] =
        reinterpret_cast<float (*)[BLOCK_Y * TN][BLOCK_X * TM]>(_smem);
    float (&res_block)[BLOCK_Y * TN][BLOCK_X * TM] = *res_block_ptr;

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int m = BLOCK_X * i + tid_x;
            int n = BLOCK_Y * j + tid_y;
            res_block[n][m] = c[j][i];
        }

    }

    #pragma unroll
    for (int i = 0; i < TN; ++i) {
        int m = C_block_base_m + tid_x * 4;
        int n = C_block_base_n + BLOCK_Y * i + tid_y;
            FLOAT4(C[m + n * M]) = FLOAT4(res_block[BLOCK_Y * i + tid_y][tid_x * 4]);
    }

}

const int BLOCK_X = 32;
const int BLOCK_Y = 16;
const int SUB_K = 8;
const int TM = 8;
const int TN = 8;

void sgemm_gpu(float* A, float* B, float* C,
               int M, int K, int N) {
    dim3 block_size(BLOCK_X, BLOCK_Y);
    dim3 grid_size((M + BLOCK_X * TM - 1) / (BLOCK_X * TM), (N + BLOCK_Y * TN - 1) / (BLOCK_Y * TN));

    sgemm_kernel_3<BLOCK_X, BLOCK_Y, TM, TN, SUB_K><<<grid_size, block_size>>>
        (A, B, C, M, N, K);
}

