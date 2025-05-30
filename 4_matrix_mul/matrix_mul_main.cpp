#include "time_statistics.h"
#include "host_data.h"
#include "device_data.h"

#include <cublas_v2.h>

const size_t _K = (1 << 10);
const size_t _M = (1 << 20);

const size_t Mat_M = 1024;
const size_t Mat_K = 512;
const size_t Mat_N = 512;

void mat_mul_cpu(float* A, float* B, float* C, size_t M, size_t K, size_t N) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double c = 0.0;
            for (int k = 0; k < K; ++k) {
                    c += A[m + k * M] * B[k + n * K];
            }
            C[m + n * M] = c;
        }
    }
}

void mat_mul_cublas(float* A, float* B, float* C, size_t M, size_t K, size_t N) {
    static cublasHandle_t handle = nullptr;
    if (handle == nullptr) {
        cublasCreate(&handle);
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                M,
                N,
                K,
                &alpha,
                A, M,
                B, K,
                &beta,
                C, M);
}

void sgemm_gpu(float* A, float* B, float* C,
               size_t M, size_t K, size_t N);

int main() {
    HostData<float> h_A(Mat_M * Mat_K);
    HostData<float> h_B(Mat_K * Mat_N);
    HostData<float> h_C(Mat_M * Mat_N);
    HostData<float> h_C_res(Mat_M * Mat_N);
    h_A.random_init(0.f, 1.f);
    h_B.random_init(0.f, 1.f);

    DeviceData<float> d_A(Mat_M * Mat_K);
    DeviceData<float> d_B(Mat_K * Mat_N);
    DeviceData<float> d_C(Mat_M * Mat_N);
    mat_mul_cublas(d_A.data(), d_B.data(), d_C.data(), Mat_M, Mat_K, Mat_N);
    d_A = h_A;
    d_B = h_B;

    TIME_USED(5, [&]() {
        mat_mul_cpu(h_A.data(), h_B.data(), h_C_res.data(), Mat_M, Mat_K, Mat_N);
    }).print("cpu mat_mul");

    TIME_USED(1, [&]() {
        mat_mul_cublas(d_A.data(), d_B.data(), d_C.data(), Mat_M, Mat_K, Mat_N);
    }).print("cublas mat_mul");
    h_C = d_C;
    std::cout << "cublas res check: " << (h_C == h_C_res) << std::endl;

    sgemm_gpu(d_A.data(), d_B.data(), d_C.data(), Mat_M, Mat_K, Mat_N);
    TIME_USED(1, [&]() {
        sgemm_gpu(d_A.data(), d_B.data(), d_C.data(), Mat_M, Mat_K, Mat_N);
        cudaDeviceSynchronize();
    }).print("gpu mat_mul");
    h_C = d_C;
    std::cout << "gpu res check: " << (h_C == h_C_res) << std::endl;

    return 0;
}

