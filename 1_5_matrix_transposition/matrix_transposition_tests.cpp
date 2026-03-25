#include <iostream>

#include "host_data.h"
#include "device_data.h"
#include "time_statistics.h"

void copy(float* A, float* B, int N);
void copy_float4(float* A, float* B, int N);
void matrix_transposition_naive(float* A, float* B, int M, int N);
void matrix_transpostion_shared_32_32(float* A, float* B, int M, int N);
void matrix_transpostion_shared_no_bank_conflict_32_32(float* A, float* B, int M, int N);
void matrix_transpostion_shared_32_8(float* A, float* B, int M, int N);
void matrix_transpostion_shared_no_bank_conflict_32_8(float* A, float* B, int M, int N);

void transpositon(float* A, float* B, int M, int N) {
    for (int n = 0; n < N; n++) {
        for (int m = 0; m < M; m++) {
            B[n + m * N] = A[m + n * M];
        }
    }
}

// 采用不是大2的次方的维度，打破 Partition Camping (分区竞争)
const int M = 10240;
const int N = 10240;

int main() {
    HostData<float> A_h(M * N);
    A_h.random_init(-10.0, 10.0);
    DeviceData<float> A_d(M * N);
    A_d = A_h;

    HostData<float> B_h(M * N);
    DeviceData<float> B_d(M * N);
    HostData<float> res_h(M * N);

    std::cout << "--- 正在 CPU 上计算标准转置结果 (这可能需要几秒钟)... ---" << std::endl;
    transpositon(A_h.data(), res_h.data(), M, N);
    std::cout << "--- CPU 计算完毕，开始 GPU 性能测试 ---" << std::endl;

    std::cout << "********************* Test [copy] *×××××××××××××××××××××××××××" << std::endl;
    copy(A_d.data(), B_d.data(), M * N);
    B_h = B_d;
    if (!(A_h == B_h)) {
        std::cout << "Result error." << std::endl;
    }
    CUDA_TIME_USED(10, [&]() {
        copy(A_d.data(), B_d.data(), M * N);
    }).print("");

    std::cout << "********************* Test [copy_float4] *×××××××××××××××××××××××××××" << std::endl;
    copy_float4(A_d.data(), B_d.data(), M * N);
    B_h = B_d;
    if (!(A_h == B_h)) {
        std::cout << "Result error." << std::endl;
    }
    auto time_res = CUDA_TIME_USED(10, [&]() {
        copy_float4(A_d.data(), B_d.data(), M * N);
    });
    time_res.print("");
    std::cout << "max mem bandwidth: 896 GB/s" << std::endl;
    std::cout << "test mem bandwidth: " << M * (double)N * 4 * 2 / 1024.0 / 1024.0 / 1024.0 / time_res.mean * 1000 << " GB/s" << std::endl;

    std::cout << "********************* Test [transposition naive] *×××××××××××××××××××××××××××" << std::endl;
    matrix_transposition_naive(A_d.data(), B_d.data(), M, N);
    B_h = B_d;
    if (!(B_h == res_h)) {
        std::cout << "naive, Result error" << std::endl;
    }
    CUDA_TIME_USED(10, [&](){
        matrix_transposition_naive(A_d.data(), B_d.data(), M, N);
    }).print("naive, ");


    std::cout << "********************* Test [transposition shared_32_32] *×××××××××××××××××××××××××××" << std::endl;
    matrix_transpostion_shared_32_32(A_d.data(), B_d.data(), M, N);
    B_h = B_d;
    if (!(B_h == res_h)) {
        std::cout << "shared_32_32, Result error" << std::endl;
    }

    CUDA_TIME_USED(10, [&]() {
        matrix_transpostion_shared_32_32(A_d.data(), B_d.data(), M, N);
    }).print("shared_32_32");


    std::cout << "********************* Test [transposition shared_32_32 no bank conflict] *×××××××××××××××××××××××××××" << std::endl;
    matrix_transpostion_shared_no_bank_conflict_32_32(A_d.data(), B_d.data(), M, N);
    B_h = B_d;
    if (!(B_h == res_h)) {
        std::cout << "shared_32_32, Result error" << std::endl;
    }

    CUDA_TIME_USED(10, [&]() {
        matrix_transpostion_shared_no_bank_conflict_32_32(A_d.data(), B_d.data(), M, N);
    }).print("shared_no_bank_conflict_32_32");


    std::cout << "********************* Test [transposition shared_32_8] *×××××××××××××××××××××××××××" << std::endl;
    matrix_transpostion_shared_32_8(A_d.data(), B_d.data(), M, N);
    B_h = B_d;
    if (!(B_h == res_h)) {
        std::cout << "shared_32_8, Result error" << std::endl;
    }

    CUDA_TIME_USED(10, [&]() {
        matrix_transpostion_shared_32_8(A_d.data(), B_d.data(), M, N);
    }).print("shared_32_8");


    std::cout << "********************* Test [transposition shared_32_8 no bank conflict] *×××××××××××××××××××××××××××" << std::endl;
    matrix_transpostion_shared_no_bank_conflict_32_8(A_d.data(), B_d.data(), M, N);
    B_h = B_d;
    if (!(B_h == res_h)) {
        std::cout << "shared_32_8, Result error" << std::endl;
    }

    CUDA_TIME_USED(10, [&]() {
        matrix_transpostion_shared_no_bank_conflict_32_8(A_d.data(), B_d.data(), M, N);
    }).print("shared_no_bank_conflict_32_8");

    return 0;
}
