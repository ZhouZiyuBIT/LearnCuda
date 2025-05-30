
#include <iomanip>
#include <iostream>
#include <vector>
#include <cmath>

#include <cublas_v2.h>

#include "device_data.h"
#include "host_data.h"
#include "time_statistics.h"

float mean(const HostData<float>& h_d) {
    double sum = 0;
    size_t N = h_d.size();
    for (size_t i = 0; i < N; ++i) {
        sum += h_d.data()[i];
    }
    float mean = sum / N;
    return mean;
}

float std_dev(const HostData<float>& h_d, float mean) {
    float sum = 0;
    size_t N = h_d.size();
    for (size_t i = 0; i < N; ++i) {
        float e = h_d.data()[i] - mean;
        sum += std::sqrt(e * e);
    }
    float dev = sum / N;
    return dev;
}

float sum_gpu(const float* v, const size_t N);

float sum_cublas(const float* v, const size_t N) {
    static float* d_ones = nullptr;
    if (d_ones == nullptr) {
        std::vector<float> h_ones(N, 1.0f);
        cudaMalloc(&d_ones, N * sizeof(float));
        cudaMemcpy(d_ones, h_ones.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    }

    float sum = 0;
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    cublasSdot(cublas_handle, N, v, 1, d_ones, 1, &sum);

    return sum;
}

int main() {
    size_t N = (1 << 20) * 105;
    // size_t N = 1000;
    HostData<float> v(N);
    DeviceData<float> d_v(N);

    d_v.random_init();
    v = d_v;

    float v_mean;
    TIME_USED(1, [&](){
        v_mean = mean(v);
    }).print("cpu");

    std::cout << std::setprecision(10);
    std::cout << "mean: " << v_mean << std::endl;

    float sum;
    sum = sum_gpu(d_v.data(), d_v.size());
    TIME_USED(100, [&](){
        sum = sum_gpu(d_v.data(), d_v.size());
    }).print("gpu");
    std::cout << "[gpu] mean: " << sum / N << std::endl;

    sum = sum_cublas(d_v.data(), d_v.size());
    TIME_USED(100, [&]() {
        sum = sum_cublas(d_v.data(), d_v.size());
    }).print("cublas");
    std::cout << "[cublas] mean: " << sum / N << std::endl;

    return 0;
}

