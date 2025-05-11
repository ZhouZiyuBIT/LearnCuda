
#include "host_data.h"
#include "device_data.h"
#include "time_statistics.h"

#include <cmath>
#include <iostream>

#include <cuda_runtime.h>

void vector_add_gpu(int* a, int* b, int* c, size_t N);

int main() {
    const size_t N = (1 << 20) * 100;
    // const size_t N = 102;

    HostData<int> a(N);
    DeviceData<int> d_a(N);
    HostData<int> b(N);
    DeviceData<int> d_b(N);
    HostData<int> c(N);
    DeviceData<int> d_c(N);
    HostData<int> res(N);

    // a.random_init();
    // b.random_init();
    d_a.random_init();
    d_b.random_init();
    a = d_a;
    b = d_b;

    TIME_USED(1, [&]() {
        for (size_t i = 0; i < N; ++i) {
            res.data()[i] = a.data()[i] + b.data()[i];
        }
    }).print("vector_add_cpu");

    // d_a = a;
    // d_b = b;

    vector_add_gpu(d_a.data(), d_b.data(), d_c.data(), N);
    c = d_c;
    if (c == res) {
        std::cout << "res check pass" << std::endl;
    } else {
        std::cout << "not check pass" << std::endl;
    }
    // for (size_t i = 0; i < 10; ++i) {
    //     std::cout << c.data()[i] << std::endl;
    // }

    TIME_USED(10, [&]() {
        d_a.random_init();
        d_b.random_init();
        vector_add_gpu(d_a.data(), d_b.data(), d_c.data(), N);
        cudaDeviceSynchronize();
    }).print("vector_add_gpu");

    return 0;
}

