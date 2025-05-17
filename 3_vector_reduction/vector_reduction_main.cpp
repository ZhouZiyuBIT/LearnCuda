
#include <iomanip>
#include <iostream>
#include <cmath>

#include "device_data.h"
#include "host_data.h"
#include "time_statistics.h"

double mean(const HostData<int>& h_d) {
    int64_t sum = 0;
    size_t N = h_d.size();
    for (size_t i = 0; i < N; ++i) {
        sum += h_d.data()[i];
    }
    double mean = static_cast<double>(sum) / N;
    return mean;
}

double std_dev(const HostData<int>& h_d, double mean) {
    double sum = 0;
    size_t N = h_d.size();
    for (size_t i = 0; i < N; ++i) {
        double e = h_d.data()[i] - mean;
        sum += std::sqrt(e * e);
    }
    double dev = sum / N;
    return dev;
}

int64_t int_sum_gpu(const int* v, const size_t N);

int main() {
    size_t N = (1 << 20) * 100;
    HostData<int> v(N);
    DeviceData<int> d_v(N);

    d_v.random_init();
    v = d_v;

    double v_mean;
    TIME_USED(10, [&](){
        v_mean = mean(v);
    }).print("[cpu]");
    double v_dev = std_dev(v, v_mean);

    std::cout << std::setprecision(10);
    std::cout << "mean: " << v_mean << ", dev: " << v_dev << std::endl;

    int64_t sum;
    TIME_USED(100, [&](){
        sum = int_sum_gpu(d_v.data(), d_v.size());
    }).print("[gpu]");
    std::cout << "[gpu] mean: " << static_cast<double>(sum) / N << std::endl;

    return 0;
}

