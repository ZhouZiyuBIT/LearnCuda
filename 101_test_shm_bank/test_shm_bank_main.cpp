
#include "host_data.h"
#include "device_data.h"
#include "time_statistics.h"

// 我的5070Ti一共70个SM，在（32，16）block大小下，可以同时运行210个block。
// 所以设置，data n = 32 * 16 * 210
#define DATA_SIZE (32 * 16 * 210)

void data_load_to(const float* data, size_t n, float* output);

int main() {
    HostData<float> h_vec(DATA_SIZE);
    DeviceData<float> d_vec(DATA_SIZE);
    DeviceData<float> d_out(DATA_SIZE);
    h_vec.random_init(0.f, 1.f);

    TIME_USED(1, [&]() {
        d_vec = h_vec;
    }).print("host to device");

    data_load_to(d_vec.data(), d_vec.size(), d_out.data());
    TIME_USED(10, [&](){
        data_load_to(d_vec.data(), d_vec.size(), d_out.data());
    }).print("data_load");

    return 0;
}

