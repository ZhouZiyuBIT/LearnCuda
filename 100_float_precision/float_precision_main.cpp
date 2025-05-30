#include <iomanip>
#include <iostream>

#include "host_data.h"


int main() {
    HostData<float> float_data(100000);
    const size_t test_num = 1000;
    double sum_err = 0;
    for (int t = 0; t < test_num; ++t) {
        float_data.random_init(0.f, 1.f);
        double sum = 0;
        float sum2 = 0;
        for (int i = 0; i < float_data.size(); ++i) {
            sum += float_data.data()[i];
            sum2 += float_data.data()[i];
        }

        sum_err += std::abs(sum - sum2);
        std::cout << std::setprecision(10);
        // std::cout << "sum: " << sum << ", sum2: " << sum2 << std::endl;
        // std::cout << "sum diff: " << ((float)sum - sum2) << std::endl;
    }
    std::cout << "mean error: " << sum_err / test_num << std::endl;

    return 0;
}

