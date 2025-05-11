#pragma once

#include <chrono>
#include <cmath>
#include <string>
#include <iostream>

struct TimeRes {
    double mean;
    double min;
    double max;

    void print(std::string s) {
        std::cout << "[" << s << " time used]"
                  << " mean: " << mean << ", min: " << min << ", max: " << max << std::endl;
    }
};

template<typename F>
TimeRes TIME_USED(size_t n, F&& func) {
    double mean = 0;
    double min = 1e10;
    double max = -1e10;
    n = (n == 0 ? 1 : n);
    for (size_t i = 0; i < n; ++i) {

        auto ts = std::chrono::high_resolution_clock::now();
        func();
        auto te = std::chrono::high_resolution_clock::now();

        auto used_duration = te - ts;
        double t_used = std::chrono::duration_cast<std::chrono::microseconds>(used_duration).count() / 1000.0;
        mean += t_used;
        max = std::fmax(max, t_used);
        min = std::fmin(min, t_used);
    }
    mean /= n;

    return {mean, min, max};
}

