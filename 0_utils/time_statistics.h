#pragma once

#include <chrono>
#include <cmath>
#include <ctime>
#include <string>
#include <iostream>
#include <cuda_runtime.h>

void warmup_gpu(); // 声明外部专用的GPU热身函数

struct TimeRes {
    double mean;
    double min;
    double max;

    void print(std::string s) {
        std::cout << "[" << s << " time used]"
                  << " mean: " << mean << " ms, min: " << min << " ms, max: " << max << " ms" << std::endl;
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

// 基于 cudaEvent 的精准计时器，消除 CPU 到 GPU 的额外同步开销
template<typename F>
TimeRes CUDA_TIME_USED(size_t n, F&& func, cudaStream_t stream = nullptr) {
    double mean = 0;
    n = (n == 0 ? 1 : n);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 1. 调用极高强度的内置空载内核，强行解除系统降频 (P8 -> P0 state)
    warmup_gpu();

    // 2. 将目标 kernel 试运行几次，确保该核函数的指令缓存 (I-Cache) 和内存页已完全加载
    for (int i = 0; i < 5; ++i) {
        func();
    }
    cudaStreamSynchronize(stream);

    // 一次性记录开始事件
    cudaEventRecord(start, stream);
    
    // 异步连续提交 N 次 Kernel，彻底掩盖 CPU 端发射开销和驱动唤醒延迟
    for (size_t i = 0; i < n; ++i) {
        func();
    }
    
    // 一次性记录结束事件，并在这时候才强制要求 CPU 同步阻塞
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // 因为这里是在整体耗时上均摊出来的，极度稳定，已经没有单次迭代的极大/极小差异了
    // 故用均值来替代 min 和 max
    mean = milliseconds / n;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return {mean, mean, mean};
}

