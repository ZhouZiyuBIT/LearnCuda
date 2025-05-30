#pragma once

#include <cuda_runtime.h>

template <typename T>
class DeviceData;

template <typename T>
class HostData {
public:
    // 定义了有参构造函数，默认构造函数就deleted掉了
    explicit HostData(size_t N);
    HostData(HostData<T>&& other);
    const HostData<T>& operator= (HostData<T>&& other);
    HostData(const HostData<T>& other) = delete;
    HostData<T> operator= (const HostData<T>& other) = delete;

    void random_init(T a, T b);

    // device to host
    const HostData<T>& operator= (const DeviceData<T>& d_other);

    // comparison func
    bool operator== (const HostData<T>& cmp);

    // getter
    size_t size() const { return _N; }
    const T* data() const { return _data; }
    T* data() { return _data; }

private:
    T* _data = nullptr;
    size_t _N = 0;
};

