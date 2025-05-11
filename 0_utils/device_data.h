#pragma once

#include <cuda_runtime.h>

template<typename T>
class HostData;

template <typename T>
class DeviceData {
public:
    // 定义了有参构造函数，默认构造函数就deleted掉了
    explicit DeviceData(size_t N);
    DeviceData(DeviceData<T>&& other);
    const DeviceData<T>& operator= (DeviceData<T>&& other);
    DeviceData(const DeviceData<T>& other) = delete;
    DeviceData<T> operator= (const DeviceData<T>& other) = delete;

    const DeviceData<T>& operator= (const HostData<T>& h_other);

    void random_init();

    size_t size() const { return _N; }
    const T* data() const { return _data; }
    T* data() { return _data; }

private:
    T* _data = nullptr;
    size_t _N = 0;
};


