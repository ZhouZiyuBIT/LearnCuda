
#include "device_data.h"
#include "host_data.h"

#include <iostream>

template <typename T>
DeviceData<T>::DeviceData(size_t N)
        : _N(N) {
    if (cudaMalloc(&_data, sizeof(T) * _N) != cudaSuccess) {
        std::cerr << "DeviceData constructed error" << std::endl;
        std::exit(1);
    }
}

template <typename T>
DeviceData<T>::DeviceData(DeviceData<T>&& other) {
    _N = other._N;
    _data = other._data;
    other._data = nullptr;
    other._N = 0;
}

template <typename T>
const DeviceData<T>& DeviceData<T>::operator= (DeviceData<T>&& other) {
    if (this == &other) {
        return *this;
    }

    size_t N_tmp = _N;
    T* data_tmp = _data;

    _N = other._N;
    _data = other._data;

    other._N = N_tmp;
    other._data = data_tmp;

    return *this;
}

template<typename T>
const DeviceData<T>& DeviceData<T>::operator=(const HostData<T>& d_other) {
    if (this->_N != d_other.size()) {
        std::cerr << "host to device error: size is not equal" << std::endl;
        std::exit(1);
    }
    if (_data == nullptr || d_other.data() == nullptr) {
        std::cerr << "host to device error: _data is nullptr" << std::endl;
        std::exit(1);
    }

    if (cudaMemcpy(_data, d_other.data(), sizeof(T) * _N, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::cerr << "host to device error: copy error" << std::endl;
        std::exit(1);
    }

    return *this;
}

void random_uniform(const int min, const int max, int* x, const size_t N);
void random_uniform(const float min, const float max, float* x, const size_t N);

template<typename T>
void DeviceData<T>::random_init() {
    // TODO: 特化为只针对int类型
    random_uniform(0.0, 10.0, _data, _N);
}

template class DeviceData<int>;
template class DeviceData<float>;

