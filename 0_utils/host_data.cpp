
#include "host_data.h"
#include "device_data.h"

#include <iostream>
#include <random>
#include <cmath>
#include <type_traits>

template<typename T>
HostData<T>::HostData(size_t N)
        : _N(N) {
    _data = new T[_N];
    if (_data == nullptr) {
        std::cerr << "HostData constructed error" << std::endl;
        std::exit(1);
    }
}
template<typename T>
HostData<T>::HostData(HostData<T>&& other) {
    _N = other._N;
    _data = other._data;
    other._data = nullptr;
    other._N = 0;
}

template<typename T>
const HostData<T>& HostData<T>::operator= (HostData<T>&& other) {
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
bool HostData<T>::operator== (const HostData<T>& cmp) {
    if (_N != cmp._N) {
        return false;
    }

    for (size_t i = 0; i < _N; ++i) {
        double diff = static_cast<double>(_data[i] - cmp._data[i]);
        if (std::fabs(diff) > 0.0001) {
            return false;
        }
    }

    return true;
}

template<typename T>
const HostData<T>& HostData<T>::operator=(const DeviceData<T>& d_other) {
    if (this->_N != d_other.size()) {
        std::cerr << "device to host error: size is not equal" << std::endl;
        std::exit(1);
    }
    if (_data == nullptr || d_other.data() == nullptr) {
        std::cerr << "device to host error: _data is nullptr" << std::endl;
        std::exit(1);
    }

    if (cudaMemcpy(_data, d_other.data(), sizeof(T) * _N, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::cerr << "device to host error: copy error" << std::endl;
        std::exit(1);
    }

    return *this;
}

static std::mt19937 random_data_gen(13483);

template<typename T>
void HostData<T>::random_init() {
    using DistributionType = typename std::conditional<std::is_integral<T>::value,
            std::uniform_int_distribution<T>, std::uniform_real_distribution<T>>::type;
    DistributionType dist(0, 100);

    for (size_t i = 0; i < _N; ++i) {
        _data[i] = dist(random_data_gen);
    }
}

template class HostData<int>;

