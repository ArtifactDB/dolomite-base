#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "uzuki2/uzuki2.hpp"
#include <cstring>

pybind11::object create_r_missing_double() {
    pybind11::module np = pybind11::module::import("numpy");
    return np.attr("float64")(uzuki2::hdf5::legacy_missing_double());
}

template<typename T>
pybind11::object create_nan_mask(const pybind11::array_t<T>& values, const pybind11::array_t<T>& placeholder) {
    auto vptr = reinterpret_cast<const unsigned char*>(values.data());
    auto pptr = reinterpret_cast<const unsigned char*>(placeholder.data());
    constexpr size_t size = sizeof(T);
    size_t number = values.size();

    pybind11::array_t<bool> mask(number);
    for (size_t i = 0; i < number; ++i, vptr += size) {
        mask.mutable_at(i) = (std::memcmp(vptr, pptr, size) == 0);
    }
    return mask;
}

pybind11::object create_nan_mask_for_double(pybind11::array_t<double> values, pybind11::array_t<double> placeholder) {
    return create_nan_mask<double>(values, placeholder);
}

pybind11::object create_nan_mask_for_float(pybind11::array_t<float> values, pybind11::array_t<float> placeholder) {
    return create_nan_mask<float>(values, placeholder);
}
