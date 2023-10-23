#ifndef DOLOMITE_BASE_UTILS_HPP
#define DOLOMITE_BASE_UTILS_HPP

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

template<typename T>
pybind11::object mask_numpy_array(const pybind11::array_t<T>& values, const std::vector<size_t>& missing) {
    size_t n = values.size();
    pybind11::array_t<bool> mask(n);
    for (size_t i = 0; i < n; ++i) {
        mask.mutable_at(i) = 0; 
    }
    for (auto m : missing) {
        mask.mutable_at(m) = 1;
    }

    using namespace pybind11::literals;
    pybind11::module np = pybind11::module::import("numpy");
    pybind11::module ma = np.attr("ma");
    return ma.attr("array")(values, "mask"_a=mask);
}

#endif
