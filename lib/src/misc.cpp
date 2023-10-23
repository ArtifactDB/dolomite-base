#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "uzuki2/uzuki2.hpp"
#include "ritsuko/ritsuko.hpp"
#include <cstring>

pybind11::object create_r_missing_double() {
    pybind11::module np = pybind11::module::import("numpy");
    return np.attr("float64")(ritsuko::r_missing_value());
}

pybind11::object create_nan_mask(uintptr_t values, size_t number, size_t size, uintptr_t placeholder) {
    auto vptr = reinterpret_cast<const unsigned char*>(values);
    auto pptr = reinterpret_cast<const unsigned char*>(placeholder);
    pybind11::array_t<bool> mask(number);
    for (size_t i = 0; i < number; ++i, vptr += size) {
        mask.mutable_at(i) = (std::memcmp(vptr, pptr, size) == 0);
    }
    return mask;
}
