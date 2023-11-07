#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "ritsuko/ritsuko.hpp"
#include <cstring>

pybind11::object choose_missing_integer_placeholder(pybind11::array_t<int32_t> x, pybind11::array_t<uint8_t> mask) {
    auto req = x.request();
    auto ptr = static_cast<int32_t*>(req.ptr);
    auto mreq = mask.request();
    auto mptr = static_cast<const uint8_t*>(mreq.ptr);
    size_t n = x.size();

    auto out = ritsuko::choose_missing_integer_placeholder(ptr, ptr + n, mptr);
    if (out.first) {
        for (size_t i = 0; i < n; ++i) {
            if (mptr[i]) {
                ptr[i] = out.second;
            }
        }
    } 

    pybind11::module np = pybind11::module::import("numpy");
    return pybind11::make_tuple(out.first, np.attr("int32")(out.second));
}

pybind11::object choose_missing_float_placeholder(pybind11::array_t<double> x, pybind11::array_t<uint8_t> mask) {
    auto req = x.request();
    auto ptr = static_cast<double*>(req.ptr);
    auto mreq = mask.request();
    auto mptr = static_cast<const uint8_t*>(mreq.ptr);
    size_t n = x.size();

    auto out = ritsuko::choose_missing_float_placeholder(ptr, ptr + n, mptr, false);
    if (out.first) {
        for (size_t i = 0; i < n; ++i) {
            if (mptr[i]) {
                ptr[i] = out.second;
            }
        }
    } 

    pybind11::module np = pybind11::module::import("numpy");
    return pybind11::make_tuple(out.first, np.attr("float64")(out.second));
}
