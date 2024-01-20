#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include <cstdint>

// Declarations:
pybind11::object choose_missing_integer_placeholder(pybind11::array_t<int32_t>, pybind11::array_t<uint8_t>);
pybind11::object choose_missing_float_placeholder(pybind11::array_t<double>, pybind11::array_t<uint8_t>);

pybind11::object load_list_json(std::string, pybind11::list);
pybind11::object load_list_hdf5(std::string, std::string, pybind11::list);

void validate(std::string);

// Binding:
PYBIND11_MODULE(lib_dolomite_base, m) {
    m.def("choose_missing_integer_placeholder", &choose_missing_integer_placeholder);
    m.def("choose_missing_float_placeholder", &choose_missing_float_placeholder);

    m.def("load_list_json", &load_list_json);
    m.def("load_list_hdf5", &load_list_hdf5);

    m.def("validate", &validate);
}
