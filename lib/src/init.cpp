#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

// Declarations:
pybind11::object create_r_missing_double();
pybind11::object create_nan_mask_for_double(pybind11::array_t<double>, pybind11::array_t<double>);
pybind11::object create_nan_mask_for_float(pybind11::array_t<float>, pybind11::array_t<float>);

pybind11::object load_csv(std::string, size_t);
pybind11::object validate_csv(std::string);

// Binding:
PYBIND11_MODULE(lib_dolomite_base, m) {
    m.def("create_r_missing_double", &create_r_missing_double);
    m.def("create_nan_mask_for_double", &create_nan_mask_for_double);
    m.def("create_nan_mask_for_float", &create_nan_mask_for_float);

    m.def("load_csv", &load_csv);
    m.def("validate_csv", &validate_csv);
}
