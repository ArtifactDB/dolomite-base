#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

// Declarations:
pybind11::object create_r_missing_double();
pybind11::object create_nan_mask(uintptr_t, size_t, size_t, uintptr_t);

pybind11::object load_csv(std::string, size_t);
void validate_csv(std::string);

// Binding:
PYBIND11_MODULE(lib_dolomite_base, m) {
    m.def("create_r_missing_double", &create_r_missing_double);
    m.def("create_nan_mask", &create_nan_mask);

    m.def("load_csv", &load_csv);
    m.def("validate_csv", &validate_csv);
}
