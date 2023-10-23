#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

// Declarations:
pybind11::object create_r_missing_double();
pybind11::object create_nan_mask(uintptr_t, size_t, size_t, uintptr_t);

pybind11::object load_csv(std::string, size_t);
void validate_csv(std::string);

pybind11::object load_list_json(std::string, pybind11::list);
void validate_list_json(std::string, size_t);
pybind11::object load_list_hdf5(std::string, std::string, pybind11::list);
void validate_list_hdf5(std::string, std::string, size_t);

// Binding:
PYBIND11_MODULE(lib_dolomite_base, m) {
    m.def("create_r_missing_double", &create_r_missing_double);
    m.def("create_nan_mask", &create_nan_mask);

    m.def("load_csv", &load_csv);
    m.def("validate_csv", &validate_csv);

    m.def("load_list_json", &load_list_json);
    m.def("validate_list_json", &validate_list_json);
    m.def("load_list_hdf5", &load_list_hdf5);
    m.def("validate_list_hdf5", &validate_list_hdf5);
}
