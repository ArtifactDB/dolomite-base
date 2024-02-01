#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include <cstdint>

// Declarations:
pybind11::object load_list_json(std::string, pybind11::list);
pybind11::object load_list_hdf5(std::string, std::string, pybind11::list);
void validate(std::string, pybind11::handle, pybind11::dict);

// Binding:
PYBIND11_MODULE(lib_dolomite_base, m) {
    m.def("load_list_json", &load_list_json);
    m.def("load_list_hdf5", &load_list_hdf5);
    m.def("validate", &validate);
}
