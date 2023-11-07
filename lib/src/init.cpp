#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include <cstdint>

// Declarations:
pybind11::object choose_missing_integer_placeholder(pybind11::array_t<int32_t>, pybind11::array_t<uint8_t>);
pybind11::object choose_missing_float_placeholder(pybind11::array_t<double>, pybind11::array_t<uint8_t>);

pybind11::object load_csv(std::string, size_t, bool, bool);
void validate_csv(std::string, bool, bool);

pybind11::object load_list_json(std::string, pybind11::list);
void validate_list_json(std::string, size_t);
pybind11::object load_list_hdf5(std::string, std::string, pybind11::list);
void validate_list_hdf5(std::string, std::string, size_t);

void check_csv_df(std::string, int, bool, pybind11::list, pybind11::list, pybind11::list, pybind11::array_t<bool>, pybind11::list, int, bool, bool);
void check_hdf5_df(std::string, std::string, int, bool, pybind11::list, pybind11::list, pybind11::list, pybind11::array_t<bool>, pybind11::list, int, int);

// Binding:
PYBIND11_MODULE(lib_dolomite_base, m) {
    m.def("choose_missing_integer_placeholder", &choose_missing_integer_placeholder);
    m.def("choose_missing_float_placeholder", &choose_missing_float_placeholder);

    m.def("load_csv", &load_csv);
    m.def("validate_csv", &validate_csv);

    m.def("load_list_json", &load_list_json);
    m.def("validate_list_json", &validate_list_json);
    m.def("load_list_hdf5", &load_list_hdf5);
    m.def("validate_list_hdf5", &validate_list_hdf5);

    m.def("check_csv_df", &check_csv_df);
    m.def("check_hdf5_df", &check_hdf5_df);
}
