#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "comservatory/comservatory.hpp"
#include <cstdint>
#include "utils.h"

using namespace pybind11::literals;

template<typename T, typename Base>
struct PythonNumpyField : public Base {
    PythonNumpyField(size_t current, size_t max) : position(current), storage(max) {
        for (size_t i = 0; i < position; ++i) {
            storage.mutable_at(i) = 0;
            masked.push_back(i);
        }
    }

    size_t size() const { 
        return position; 
    }

    void push_back(T x) {
        if (position == storage.size()) {
            throw std::runtime_error("more rows present in the CSV than expected");
        }
        storage.mutable_at(position) = x;
        ++position;
    }

    void add_missing() {
        if (position == storage.size()) {
            throw std::runtime_error("more rows present in the CSV than expected");
        }
        storage.mutable_at(position) = 0;
        masked.push_back(position);
        ++position;
    }

    pybind11::object format_output() const {
        if (masked.empty()) {
            return storage;
        } else {
            return mask_numpy_array(storage, masked);
        }
    }

    size_t position;
    pybind11::array_t<T> storage;
    std::vector<size_t> masked;
};

typedef PythonNumpyField<double, comservatory::NumberField> PythonNumberField;
typedef PythonNumpyField<char, comservatory::BooleanField> PythonBooleanField;

struct PythonStringField : public comservatory::StringField {
    PythonStringField(size_t current) {
        for (size_t i = 0; i < current; ++i) {
            storage.append(pybind11::none());
        }
    }

    size_t size() const { 
        return storage.size();
    }

    void push_back(std::string x) {
        storage.append(std::move(x));
    }

    void add_missing() {
        storage.append(pybind11::none());
    }

    pybind11::object format_output() const {
        return storage;
    }

    pybind11::list storage;
};

struct PythonFieldCreator : public comservatory::FieldCreator {
    PythonFieldCreator(size_t n) : num_records(n) {}

    comservatory::Field* create(comservatory::Type observed, size_t n, bool) const {
        comservatory::Field* ptr;

        switch (observed) {
            case comservatory::STRING:
                ptr = new PythonStringField(n);
                break;
            case comservatory::NUMBER:
                ptr = new PythonNumberField(n, num_records);
                break;
            case comservatory::BOOLEAN:
                ptr = new PythonBooleanField(n, num_records);
                break;
            default:
                throw std::runtime_error("unsupported type during field creation");
        }

        return ptr;
    }

    size_t num_records;
};

pybind11::object load_csv(std::string path, size_t nrow) {
    comservatory::ReadCsv reader;
    PythonFieldCreator creator(nrow);
    reader.creator = &creator;

    auto contents = reader.read(path.c_str()); // throws error for invalid formats.
    if (contents.num_records() != nrow) {
        throw std::runtime_error("difference between the observed and expected number of CSV rows (" + std::to_string(contents.num_records()) + " to " + std::to_string(nrow) + ")");
    }

    pybind11::list names;
    for (const auto& n : contents.names) {
        names.append(n);
    }

    pybind11::list fields;
    for (size_t o = 0; o < contents.num_fields(); ++o) {
        switch (contents.fields[o]->type()) {
            case comservatory::STRING:
                fields.append(static_cast<PythonStringField*>(contents.fields[o].get())->format_output());
                break;
            case comservatory::NUMBER:
                fields.append(static_cast<PythonNumberField*>(contents.fields[o].get())->format_output());
                break;
            case comservatory::BOOLEAN:
                fields.append(static_cast<PythonBooleanField*>(contents.fields[o].get())->format_output());
                break;
            case comservatory::UNKNOWN:
                {
                    pybind11::array_t<bool> values(nrow);
                    pybind11::array_t<bool> mask(nrow);
                    for (size_t i = 0; i < nrow; ++i) {
                        values.mutable_at(i) = 0;
                        mask.mutable_at(i) = 1; 
                    }
                    pybind11::module np = pybind11::module::import("numpy");
                    pybind11::module ma = np.attr("ma");
                    fields.append(ma.attr("array")(values, "mask"_a=mask));
                }
                break;
            default:
                throw std::runtime_error("unrecognized type during list assignment");
        }
    }

    return pybind11::dict("names"_a = names, "fields"_a = fields);
}

void validate_csv(std::string path) {
    comservatory::ReadCsv reader;
    reader.validate_only = true;
    reader.read(path);
}
