#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "uzuki2/uzuki2.hpp"

#include "utils.h"

#include <cstdint>
#include <iostream>

/** Defining the various elements. **/

struct PythonBase {
    virtual ~PythonBase() = default;
    virtual pybind11::object extract() const = 0;
};

template<typename T, class Base>
struct PythonNumpyVector : public Base, public PythonBase {
    PythonNumpyVector(size_t l, bool n, bool s) : storage(l), names(n ? l : 0), is_scalar(s) {}

    size_t size() const { 
        return storage.size();
    }

    void set(size_t i, T val) {
        storage[i] = val;
    }

    void set_missing(size_t i) {
        storage[i] = pybind11::none();
    }

    void set_name(size_t i, std::string n) {
        names[i] = std::move(n);
    }

    pybind11::object extract() const {
        if (names.empty()) {
            if (is_scalar) {
                return storage[0];
            } else {
                pybind11::module bu = pybind11::module::import("biocutils");
                if constexpr(std::is_same<T, int32_t>::value) {
                    return bu.attr("IntegerList")(storage);
                } else if constexpr(std::is_same<T, bool>::value) {
                    return bu.attr("BooleanList")(storage);
                } else {
                    return bu.attr("FloatList")(storage);
                }
            }
        } else {
            pybind11::module bu = pybind11::module::import("biocutils");
            using namespace pybind11::literals;
            if constexpr(std::is_same<T, int32_t>::value) {
                return bu.attr("IntegerList")(storage, "names"_a = names);
            } else if constexpr(std::is_same<T, bool>::value) {
                return bu.attr("BooleanList")(storage, "names"_a = names);
            } else {
                return bu.attr("FloatList")(storage, "names"_a = names);
            }
        }
    }

    pybind11::list storage;
    pybind11::list names;
    bool is_scalar;
};

typedef PythonNumpyVector<double, uzuki2::NumberVector> PythonNumberVector;
typedef PythonNumpyVector<int32_t, uzuki2::IntegerVector> PythonIntegerVector;
typedef PythonNumpyVector<bool, uzuki2::BooleanVector> PythonBooleanVector;

struct PythonStringVector : public uzuki2::StringVector, public PythonBase {
    PythonStringVector(size_t l, bool n, bool s, uzuki2::StringVector::Format) : storage(l), names(n ? l : 0), is_scalar(s) {}

    size_t size() const { 
        return storage.size();
    }

    void set(size_t i, std::string val) {
        storage[i] = std::move(val);
    }

    void set_missing(size_t i) {
        storage[i] = pybind11::none();
    }

    void set_name(size_t i, std::string name) {
        names[i] = std::move(name);
    }

    pybind11::object extract() const {
        if (names.empty()) {
            if (is_scalar) {
                return storage[0];
            } else {
                pybind11::module bu = pybind11::module::import("biocutils");
                return bu.attr("StringList")(storage);
            }
        } else {
            pybind11::module bu = pybind11::module::import("biocutils");
            using namespace pybind11::literals;
            return bu.attr("StringList")(storage, "names"_a = names);
        }
    }

    pybind11::list storage;
    pybind11::list names;
    bool is_scalar;
};

struct PythonFactor : public uzuki2::Factor, public PythonBase {
    PythonFactor(size_t l, bool n, bool s, size_t ll, bool o) : storage(l), names(n ? l : 0), is_scalar(s), levels(ll), ordered(o) {}

    size_t size() const { 
        return storage.size(); 
    }

    void set(size_t i, size_t l) {
        storage.mutable_at(i) = l;
    }

    void set_missing(size_t i) {
        storage.mutable_at(i) = -1;
    }

    void set_name(size_t i, std::string name) {
        names[i] = std::move(name);
    }

    void set_level(size_t i, std::string l) {
        levels[i] = std::move(l);
    }

    pybind11::object extract() const {
        pybind11::module bu = pybind11::module::import("biocutils");
        using namespace pybind11::literals;
        if (names.size() == 0) {
            return bu.attr("Factor")(storage, levels, "ordered"_a = ordered);
        } else {
            return bu.attr("Factor")(storage, levels, "ordered"_a = ordered, "names"_a = names);
        }
    }

    pybind11::array_t<int32_t> storage;
    pybind11::list names;
    bool is_scalar;
    pybind11::list levels;
    bool ordered;
};

struct PythonNothing : public uzuki2::Nothing, public PythonBase {
    pybind11::object extract() const {
        return pybind11::none();
    }
};

struct PythonExternal : public uzuki2::External, public PythonBase {
    PythonExternal(void *p) : ptr(p) {}

    pybind11::object extract() const {
        return *reinterpret_cast<pybind11::object*>(ptr);
    }

    void* ptr;
};

struct PythonList : public uzuki2::List, public PythonBase {
    PythonList(size_t l, bool n) : values(l), has_names(n), names(n ? l : 0) {}

    size_t size() const { 
        return values.size(); 
    }

    void set(size_t i, std::shared_ptr<uzuki2::Base> ptr) {
        values[i] = dynamic_cast<PythonBase*>(ptr.get())->extract();
    }

    void set_name(size_t i, std::string name) {
        names[i] = std::move(name);
    }

    pybind11::object extract() const {
        pybind11::module bu = pybind11::module::import("biocutils");
        if (!has_names) {
            return bu.attr("NamedList")(values);
        } else {
            using namespace pybind11::literals;
            return bu.attr("NamedList")(values, "names"_a = names);
        }
    }

    pybind11::list values;
    bool has_names = false;
    pybind11::list names;
};

/** Provisioner. **/

struct PythonProvisioner {
    static uzuki2::Nothing* new_Nothing() { return (new PythonNothing); }

    static uzuki2::External* new_External(void* p) { return (new PythonExternal(p)); }

    template<class ... Args_>
    static uzuki2::List* new_List(Args_&& ... args) { return (new PythonList(std::forward<Args_>(args)...)); }

    template<class ... Args_>
    static uzuki2::IntegerVector* new_Integer(Args_&& ... args) { return (new PythonIntegerVector(std::forward<Args_>(args)...)); }

    template<class ... Args_>
    static uzuki2::NumberVector* new_Number(Args_&& ... args) { return (new PythonNumberVector(std::forward<Args_>(args)...)); }

    template<class ... Args_>
    static uzuki2::StringVector* new_String(Args_&& ... args) { return (new PythonStringVector(std::forward<Args_>(args)...)); }

    template<class ... Args_>
    static uzuki2::BooleanVector* new_Boolean(Args_&& ... args) { return (new PythonBooleanVector(std::forward<Args_>(args)...)); }

    template<class ... Args_>
    static uzuki2::Factor* new_Factor(Args_&& ... args) { return (new PythonFactor(std::forward<Args_>(args)...)); }
};

struct PythonExternals {
    PythonExternals(const pybind11::list& current) {
        stored.reserve(current.size());
        for (size_t i = 0, end = current.size(); i < end; ++i) {
            stored.emplace_back(current[i]);
        }
    }

    void* get(size_t i) {
        return reinterpret_cast<void*>(&(stored[i]));
    }

    size_t size() const {
        return stored.size();
    }

    std::vector<pybind11::object> stored;
};

/** General methods. **/

pybind11::object load_list_json(std::string path, pybind11::list children) {
    auto parsed = uzuki2::json::parse_file<PythonProvisioner>(path, PythonExternals(children));
    return dynamic_cast<PythonBase*>(parsed.get())->extract();
}

pybind11::object load_list_hdf5(std::string path, std::string name, pybind11::list children) {
    auto parsed = uzuki2::hdf5::parse<PythonProvisioner>(path, name, PythonExternals(children));
    return dynamic_cast<PythonBase*>(parsed.get())->extract();
}
