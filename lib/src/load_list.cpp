#include "uzuki2/uzuki2.hpp"
#include <cstdint>

/** Defining the various elements. **/

template<typename T>
struct DefaultVectorBase { 
    DefaultVectorBase(size_t l, bool n, bool s) : values(l), has_names(n), names(n ? l : 0), scalar(s) {}

    size_t size() const { 
        return values.size(); 
    }

    void set(size_t i, T val) {
        values[i] = std::move(val);
    }

    void set_missing(size_t i) {
        missing.push_back(i);
    }

    void set_name(size_t i, std::string name) {
        names[i] = std::move(name);
    }

    std::vector<T> values;
    std::vector<int32_t> missing;
    bool has_names;
    std::vector<std::string> names;
    bool scalar;
};

struct DefaultIntegerVector : public uzuki2::IntegerVector {
    DefaultIntegerVector(size_t l, bool n, bool s) : base(l, n, s) {}

    size_t size() const { 
        return base.size();
    }

    void set(size_t i, int32_t val) {
        base.set(i, val);
    }

    void set_missing(size_t i) {
        base.set_missing(i);
    }

    void set_name(size_t i, std::string name) {
        base.set_name(i, std::move(name));
    }

    DefaultVectorBase<int32_t> base;
};

struct DefaultNumberVector : public uzuki2::NumberVector {
    DefaultNumberVector(size_t l, bool n, bool s) : base(l, n, s) {}

    size_t size() const { 
        return base.size();
    }

    void set(size_t i, double val) {
        base.set(i, val);
    }

    void set_missing(size_t i) {
        base.set_missing(i);
    }

    void set_name(size_t i, std::string name) {
        base.set_name(i, std::move(name));
    }

    DefaultVectorBase<double> base;
};

struct DefaultBooleanVector : public uzuki2::BooleanVector {
    DefaultBooleanVector(size_t l, bool n, bool s) : base(l, n, s) {}

    size_t size() const { 
        return base.size();
    }

    void set(size_t i, bool val) {
        base.set(i, val);
    }

    void set_missing(size_t i) {
        base.set_missing(i);
    }

    void set_name(size_t i, std::string name) {
        base.set_name(i, std::move(name));
    }

    DefaultVectorBase<uint8_t> base;
};

struct DefaultStringVector : public uzuki2::StringVector {
    DefaultStringVector(size_t l, bool n, bool s, uzuki2::StringVector::Format f) : base(l, n, s), format(f) {}

    size_t size() const { 
        return base.size();
    }

    void set(size_t i, std::string val) {
        base.set(i, std::move(val));
    }

    void set_missing(size_t i) {
        base.set_missing(i);
    }

    void set_name(size_t i, std::string name) {
        base.set_name(i, std::move(name));
    }

    DefaultVectorBase<std::string> base;
    uzuki2::StringVector::Format format;
};

struct DefaultFactor : public uzuki2::Factor {
    DefaultFactor(size_t l, bool n, bool s, size_t ll, bool o) : vbase(l, n, s), levels(ll), ordered(o) {}

    size_t size() const { 
        return vbase.size(); 
    }

    void set(size_t i, size_t l) {
        vbase.set(i, l);
    }

    void set_missing(size_t i) {
        vbase.set_missing(i);
    }

    void set_name(size_t i, std::string name) {
        vbase.set_name(i, std::move(name));
    }

    void set_level(size_t i, std::string l) {
        levels[i] = std::move(l);
    }

    DefaultVectorBase<size_t> vbase;
    std::vector<std::string> levels;
    bool ordered;
};

struct DefaultNothing : public uzuki2::Nothing {};

struct DefaultExternal : public uzuki2::External {
    DefaultExternal(void *p) : ptr(p) {}
    void* ptr;
};

struct DefaultList : public uzuki2::List {
    DefaultList(size_t l, bool n) : values(l), has_names(n), names(n ? l : 0) {}

    size_t size() const { 
        return values.size(); 
    }

    void set(size_t i, std::shared_ptr<uzuki2::Base> ptr) {
        values[i] = std::move(ptr);
    }

    void set_name(size_t i, std::string name) {
        names[i] = std::move(name);
    }

    std::vector<std::shared_ptr<uzuki2::Base> > values;
    bool has_names = false;
    std::vector<std::string> names;
};

/** Provisioner. **/

struct DefaultProvisioner {
    static uzuki2::Nothing* new_Nothing() { return (new DefaultNothing); }

    static uzuki2::External* new_External(void* p) { return (new DefaultExternal(p)); }

    template<class ... Args_>
    static uzuki2::List* new_List(Args_&& ... args) { return (new DefaultList(std::forward<Args_>(args)...)); }

    template<class ... Args_>
    static uzuki2::IntegerVector* new_Integer(Args_&& ... args) { return (new DefaultIntegerVector(std::forward<Args_>(args)...)); }

    template<class ... Args_>
    static uzuki2::NumberVector* new_Number(Args_&& ... args) { return (new DefaultNumberVector(std::forward<Args_>(args)...)); }

    template<class ... Args_>
    static uzuki2::StringVector* new_String(Args_&& ... args) { return (new DefaultStringVector(std::forward<Args_>(args)...)); }

    template<class ... Args_>
    static uzuki2::BooleanVector* new_Boolean(Args_&& ... args) { return (new DefaultBooleanVector(std::forward<Args_>(args)...)); }

    template<class ... Args_>
    static uzuki2::Factor* new_Factor(Args_&& ... args) { return (new DefaultFactor(std::forward<Args_>(args)...)); }
};

struct DefaultExternals {
    DefaultExternals(size_t n) : number(n) {}

    void* get(size_t i) {
        return reinterpret_cast<void*>(static_cast<uintptr_t>(i + 1));
    }

    size_t size() const {
        return number;
    }

    size_t number;
};

/** General methods. **/

//[[export]]
void* load_list_json(const char* path, int32_t n) {
    auto parsed = uzuki2::json::parse_file<DefaultProvisioner>(path, DefaultExternals(n));
    return new uzuki2::ParsedList(std::move(parsed));
}

//[[export]]
void validate_list_json(const char* path, int32_t n) {
    uzuki2::json::validate_file(path, n);
}

//[[export]]
void* load_list_hdf5(const char* path, const char* name, int32_t n) {
    auto parsed = uzuki2::hdf5::parse<DefaultProvisioner>(path, name, DefaultExternals(n));
    return new uzuki2::ParsedList(std::move(parsed));
}

//[[export]]
void validate_list_hdf5(const char* path, const char* name, int32_t n) {
    uzuki2::hdf5::validate(path, name, n);
}

//[[export]]
void uzuki2_free_list(void* ptr) {
    delete reinterpret_cast<uzuki2::ParsedList*>(ptr);
}

//[[export]]
void* uzuki2_get_parent_node(void * ptr) {
    return reinterpret_cast<uzuki2::ParsedList*>(ptr)->get();
}

//[[export]]
int32_t uzuki2_get_node_type(void* ptr) {
    auto casted = reinterpret_cast<uzuki2::Base*>(ptr);
    auto list_type = casted->type();

    if (list_type == uzuki2::INTEGER) {
        return 0;
    } else if (list_type == uzuki2::NUMBER) {
        return 1;
    } else if (list_type == uzuki2::BOOLEAN) {
        return 2;
    } else if (list_type == uzuki2::STRING) {
        return 3;
    } else if (list_type == uzuki2::LIST) {
        return 4;
    } else if (list_type == uzuki2::NOTHING) {
        return 5;
    } else if (list_type == uzuki2::EXTERNAL) {
        return 6;
    }

    throw std::runtime_error("unsupported uzuki2 type");
    return -1;
}

/* Integer vector handlers */

//[[export]]
int32_t uzuki2_get_integer_vector_length(void* ptr) {
    auto casted = reinterpret_cast<DefaultIntegerVector*>(ptr);
    if (casted->base.scalar) {
        return -1;
    } else {
        return casted->size();
    }
}

//[[export]]
uint8_t uzuki2_get_integer_vector_values(void* ptr, int32_t* contents /** numpy */) {
    auto casted = reinterpret_cast<DefaultIntegerVector*>(ptr);
    const auto& vals = casted->base.values;
    std::copy(vals.begin(), vals.end(), contents);
    return !(casted->base.missing.empty());
}

//[[export]]
void uzuki2_get_integer_vector_mask(void* ptr, uint8_t* mask /** numpy */) {
    auto casted = reinterpret_cast<DefaultIntegerVector*>(ptr);
    for (auto i : casted->base.missing) {
        mask[i] = 1;
    }
}

/* Number vector handlers */

//[[export]]
int32_t uzuki2_get_number_vector_length(void* ptr) {
    auto casted = reinterpret_cast<DefaultNumberVector*>(ptr);
    if (casted->base.scalar) {
        return -1;
    } else {
        return casted->size();
    }
}

//[[export]]
uint8_t uzuki2_get_number_vector_values(void* ptr, double* contents /** numpy */) {
    auto casted = reinterpret_cast<DefaultNumberVector*>(ptr);
    const auto& vals = casted->base.values;
    std::copy(vals.begin(), vals.end(), contents);
    return !(casted->base.missing.empty());
}

//[[export]]
void uzuki2_get_number_vector_mask(void* ptr, uint8_t* mask /** numpy */) {
    auto casted = reinterpret_cast<DefaultNumberVector*>(ptr);
    for (auto i : casted->base.missing) {
        mask[i] = 1;
    }
}

/* Boolean vector handlers */

//[[export]]
int32_t uzuki2_get_boolean_vector_length(void* ptr) {
    auto casted = reinterpret_cast<DefaultBooleanVector*>(ptr);
    if (casted->base.scalar) {
        return -1;
    } else {
        return casted->size();
    }
}

//[[export]]
uint8_t uzuki2_get_boolean_vector_values(void* ptr, uint8_t* contents /** numpy */) {
    auto casted = reinterpret_cast<DefaultBooleanVector*>(ptr);
    const auto& vals = casted->base.values;
    std::copy(vals.begin(), vals.end(), contents);
    return !(casted->base.missing.empty());
}

//[[export]]
void uzuki2_get_boolean_vector_mask(void* ptr, uint8_t* mask /** numpy */) {
    auto casted = reinterpret_cast<DefaultBooleanVector*>(ptr);
    for (auto i : casted->base.missing) {
        mask[i] = 1;
    }
}

/* String vector handlers */

//[[export]]
int32_t uzuki2_get_string_vector_length(void* ptr) {
    auto casted = reinterpret_cast<DefaultStringVector*>(ptr);
    if (casted->base.scalar) {
        return -1;
    } else {
        return casted->size();
    }
}

//[[export]]
uint64_t uzuki2_get_string_vector_lengths(void* ptr, int32_t* lengths /** numpy */) {
    const auto& vals = reinterpret_cast<DefaultStringVector*>(ptr)->base.values;
    uint64_t total = 0;
    for (auto x : vals) {
        *lengths = x.size();
        total += x.size();
        ++lengths;
    }
    return total;
}

//[[export]]
uint8_t uzuki2_get_string_vector_contents(void* ptr, char* contents) {
    auto casted = reinterpret_cast<DefaultStringVector*>(ptr);
    const auto& vals = casted->base.values;
    for (const auto& x : vals) {
        std::copy(x.begin(), x.end(), contents);
        contents += x.size();
    }
    return !(casted->base.missing.empty());
}

//[[export]]
void uzuki2_get_string_vector_mask(void* ptr, uint8_t* mask /** numpy */) {
    auto casted = reinterpret_cast<DefaultStringVector*>(ptr);
    for (auto i : casted->base.missing) {
        mask[i] = 1;
    }
}

/* List handlers */

//[[export]]
int32_t uzuki2_get_list_length(void* ptr) {
    return reinterpret_cast<DefaultList*>(ptr)->size(); 
}

//[[export]]
uint8_t uzuki2_get_list_named(void* ptr) {
    return reinterpret_cast<DefaultList*>(ptr)->has_names;
}

//[[export]]
uint64_t uzuki2_get_list_names_lengths(void* ptr, int32_t* lengths /** numpy */) {
    const auto& names = reinterpret_cast<DefaultList*>(ptr)->names;
    uint64_t total = 0;
    for (const auto& x : names) {
        *lengths = x.size();
        total += x.size();
        ++lengths;
    }
    return total;
}

//[[export]]
void uzuki2_get_list_names_contents(void* ptr, char* contents) {
    const auto& names = reinterpret_cast<DefaultList*>(ptr)->names;
    for (const auto& x : names) {
        std::copy(x.begin(), x.end(), contents);
        contents += x.size();
    }
    return;
}

//[[export]]
void* uzuki2_get_list_element(void* ptr, int32_t i) {
    return reinterpret_cast<DefaultList*>(ptr)->values[i].get();
}

/* External handlers */

//[[export]]
int32_t uzuki2_get_external_index(void* ptr) {
    return reinterpret_cast<uintptr_t>(reinterpret_cast<DefaultExternal*>(ptr)->ptr);
}
