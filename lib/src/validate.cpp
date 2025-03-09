#include "takane/takane.hpp"
#include "pybind11/pybind11.h"

std::shared_ptr<millijson::Base> convert_to_millijson(const pybind11::handle& x) {
    std::shared_ptr<millijson::Base> output;

    if (pybind11::isinstance<pybind11::none>(x)) {
        output.reset(new millijson::Nothing);
    } else if (pybind11::isinstance<pybind11::bool_>(x)) {
        output.reset(new millijson::Boolean(pybind11::cast<bool>(x)));
    } else if (pybind11::isinstance<pybind11::int_>(x)) {
        output.reset(new millijson::Number(pybind11::cast<double>(x)));
    } else if (pybind11::isinstance<pybind11::float_>(x)) {
        output.reset(new millijson::Number(pybind11::cast<double>(x)));
    } else if (pybind11::isinstance<pybind11::str>(x)) {
        output.reset(new millijson::String(pybind11::cast<std::string>(x)));
    } else if (pybind11::isinstance<pybind11::list>(x)) {
        auto y = pybind11::reinterpret_borrow<pybind11::list>(x);
        auto aptr = new millijson::Array;
        output.reset(aptr);
        for (size_t e = 0, end = y.size(); e < end; ++e) {
            aptr->add(convert_to_millijson(y[e]));
        }
    } else if (pybind11::isinstance<pybind11::dict>(x)) {
        auto y = pybind11::reinterpret_borrow<pybind11::dict>(x);
        auto optr = new millijson::Object;
        output.reset(optr);
        for (auto it = y.begin(); it != y.end(); ++it) {
            auto field = pybind11::cast<std::string>(it->first);
            optr->add(std::move(field), convert_to_millijson(it->second));
        } 
    } else {
        throw std::runtime_error("cannot convert unknown python object to JSON");
    }

    return output;
}

pybind11::object convert_to_python(const millijson::Base* x) {
    if (x->type() == millijson::NOTHING) {
        return pybind11::none();
    } else if (x->type() == millijson::BOOLEAN) {
        return pybind11::bool_(reinterpret_cast<const millijson::Boolean*>(x)->value);
    } else if (x->type() == millijson::NUMBER) {
        return pybind11::float_(reinterpret_cast<const millijson::Number*>(x)->value);
    } else if (x->type() == millijson::STRING) {
        return pybind11::str(reinterpret_cast<const millijson::String*>(x)->value);
    } else if (x->type() == millijson::ARRAY) {
        const auto& y = reinterpret_cast<const millijson::Array*>(x)->values;
        pybind11::list output(y.size());
        for (size_t i = 0, end = y.size(); i < end; ++i) {
            output[i] = convert_to_python(y[i].get());
        }
        return output;
    } else if (x->type() == millijson::OBJECT) {
        const auto& y = reinterpret_cast<const millijson::Object*>(x)->values;
        pybind11::dict output;
        for (const auto& pair : y) {
            output[pair.first.c_str()] = convert_to_python(pair.second.get());
        }
        return output;
    } else {
        throw std::runtime_error("unknown millijson type '" + std::to_string(x->type()) + "'");
        return pybind11::none();
    }
}

pybind11::dict convert_to_python(const takane::ObjectMetadata& x) {
    pybind11::dict output;
    output["type"] = x.type;
    for (const auto& pair : x.other) {
        output[pair.first.c_str()] = convert_to_python(pair.second.get());
    }
    return output;
}

void validate(std::string path, pybind11::handle metadata, pybind11::dict custom) {
    // We re-create the Options on every call to avoid problems with storing
    // the pybind11::function object in a static global (via the lambda
    // capture) - this results in GIL-related issues upon destruction.
    takane::Options options;
    for (auto it = custom.begin(); it != custom.end(); ++it) {
        if (!pybind11::isinstance<pybind11::str>(it->first)) {
            throw std::runtime_error("keys of 'validate_object_registry' should be strings");
        }
        auto objname = pybind11::cast<std::string>(it->first);

        if (!pybind11::isinstance<pybind11::function>(it->second)) {
            throw std::runtime_error("expected 'validate_object_registry' to contain a function for '" + objname + "'");
        }
        auto fun = pybind11::reinterpret_borrow<pybind11::function>(it->second);
        options.custom_validate[std::move(objname)] = [fun](const std::filesystem::path& path, const takane::ObjectMetadata& metadata, takane::Options&) {
             fun(pybind11::str(path.c_str()), convert_to_python(metadata));
             return;
        };
    }

    if (pybind11::isinstance<pybind11::none>(metadata)) {
        takane::validate(path, options);
    } else {
        auto converted = convert_to_millijson(metadata);
        auto objmeta = takane::reformat_object_metadata(converted.get());
        takane::validate(path, objmeta, options);
    }
}
