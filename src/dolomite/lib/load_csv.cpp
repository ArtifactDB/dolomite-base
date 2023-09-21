#include "comservatory/comservatory.hpp"
#include <cstdint>

//[[export]]
void* load_csv(const char* path) {
    comservatory::ReadCsv reader;
    auto contents = reader.read(path); // throws error for invalid formats.
    return new Contents(std::move(contents));
}

//[[export]]
void free_csv(void* ptr) {
    delete reinterpret_cast<Contents*>(ptr);
}

//[[export]]
void get_column_stats(void* ptr, int32_t column, int32_t* type, int32_t* size, int32_t* loaded) {
    auto mat = reinterpret_cast<const comservatory::Contents*>(ptr);
    const auto& current = mat->fields[column];

    // Manually specifying it here, so that we're robust to changes in comservatory itself.
    auto mytype = current->type();
    if (type == comservatory::STRING) {
        *type = 0;
    } else if (type == comservatory::NUMBER) {
        *type = 1;
    } else if (type == comservatory::COMPLEX) {
        *type = 2;
    } else if (type == comservatory::BOOLEAN) {
        *type = 3;
    } else if (type == comservatory::UNKNOWN) {
        *type = -1;
    }

    *size = current->size();
    *loaded = current->filled();
}

//[[export]]
uint8_t fetch_numbers(void* ptr, int32_t column, double* output /** numpy */, uint8_t* mask /** numpy */, uint8_t pop) {
    auto mat = reinterpret_cast<comservatory::Contents*>(ptr);
    auto& current = mat->fields[column];

    auto nptr = reinterpret_cast<const comservatory::FilledNumberField*>(current.get());
    std::copy(nptr->values.begin(), nptr->values.end(), output);
    for (auto i : nptr->missing) {
        mask[i] = 1;
    }

    if (pop) { // save memory by freeing the memory immediately.
        current.reset();
    }

    return !(nptr->missing.empty());
}

//[[export]]
uint8_t fetch_booleans(void* ptr, int32_t column, uint8_t* output /** numpy */, uint8_t pop) {
    auto mat = reinterpret_cast<comservatory::Contents*>(ptr);
    auto& current = mat->fields[column];

    auto nptr = reinterpret_cast<const comservatory::FilledBooleanField*>(current.get());
    std::copy(nptr->values.begin(), nptr->values.end(), output);
    for (auto i : nptr->missing) {
        output[i] = 2;
    }

    if (pop) { // save memory by freeing the memory immediately.
        current.reset();
    }

    return !(nptr->missing.empty());
}

//[[export]]
uint8_t get_string_stats(void* ptr, int32_t column, int32_t* lengths /** numpy */, uint8_t* mask /** numpy */) {
    auto mat = reinterpret_cast<const comservatory::Contents*>(ptr);
    const auto& current = mat->fields[column];

    auto nptr = reinterpret_cast<const comservatory::FilledStringField*>(current.get());
    for (const auto& x : nptr->values) {
        *lengths = x.size();
        ++lengths;
    }

    for (auto i : nptr->mask) {
        mask[i] = 1;
    }

    return !(nptr->missing.empty());
}

//[[export]]
uint8_t fetch_strings(void* ptr, int32_t column, char* output, uint8_t pop) {
    auto mat = reinterpret_cast<comservatory::Contents*>(ptr);
    auto& current = mat->fields[column];

    auto nptr = reinterpret_cast<const comservatory::FilledStringField*>(current.get());
    for (const auto& x : nptr->values) {
        std::copy(x.begin(), x.end(), output);
        output += x.size();
    }

    if (pop) { // save memory by freeing the memory immediately.
        current.reset();
    }

    return !(nptr->missing.empty());
}
