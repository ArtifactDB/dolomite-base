#include "comservatory/comservatory.hpp"
#include <cstdint>

//[[export]]
void* load_csv(const char* path) {
    comservatory::ReadCsv reader;
    auto contents = reader.read(path); // throws error for invalid formats.
    return new comservatory::Contents(std::move(contents));
}

//[[export]]
void validate_csv(const char* path) {
    comservatory::ReadCsv reader;
    reader.validate_only = true;
    reader.read(path);
}

//[[export]]
void free_csv(void* ptr) {
    delete reinterpret_cast<comservatory::Contents*>(ptr);
}

//[[export]]
int32_t get_csv_num_fields(void* ptr) {
    return reinterpret_cast<const comservatory::Contents*>(ptr)->num_fields();
}

//[[export]]
int32_t get_csv_num_records(void* ptr) {
    return reinterpret_cast<const comservatory::Contents*>(ptr)->num_records();
}

//[[export]]
void get_csv_column_stats(void* ptr, int32_t column, int32_t* type, int32_t* size, int32_t* loaded) {
    auto mat = reinterpret_cast<const comservatory::Contents*>(ptr);
    const auto& current = mat->fields[column];

    // Manually specifying it here, so that we're robust to changes in comservatory itself.
    auto mytype = current->type();
    if (mytype == comservatory::STRING) {
        *type = 0;
    } else if (mytype == comservatory::NUMBER) {
        *type = 1;
    } else if (mytype == comservatory::COMPLEX) {
        *type = 2;
    } else if (mytype == comservatory::BOOLEAN) {
        *type = 3;
    } else if (mytype == comservatory::UNKNOWN) {
        *type = -1;
    }

    *size = current->size();
    *loaded = current->filled();
}

//[[export]]
uint8_t fetch_csv_numbers(void* ptr, int32_t column, double* contents /** numpy */, uint8_t* mask /** numpy */) {
    auto mat = reinterpret_cast<comservatory::Contents*>(ptr);
    auto& current = mat->fields[column];

    auto nptr = reinterpret_cast<const comservatory::FilledNumberField*>(current.get());
    std::copy(nptr->values.begin(), nptr->values.end(), contents);
    for (auto i : nptr->missing) {
        mask[i] = 1;
    }

    return !(nptr->missing.empty());
}

//[[export]]
uint8_t fetch_csv_booleans(void* ptr, int32_t column, uint8_t* contents /** numpy */) {
    auto mat = reinterpret_cast<comservatory::Contents*>(ptr);
    auto& current = mat->fields[column];

    auto nptr = reinterpret_cast<const comservatory::FilledBooleanField*>(current.get());
    std::copy(nptr->values.begin(), nptr->values.end(), contents);
    for (auto i : nptr->missing) {
        contents[i] = 2;
    }

    return !(nptr->missing.empty());
}

//[[export]]
uint8_t get_csv_string_stats(void* ptr, int32_t column, int32_t* lengths /** numpy */, uint8_t* mask /** numpy */) {
    auto mat = reinterpret_cast<const comservatory::Contents*>(ptr);
    const auto& current = mat->fields[column];

    auto nptr = reinterpret_cast<const comservatory::FilledStringField*>(current.get());
    for (const auto& x : nptr->values) {
        *lengths = x.size();
        ++lengths;
    }

    for (auto i : nptr->missing) {
        mask[i] = 1;
    }

    return !(nptr->missing.empty());
}

//[[export]]
void fetch_csv_strings(void* ptr, int32_t column, char* contents) {
    auto mat = reinterpret_cast<comservatory::Contents*>(ptr);
    auto& current = mat->fields[column];

    auto nptr = reinterpret_cast<const comservatory::FilledStringField*>(current.get());
    for (const auto& x : nptr->values) {
        std::copy(x.begin(), x.end(), contents);
        contents += x.size();
    }
}
