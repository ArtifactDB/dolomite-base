#include "uzuki2/uzuki2.hpp"
#include <cstring>

//[[export]]
void extract_r_missing_double(double* buffer /** numpy */) {
    *buffer = uzuki2::hdf5::legacy_missing_double();
    return;
}

//[[export]]
void fill_nan_mask(void* values, int32_t number, void* placeholder, int32_t size, uint8_t* mask /** numpy */) {
    auto vptr = reinterpret_cast<const unsigned char*>(values);
    auto pptr = reinterpret_cast<const unsigned char*>(placeholder);
    for (int32_t i = 0; i < number; ++i, vptr += size) {
        mask[i] = (std::memcmp(vptr, pptr, size) == 0);
    }
}
