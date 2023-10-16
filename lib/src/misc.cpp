#include "uzuki2/uzuki2.hpp"

//[[export]]
void extract_r_missing_double(double* buffer /** numpy */) {
    *buffer = uzuki2::hdf5::legacy_missing_double();
    return;
}
