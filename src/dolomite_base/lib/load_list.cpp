#include "uzuki2/uzuki2.hpp"
#include <cstdint>

//[[export]]
void validate_list_json(const char* path, int32_t n) {
    uzuki2::json::validate_file(path, n);
    return;
}
