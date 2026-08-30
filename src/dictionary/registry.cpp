// SPDX-License-Identifier: Apache-2.0
//
// List of the bundled dictionaries. The tables under generated/ are gathered here.
#include "aruco3cuda/dictionary.hpp"

#include <cstddef>
#include <cstring>

namespace aruco3cuda {

namespace generated {
extern const DictionaryTable kDictArucoMip36h12Table;
}  // namespace generated

namespace {

const DictionaryTable* const kBuiltinTables[] = {
        &generated::kDictArucoMip36h12Table,
};

constexpr std::size_t kBuiltinCount = sizeof(kBuiltinTables) / sizeof(kBuiltinTables[0]);

}  // namespace

std::size_t builtin_dictionary_count() {
    return kBuiltinCount;
}

const DictionaryTable* builtin_dictionary_at(std::size_t index) {
    if (index >= kBuiltinCount) {
        return nullptr;
    }
    return kBuiltinTables[index];
}

const DictionaryTable* find_builtin_dictionary(const char* name) {
    if (name == nullptr) {
        return nullptr;
    }
    for (std::size_t i = 0; i < kBuiltinCount; ++i) {
        if (std::strcmp(kBuiltinTables[i]->name_, name) == 0) {
            return kBuiltinTables[i];
        }
    }
    return nullptr;
}

}  // namespace aruco3cuda
