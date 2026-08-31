// SPDX-License-Identifier: Apache-2.0
//
// List of the bundled dictionaries. The tables under generated/ are gathered here.
//
// The sixteen DICT_NxN_* dictionaries predefined by OpenCV 4.x, plus
// DICT_ARUCO_MIP_36h12, in the order of the OpenCV PredefinedDictionaryType
// enumeration, so that builtin_dictionary_at() walks them as OpenCV declares
// them.
//
// Two of the OpenCV predefined dictionaries are deliberately left out.
//
//   The four AprilTag families are not ArUco.
//
//   DICT_ARUCO_ORIGINAL contains markers that are unchanged by a 180 degree
//   rotation, so their orientation cannot be recovered: rotation 2 decodes as
//   rotation 0 and rotation 3 as rotation 1. Its minimum Hamming distance over
//   all ids and rotations is also 1, meaning two of its markers differ by a
//   single bit, while OpenCV declares one correctable bit for it. Bundling a
//   dictionary whose orientation is ambiguous would put a defect into every
//   pose the caller computes.
//
// DICT_ARUCO_MIP_36h12 is the only MIP dictionary present. The other MIP
// families exist solely in the GPLv3 ArUco distribution, which the code
// provenance rules in CONTRIBUTING.md rule out as a source.
//
// This file is written by hand. The tables it refers to are not: see tools/dictgen,
// and the regeneration check in test/cli.
#include "aruco3cuda/dictionary.hpp"

#include <cstddef>
#include <cstring>

namespace aruco3cuda {

namespace generated {
extern const DictionaryTable kDict4x450Table;
extern const DictionaryTable kDict4x4100Table;
extern const DictionaryTable kDict4x4250Table;
extern const DictionaryTable kDict4x41000Table;
extern const DictionaryTable kDict5x550Table;
extern const DictionaryTable kDict5x5100Table;
extern const DictionaryTable kDict5x5250Table;
extern const DictionaryTable kDict5x51000Table;
extern const DictionaryTable kDict6x650Table;
extern const DictionaryTable kDict6x6100Table;
extern const DictionaryTable kDict6x6250Table;
extern const DictionaryTable kDict6x61000Table;
extern const DictionaryTable kDict7x750Table;
extern const DictionaryTable kDict7x7100Table;
extern const DictionaryTable kDict7x7250Table;
extern const DictionaryTable kDict7x71000Table;
extern const DictionaryTable kDictArucoMip36h12Table;
}  // namespace generated

namespace {

const DictionaryTable* const kBuiltinTables[] = {
        &generated::kDict4x450Table,         &generated::kDict4x4100Table,
        &generated::kDict4x4250Table,        &generated::kDict4x41000Table,
        &generated::kDict5x550Table,         &generated::kDict5x5100Table,
        &generated::kDict5x5250Table,        &generated::kDict5x51000Table,
        &generated::kDict6x650Table,         &generated::kDict6x6100Table,
        &generated::kDict6x6250Table,        &generated::kDict6x61000Table,
        &generated::kDict7x750Table,         &generated::kDict7x7100Table,
        &generated::kDict7x7250Table,        &generated::kDict7x71000Table,
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
