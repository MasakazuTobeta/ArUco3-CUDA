// SPDX-License-Identifier: Apache-2.0
#include "aruco3cuda/apriltag_dictionary.hpp"

#include <cstring>

namespace aruco3cuda {
namespace generated {
extern const DictionaryTable kDictApriltag16h5Table;
extern const DictionaryTable kDictApriltag25h9Table;
extern const DictionaryTable kDictApriltag36h10Table;
extern const DictionaryTable kDictApriltag36h11Table;
}  // namespace generated

const DictionaryTable* find_apriltag_dictionary(const char* name) {
    if (name == nullptr) {
        return nullptr;
    }
    const DictionaryTable* const families[] = {
            &generated::kDictApriltag16h5Table, &generated::kDictApriltag25h9Table,
            &generated::kDictApriltag36h10Table, &generated::kDictApriltag36h11Table};
    for (const DictionaryTable* family : families) {
        if (std::strcmp(name, family->name_) == 0) {
            return family;
        }
    }
    return nullptr;
}
}  // namespace aruco3cuda
