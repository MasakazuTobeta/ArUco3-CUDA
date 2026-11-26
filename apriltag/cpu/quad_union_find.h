// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_APRILTAG_CPU_QUAD_UNION_FIND_H
#define ARUCO3CUDA_APRILTAG_CPU_QUAD_UNION_FIND_H

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

// A zero entry represents an isolated pixel. A negative entry stores a root's
// component size, and a positive entry stores the parent index plus one.
typedef struct unionfind {
    uint32_t max_id;
    int32_t entries[];
} unionfind_t;

static inline unionfind_t* unionfind_create(uint32_t max_id) {
    // Keep both the encoded parent and the largest component size representable.
    const size_t capacity = (SIZE_MAX - sizeof(unionfind_t)) / sizeof(int32_t);
    if (max_id >= INT32_MAX || max_id >= capacity)
        return NULL;
    const size_t count = (size_t)max_id + 1U;
    unionfind_t* set = calloc(1, sizeof(*set) + count * sizeof(*set->entries));
    if (set)
        set->max_id = max_id;
    return set;
}

static inline void unionfind_destroy(unionfind_t* set) {
    free(set);
}

static inline uint32_t unionfind_get_representative(unionfind_t* set, uint32_t id) {
    assert(id <= set->max_id);
    int32_t node = set->entries[id];
    while (node > 0) {
        const uint32_t parent = (uint32_t)(node - 1);
        node = set->entries[parent];
        if (node <= 0)
            return parent;
        set->entries[id] = node;
        id = (uint32_t)(node - 1);
        node = set->entries[id];
    }
    return id;
}

// After the last union, boundary-cluster workers can share this forest.
static inline uint32_t unionfind_find_read_only(const unionfind_t* set, uint32_t id) {
    assert(id <= set->max_id);
    int32_t node = set->entries[id];
    while (node > 0) {
        id = (uint32_t)(node - 1);
        node = set->entries[id];
    }
    return id;
}

static inline int32_t unionfind_root_size(const unionfind_t* set, uint32_t root) {
    const int32_t node = set->entries[root];
    assert(node <= 0);
    return node == 0 ? 1 : -node;
}

static inline uint32_t unionfind_get_set_size(unionfind_t* set, uint32_t id) {
    return (uint32_t)unionfind_root_size(set, unionfind_get_representative(set, id));
}

static inline uint32_t unionfind_connect(unionfind_t* set, uint32_t a, uint32_t b) {
    const uint32_t first = unionfind_get_representative(set, a);
    const uint32_t second = unionfind_get_representative(set, b);
    if (first == second)
        return first;

    const int32_t first_size = unionfind_root_size(set, first);
    const int32_t second_size = unionfind_root_size(set, second);
    // Keep the pinned extractor's tie-breaking rule. Region-pair keys depend on
    // representative IDs, not just on the partition of the pixels.
    const uint32_t root = first_size > second_size ? first : second;
    const uint32_t child = root == first ? second : first;
    set->entries[child] = (int32_t)root + 1;
    set->entries[root] = -(first_size + second_size);
    return root;
}

#endif  // ARUCO3CUDA_APRILTAG_CPU_QUAD_UNION_FIND_H
