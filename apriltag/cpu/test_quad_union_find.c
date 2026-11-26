// SPDX-License-Identifier: Apache-2.0
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static bool fail_allocations;
static size_t allocation_count;
static size_t allocation_bytes;

static void* record_calloc(size_t count, size_t size) {
    ++allocation_count;
    allocation_bytes = count * size;
    return fail_allocations ? NULL : calloc(count, size);
}

// Substitute the allocator for this translation unit only. The largest accepted
// size is checked without asking the host to reserve that much memory.
#define calloc record_calloc
#include "quad_union_find.h"
#undef calloc

static void check(bool condition, const char* message) {
    if (!condition) {
        fprintf(stderr, "%s\n", message);
        exit(EXIT_FAILURE);
    }
}

static void check_encoding(void) {
    unionfind_t* set = unionfind_create(5);
    check(set != NULL, "small forest allocation failed");
    check(allocation_bytes == sizeof(*set) + 6 * sizeof(*set->entries),
          "allocation must include the maximum index");
    for (uint32_t i = 0; i <= 5; ++i) {
        check(set->entries[i] == 0, "isolated pixels must use the zero encoding");
        check(unionfind_find_read_only(set, i) == i, "isolated pixel has the wrong root");
    }
    check(unionfind_get_set_size(set, 5) == 1, "isolated pixel must have size one");

    check(unionfind_connect(set, 1, 0) == 0, "equal singleton sizes must select the second root");
    check(set->entries[1] == 1, "parent zero must use the positive encoding");
    check(set->entries[0] == -2, "a nonempty root must store its negative size");
    check(unionfind_connect(set, 3, 2) == 2, "second pair has the wrong root");
    check(unionfind_connect(set, 1, 3) == 2, "equal component sizes must select the second root");
    check(unionfind_root_size(set, 2) == 4, "merged component has the wrong size");

    check(unionfind_find_read_only(set, 1) == 2, "read-only lookup must follow the parent chain");
    check(set->entries[1] == 1, "read-only lookup must leave the parent chain intact");
    check(unionfind_get_representative(set, 1) == 2, "mutable lookup returned the wrong root");
    check(set->entries[1] == 3, "mutable lookup must shorten this two-hop chain");

    check(unionfind_connect(set, 1, 4) == 2, "the larger first component must remain the root");
    check(unionfind_connect(set, 5, 0) == 2, "the larger second component must remain the root");
    check(unionfind_connect(set, 4, 5) == 2, "union within one component must return its root");
    check(unionfind_get_set_size(set, 5) == 6, "repeated union must preserve the component size");
    unionfind_destroy(set);
}

static void check_allocation_bounds(void) {
    const size_t capacity = (SIZE_MAX - sizeof(unionfind_t)) / sizeof(int32_t);
    const size_t count = capacity < INT32_MAX ? capacity : INT32_MAX;
    allocation_count = 0;
    fail_allocations = true;

    check(unionfind_create(INT32_MAX) == NULL, "reject an unrepresentable parent index");
    if (capacity < INT32_MAX)
        check(unionfind_create((uint32_t)capacity) == NULL, "reject overflowing allocation size");
    check(allocation_count == 0, "invalid dimensions must be rejected before allocation");

    check(unionfind_create((uint32_t)(count - 1)) == NULL, "propagate allocator failure");
    check(allocation_count == 1, "valid boundary size must reach the allocator");
    check(allocation_bytes == sizeof(unionfind_t) + count * sizeof(int32_t),
          "boundary allocation must use the checked byte count");
}

int main(void) {
    check_encoding();
    check_allocation_bounds();
    return EXIT_SUCCESS;
}
