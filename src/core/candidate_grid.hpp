// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_CORE_CANDIDATE_GRID_HPP
#define ARUCO3CUDA_CORE_CANDIDATE_GRID_HPP

#ifndef ARUCO3CUDA_EXPERIMENT_PERSISTENT_CANDIDATES
#define ARUCO3CUDA_EXPERIMENT_PERSISTENT_CANDIDATES 0
#endif
#ifndef ARUCO3CUDA_CANDIDATE_GRID_BLOCKS
#define ARUCO3CUDA_CANDIDATE_GRID_BLOCKS 128
#endif

namespace aruco3cuda::detail {
static_assert(ARUCO3CUDA_CANDIDATE_GRID_BLOCKS > 0,
              "candidate grid must contain at least one block");

/// Select a bounded grid for current-frame device-count iteration.
/// @param capacity Positive allocated candidate capacity, already validated.
/// @return Capacity for the control route, otherwise the smaller configured grid.
/// Ownership: holds no resources and retains no argument.
/// Synchronization: host arithmetic only; never reads the device count.
/// Example input: capacity 16384 with persistent mode and a 128-block limit.
/// Example output: 128, independent of the current frame's candidate count.
inline unsigned int candidate_grid_blocks(int capacity) {
#if ARUCO3CUDA_EXPERIMENT_PERSISTENT_CANDIDATES
    return static_cast<unsigned int>(capacity < ARUCO3CUDA_CANDIDATE_GRID_BLOCKS
                                             ? capacity
                                             : ARUCO3CUDA_CANDIDATE_GRID_BLOCKS);
#else
    return static_cast<unsigned int>(capacity);
#endif
}
}  // namespace aruco3cuda::detail
#endif
