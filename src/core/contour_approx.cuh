// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
// Closed RDP adapted from pinned OpenCV approx.cpp; see contour_candidates.md.
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
#ifndef ARUCO3CUDA_CORE_CONTOUR_APPROX_CUH
#define ARUCO3CUDA_CORE_CONTOUR_APPROX_CUH
#include <cmath>
#include "contour_candidates.hpp"
#if defined(__CUDACC__)
#define ARUCO3CUDA_CONTOUR_HD __host__ __device__
#else
#define ARUCO3CUDA_CONTOUR_HD
#endif
namespace aruco3cuda::detail {

ARUCO3CUDA_CONTOUR_HD inline ContourPoint contour_read(const ContourPoint* points, int& position,
                                                       int count) {
    const ContourPoint result = points[position];
    if (++position >= count) {
        position = 0;
    }
    return result;
}

/// The pinned almost-collinear cleanup is shared by serial and cooperative RDP.
ARUCO3CUDA_CONTOUR_HD inline int contour_cleanup_closed(ContourPoint* destination, int count,
                                                        double epsilon_squared) {
    int written = count;
    int position = count - 1;
    ContourPoint start = contour_read(destination, position, count);
    int write_position = position;
    ContourPoint point = contour_read(destination, position, count);
    for (int index = 0; index < count && written > 2; ++index) {
        const ContourPoint end = contour_read(destination, position, count);
        const double dx = end.x_ - start.x_;
        const double dy = end.y_ - start.y_;
        const double cross = fabs((point.x_ - start.x_) * dy - (point.y_ - start.y_) * dx);
        const double inner = static_cast<double>(point.x_ - start.x_) * (end.x_ - point.x_) +
                             static_cast<double>(point.y_ - start.y_) * (end.y_ - point.y_);
        if (cross * cross <= 0.5 * epsilon_squared * (dx * dx + dy * dy) && dx != 0.0 &&
            dy != 0.0 && inner >= 0.0) {
            --written;
            destination[write_position] = start = end;
            if (++write_position >= count) {
                write_position = 0;
            }
            point = contour_read(destination, position, count);
            ++index;
        } else {
            destination[write_position] = start = point;
            if (++write_position >= count) {
                write_position = 0;
            }
            point = end;
        }
    }
    return written;
}

/// Closed integer RDP with the pinned oracle's farthest-point initialization,
/// segment-distance rule and final almost-collinear cleanup. Scratch capacity is
/// exactly count; a negative return reports malformed geometry or overflow.
ARUCO3CUDA_CONTOUR_HD inline int contour_approximate_closed(const ContourPoint* source, int count,
                                                            ContourPoint* destination,
                                                            ContourSlice* stack, double epsilon) {
    if (count <= 0) {
        return 0;
    }
    const int original_count = count;
    const double epsilon_squared = epsilon * epsilon;
    ContourSlice slice{};
    ContourSlice right{};
    ContourPoint start{};
    ContourPoint end{};
    ContourPoint point{};
    int position = 0;
    int written = 0;
    int top = 0;
    bool within = false;
    for (int iteration = 0; iteration < 3; ++iteration) {
        double maximum = 0.0;
        position = (position + right.first_) % count;
        start = contour_read(source, position, count);
        for (int index = 1; index < count; ++index) {
            point = contour_read(source, position, count);
            const double dx = point.x_ - start.x_;
            const double dy = point.y_ - start.y_;
            const double squared = dx * dx + dy * dy;
            if (squared > maximum) {
                maximum = squared;
                right.first_ = index;
            }
        }
        within = maximum <= epsilon_squared;
    }
    if (!within) {
        right.last_ = slice.first_ = position % count;
        slice.last_ = right.first_ = (right.first_ + slice.first_) % count;
        if (count < 2) {
            return -1;
        }
        stack[top++] = right;
        stack[top++] = slice;
    } else {
        destination[written++] = start;
    }
    while (top > 0) {
        slice = stack[--top];
        end = source[slice.last_];
        position = slice.first_;
        start = contour_read(source, position, count);
        if (position != slice.last_) {
            const double dx = end.x_ - start.x_;
            const double dy = end.y_ - start.y_;
            const double segment_squared = dx * dx + dy * dy;
            if (segment_squared == 0.0) {
                return -1;
            }
            double maximum = 0.0;
            while (position != slice.last_) {
                point = contour_read(source, position, count);
                const double px = point.x_ - start.x_;
                const double py = point.y_ - start.y_;
                const double projection = px * dx + py * dy;
                double squared = 0.0;
                if (projection < 0.0) {
                    squared = (px * px + py * py) * segment_squared;
                } else if (projection > segment_squared) {
                    const double ex = point.x_ - end.x_;
                    const double ey = point.y_ - end.y_;
                    squared = (ex * ex + ey * ey) * segment_squared;
                } else {
                    const double cross = py * dx - px * dy;
                    squared = cross * cross;
                }
                if (squared > maximum) {
                    maximum = squared;
                    right.first_ = (position + count - 1) % count;
                }
            }
            within = maximum <= epsilon_squared * segment_squared;
        } else {
            within = true;
            start = source[slice.first_];
        }
        if (within) {
            if (written >= original_count) {
                return -1;
            }
            destination[written++] = start;
        } else {
            right.last_ = slice.last_;
            slice.last_ = right.first_;
            if (top + 2 > original_count) {
                return -1;
            }
            stack[top++] = right;
            stack[top++] = slice;
        }
    }
    return contour_cleanup_closed(destination, written, epsilon_squared);
}
#if defined(__CUDACC__)
/// Stable warp maximum: equal distances choose the earliest traversal index.
/// All 32 lanes participate; reductions select values and do not sum distances.
__device__ inline void contour_warp_maximum(double& maximum, int& first) {
    for (int delta = 16; delta > 0; delta >>= 1) {
        const double other = __shfl_down_sync(0xffffffffU, maximum, delta);
        const int index = __shfl_down_sync(0xffffffffU, first, delta);
        if (other > maximum || (other == maximum && index < first)) {
            maximum = other;
            first = index;
        }
    }
    maximum = __shfl_sync(0xffffffffU, maximum, 0);
    first = __shfl_sync(0xffffffffU, first, 0);
}

/// One full warp follows the serial stack order. Independent point distances
/// are distributed over lanes; the same arithmetic and earliest-maximum ties
/// preserve the serial oracle's split decisions. Lane zero owns all writes.
__device__ inline int contour_approximate_closed_warp(const ContourPoint* source, int count,
                                                      ContourPoint* destination,
                                                      ContourSlice* stack, double epsilon) {
    const unsigned int lane = threadIdx.x & 31U;
    if (count <= 0) {
        return 0;
    }
    const double epsilon_squared = epsilon * epsilon;
    ContourSlice slice{};
    ContourSlice right{};
    ContourPoint start{};
    int position = 0;
    int written = 0;
    int top = 0;
    bool within = false;
    for (int iteration = 0; iteration < 3; ++iteration) {
        position = (position + right.first_) % count;
        start = source[position];
        double maximum = 0.0;
        int first = 2147483647;
        for (int index = static_cast<int>(lane) + 1; index < count; index += 32) {
            const ContourPoint point = source[(position + index) % count];
            const double dx = point.x_ - start.x_;
            const double dy = point.y_ - start.y_;
            const double squared = dx * dx + dy * dy;
            if (squared > maximum) {
                maximum = squared;
                first = index;
            }
        }
        contour_warp_maximum(maximum, first);
        if (maximum > 0.0) {
            right.first_ = first;
        }
        within = maximum <= epsilon_squared;
    }
    if (!within) {
        right.last_ = slice.first_ = position;
        slice.last_ = right.first_ = (right.first_ + slice.first_) % count;
        if (count < 2) {
            return -1;
        }
        if (lane == 0U) {
            stack[0] = right;
            stack[1] = slice;
        }
        top = 2;
    } else {
        if (lane == 0U) {
            destination[0] = start;
        }
        written = 1;
    }
    __syncwarp();
    while (top > 0) {
        slice = stack[--top];
        const ContourPoint end = source[slice.last_];
        start = source[slice.first_];
        position = (slice.first_ + 1) % count;
        if (position != slice.last_) {
            const double dx = end.x_ - start.x_;
            const double dy = end.y_ - start.y_;
            const double segment_squared = dx * dx + dy * dy;
            if (segment_squared == 0.0) {
                return -1;
            }
            const int intermediate = (slice.last_ - position + count) % count;
            double maximum = 0.0;
            int first = 2147483647;
            for (int index = static_cast<int>(lane); index < intermediate; index += 32) {
                const ContourPoint point = source[(position + index) % count];
                const double px = point.x_ - start.x_;
                const double py = point.y_ - start.y_;
                const double projection = px * dx + py * dy;
                double squared = 0.0;
                if (projection < 0.0) {
                    squared = (px * px + py * py) * segment_squared;
                } else if (projection > segment_squared) {
                    const double ex = point.x_ - end.x_;
                    const double ey = point.y_ - end.y_;
                    squared = (ex * ex + ey * ey) * segment_squared;
                } else {
                    const double cross = py * dx - px * dy;
                    squared = cross * cross;
                }
                if (squared > maximum) {
                    maximum = squared;
                    first = index;
                }
            }
            contour_warp_maximum(maximum, first);
            if (maximum > 0.0) {
                right.first_ = (position + first) % count;
            }
            within = maximum <= epsilon_squared * segment_squared;
        } else {
            within = true;
        }
        if (within) {
            if (written >= count) {
                return -1;
            }
            if (lane == 0U) {
                destination[written] = start;
            }
            ++written;
        } else {
            right.last_ = slice.last_;
            slice.last_ = right.first_;
            if (top + 2 > count) {
                return -1;
            }
            if (lane == 0U) {
                stack[top] = right;
                stack[top + 1] = slice;
            }
            top += 2;
        }
        __syncwarp();
    }
    int result = 0;
    if (lane == 0U) {
        result = contour_cleanup_closed(destination, written, epsilon_squared);
    }
    return __shfl_sync(0xffffffffU, result, 0);
}
#endif
}  // namespace aruco3cuda::detail
#undef ARUCO3CUDA_CONTOUR_HD
#endif  // ARUCO3CUDA_CORE_CONTOUR_APPROX_CUH
