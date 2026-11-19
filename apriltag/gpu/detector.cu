// SPDX-License-Identifier: Apache-2.0
#include "aruco3cuda/apriltag/gpu/detector.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace aruco3cuda::apriltag_gpu {
namespace {

void check(cudaError_t result, const char* stage) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(stage) + ": " + cudaGetErrorString(result));
    }
}

template <class T>
class Buffer {
public:
    explicit Buffer(std::size_t count) {
        check(cudaMalloc(&this->data_, count * sizeof(T)), "AprilTag workspace allocation");
    }
    ~Buffer() { cudaFree(this->data_); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    T* get() const { return this->data_; }

private:
    T* data_ = nullptr;
};

struct Image {
    const unsigned char* data;
    std::size_t pitch;
    int width;
    int height;
};
struct Size {
    int width;
    int height;
};
using Word = unsigned long long;
struct Cluster {
    Word key;
    Word extrema[8];
    int count;
};
struct DeviceTag {
    int id;
    int hamming;
    int support;
    float xy[8];
};
// The host reads only after stream completion. Mapped memory eliminates a
// second result-copy round trip on nonempty frames, including the fine pass.
class HostTags {
public:
    explicit HostTags(int count) {
        check(cudaHostAlloc(&host, count * sizeof(DeviceTag), cudaHostAllocMapped),
              "AprilTag pinned output");
        auto status = cudaHostGetDevicePointer(&device, host, 0);
        if (status != cudaSuccess) {
            cudaFreeHost(host);
            check(status, "AprilTag output mapping");
        }
    }
    ~HostTags() { cudaFreeHost(host); }
    HostTags(const HostTags&) = delete;
    HostTags& operator=(const HostTags&) = delete;
    DeviceTag* host = nullptr;
    DeviceTag* device = nullptr;
};
struct Counts {
    int tags;
    int overflow;
    int active;
    int fine;
};
constexpr int kThreads = 128;
int blocks(int count) {
    return (count + kThreads - 1) / kThreads;
}

__device__ int linear_index() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}
__device__ int pixel(Image im, int x, int y) {
    return im.data[y * im.pitch + x];
}

// Tiles summarize sampled luminance. The next stage pools neighboring ranges
// so an edge can cross a tile boundary without changing the local decision.
__global__ void tile_range(Image im, Size seg, Size tiles, int step, uchar2* range) {
    const int i = linear_index();
    if (i >= tiles.width * tiles.height)
        return;
    const int x0 = 4 * (i % tiles.width), y0 = 4 * (i / tiles.width);
    int lo = 255, hi = 0;
    for (int y = y0; y < min(y0 + 4, seg.height); ++y) {
        for (int x = x0; x < min(x0 + 4, seg.width); ++x) {
            const int value = pixel(im, x * step, y * step);
            lo = min(lo, value);
            hi = max(hi, value);
        }
    }
    range[i] = make_uchar2(lo, hi);
}

__global__ void classify(Image im, Size seg, Size tiles, const uchar2* range, int step,
                         int contrast, unsigned char* colors) {
    const int i = linear_index();
    if (i >= seg.width * seg.height)
        return;
    const int x = i % seg.width, y = i / seg.width;
    int lo = 255, hi = 0;
    for (int yy = max(0, y / 4 - 1); yy <= min(tiles.height - 1, y / 4 + 1); ++yy) {
        for (int xx = max(0, x / 4 - 1); xx <= min(tiles.width - 1, x / 4 + 1); ++xx) {
            const uchar2 r = range[yy * tiles.width + xx];
            lo = min(lo, int(r.x));
            hi = max(hi, int(r.y));
        }
    }
    colors[i] = hi - lo < contrast ? 2 : (pixel(im, x * step, y * step) > (lo + hi) / 2);
}

__device__ int root(int* parents, int i) {
    int next = atomicAdd(parents + i, 0);
    while (next != i) {
        i = next;
        next = atomicAdd(parents + i, 0);
    }
    return i;
}

__device__ void join(int* parents, int a, int b) {
    for (;;) {
        a = root(parents, a);
        b = root(parents, b);
        if (a == b)
            return;
        const int hi = max(a, b), lo = min(a, b);
        if (atomicCAS(parents + hi, hi, lo) == hi)
            return;
    }
}

// Seed row-runs, then merge adjacent rows. All parent links decrease, including
// concurrent bridge operations. White components also connect diagonally.
__global__ void seed_runs(Size seg, const unsigned char* colors, int* parents) {
    const int i = linear_index();
    const int lane = threadIdx.x & 31;
    const bool valid = i < seg.width * seg.height;
    const bool start = !valid || colors[i] == 2 || i % seg.width == 0 || colors[i] != colors[i - 1];
    const unsigned starts = __ballot_sync(0xffffffffU, start);
    if (valid) {
        const unsigned left = starts & (0xffffffffU >> (31 - lane));
        // A run continuing from the previous warp links to its last pixel.
        const int head_lane = 31 - __clz(left);
        parents[i] = i - lane + head_lane;
    }
}
// Root traversal has no writers to the input plane. This avoids issuing
// globally coherent atomics for the two full-frame compression passes.
__global__ void shorten(const int* parents, int* roots, int count) {
    const int i = linear_index();
    if (i < count) {
        int r = i;
        while (parents[r] != r)
            r = parents[r];
        roots[i] = r;
    }
}

__global__ void merge_rows(Size seg, const unsigned char* colors, int* parents) {
    const int i = linear_index();
    if (i < seg.width || i >= seg.width * seg.height || colors[i] == 2)
        return;
    const int x = i % seg.width, above = i - seg.width;
    const auto color = colors[i];
    if (colors[above] == color) {
        if (!x || colors[i - 1] != color || colors[above - 1] != color)
            join(parents, i, above);
    } else if (color == 1) {
        if (x && colors[above - 1] == color)
            join(parents, i, above - 1);
        if (x + 1 < seg.width && colors[above + 1] == color)
            join(parents, i, above + 1);
    }
}

// Interfaces are grouped by both neighboring components. Distinct dark marker
// borders connected by a cable may still have distinct light neighbors.
__device__ void boundary(int i, int j, int x2, int y2, const int* parents, Cluster* slots,
                         int capacity, int max_probes, Counts* counts) {
    const Word key = (Word(parents[i] + 1) << 32) | unsigned(parents[j] + 1);
    Word hash = key ^ (key >> 33);
    hash *= 0xff51afd7ed558ccdULL;
    int index = int((hash ^ (hash >> 33)) & unsigned(capacity - 1));
    for (int probe = 0; probe < max_probes; ++probe) {
        Cluster* slot = slots + index;
        const Word old = atomicCAS(&slot->key, 0ULL, key);
        if (!old || old == key) {
            atomicAdd(&slot->count, 1);
            const unsigned xy = (unsigned(y2) << 16) | unsigned(x2);
            const int scores[8] = {-x2 - y2, x2 - y2, x2 + y2, y2 - x2, -y2, x2, y2, -x2};
            for (int dir = 0; dir < 8; ++dir) {
                const Word score = (Word(scores[dir] + 131072) << 32) | xy;
                atomicMax(&slot->extrema[dir], score);
            }
            return;
        }
        if ((probe & 15) == 15 && atomicAdd(&counts->overflow, 0))
            return;
        index = (index + 1) & (capacity - 1);
    }
    atomicExch(&counts->overflow, 1);
}

__global__ void cluster_edges(Size seg, const unsigned char* colors, const int* parents,
                              Cluster* slots, int capacity, int max_probes, Counts* counts) {
    const int i = linear_index();
    if (i >= seg.width * seg.height || colors[i] != 0)
        return;
    const int x = i % seg.width, y = i / seg.width;
    if (x && colors[i - 1] == 1)
        boundary(i, i - 1, 2 * x - 1, 2 * y, parents, slots, capacity, max_probes, counts);
    if (x + 1 < seg.width && colors[i + 1] == 1)
        boundary(i, i + 1, 2 * x + 1, 2 * y, parents, slots, capacity, max_probes, counts);
    if (y && colors[i - seg.width] == 1)
        boundary(i, i - seg.width, 2 * x, 2 * y - 1, parents, slots, capacity, max_probes, counts);
    if (y + 1 < seg.height && colors[i + seg.width] == 1)
        boundary(i, i + seg.width, 2 * x, 2 * y + 1, parents, slots, capacity, max_probes, counts);
}

__device__ float2 sub(float2 a, float2 b) {
    return make_float2(a.x - b.x, a.y - b.y);
}
__device__ float cross(float2 a, float2 b) {
    return a.x * b.y - a.y * b.x;
}
__device__ float length(float2 v) {
    return hypotf(v.x, v.y);
}
__device__ float bilinear(Image im, float x, float y) {
    if (!(x >= 0 && y >= 0 && x < im.width - 1 && y < im.height - 1))
        return -1;
    const int ix = int(x), iy = int(y);
    const float fx = x - ix, fy = y - iy;
    const float a = pixel(im, ix, iy) * (1 - fx) + pixel(im, ix + 1, iy) * fx;
    const float b = pixel(im, ix, iy + 1) * (1 - fx) + pixel(im, ix + 1, iy + 1) * fx;
    return a * (1 - fy) + b * fy;
}

struct Line {
    float2 center, direction;
};
__device__ float warp_sum(float value) {
    for (int off = 16; off; off /= 2)
        value += __shfl_down_sync(0xffffffffU, value, off);
    return __shfl_sync(0xffffffffU, value, 0);
}

// Estimate a subpixel line from signed gray-level gradients across the black
// frame. A centered covariance limits loss of precision in large images.
__device__ bool fit_edge(Image im, float2 a, float2 b, int radius, Line& line) {
    const float2 d = sub(b, a);
    const float len = length(d);
    if (len < 6)
        return false;
    const float2 n =
            make_float2(d.y / len, -d.x / len);  // points outward for clockwise image corners
    float sx = 0, sy = 0, sxx = 0, sxy = 0, syy = 0;
    int used = 0;
    const int samples = min(48, max(8, int(len / 4)));
    for (int j = 1 + (threadIdx.x & 31); j <= samples; j += 32) {
        const float t = float(j) / (samples + 1);
        const float x = a.x + t * d.x, y = a.y + t * d.y;
        float weights = 0, offset_sum = 0;
        for (int k = -2 * radius; k <= 2 * radius; ++k) {
            const float off = .5f * k;
            const float bright = bilinear(im, x + (off + .75f) * n.x, y + (off + .75f) * n.y);
            const float dark = bilinear(im, x + (off - .75f) * n.x, y + (off - .75f) * n.y);
            if (bright <= dark || dark < 0)
                continue;
            const float w = (bright - dark) * (bright - dark);
            weights += w;
            offset_sum += off * w;
        }
        if (weights < 1)
            continue;
        const float tx = t * d.x + n.x * offset_sum / weights;
        const float ty = t * d.y + n.y * offset_sum / weights;
        sx += tx;
        sy += ty;
        sxx += tx * tx;
        syy += ty * ty;
        sxy += tx * ty;
        ++used;
    }
    used = int(warp_sum(float(used)));
    sx = warp_sum(sx);
    sy = warp_sum(sy);
    sxx = warp_sum(sxx);
    syy = warp_sum(syy);
    sxy = warp_sum(sxy);
    if (used < 4)
        return false;
    const float angle = .5f * atan2f(2 * (sxy - sx * sy / used),
                                     (sxx - sx * sx / used) - (syy - sy * sy / used));
    line.center = make_float2(a.x + sx / used, a.y + sy / used);
    line.direction = make_float2(cosf(angle), sinf(angle));
    return true;
}

__device__ bool quad_from_cluster(const Cluster& cluster, int step, float2* out) {
    float best = 0;
    for (int group = 0; group < 2; ++group) {
        float2 p[4];
        for (int j = 0; j < 4; ++j) {
            const unsigned xy = unsigned(cluster.extrema[group * 4 + j]);
            p[j] = make_float2((xy & 65535U) * (.5f * step), (xy >> 16) * (.5f * step));
        }
        float min_area = 1e30f;
        for (int j = 0; j < 4; ++j) {
            min_area = fminf(min_area,
                             cross(sub(p[(j + 1) % 4], p[j]), sub(p[(j + 2) % 4], p[(j + 1) % 4])));
        }
        if (min_area > best) {
            best = min_area;
            for (int j = 0; j < 4; ++j)
                out[j] = p[j];
        }
    }
    return best > 16;
}

__device__ bool refine_quad(Image im, float2* p, int radius) {
    Line lines[4];
    for (int j = 0; j < 4; ++j)
        if (!fit_edge(im, p[j], p[(j + 1) % 4], radius, lines[j]))
            return false;
    for (int j = 0; j < 4; ++j) {
        const Line& a = lines[(j + 3) % 4];
        const Line& b = lines[j];
        const float denom = cross(a.direction, b.direction);
        if (fabsf(denom) < .12f)
            return false;
        const float t = cross(sub(b.center, a.center), b.direction) / denom;
        const float2 point =
                make_float2(a.center.x + t * a.direction.x, a.center.y + t * a.direction.y);
        if (length(sub(point, p[j])) > 3.f * radius)
            return false;
        p[j] = point;
    }
    return true;
}

struct Projection {
    float a, b, c, d, e, f, g, h;
    __device__ float sample(Image im, float u, float v) const {
        const float z = 1 + g * u + h * v;
        return bilinear(im, (a * u + b * v + c) / z, (d * u + e * v + f) / z);
    }
};

__device__ bool project(const float2* p, Projection& m) {
    const float2 v1 = sub(p[1], p[2]), v2 = sub(p[3], p[2]);
    const float2 diff = sub(sub(p[0], p[1]), v2);
    const float det = cross(v1, v2);
    if (fabsf(det) < 1.e-5f)
        return false;
    m.g = cross(diff, v2) / det;
    m.h = cross(v1, diff) / det;
    m.a = p[1].x * (1 + m.g) - p[0].x;
    m.b = p[3].x * (1 + m.h) - p[0].x;
    m.c = p[0].x;
    m.d = p[1].y * (1 + m.g) - p[0].y;
    m.e = p[3].y * (1 + m.h) - p[0].y;
    m.f = p[0].y;
    return true;
}

// Estimate independent black and white luminance planes. A tag at the image
// boundary can have part of its outer white sampling ring outside the frame.
struct GrayPlane {
    float gray = 0, u = 0, v = 0, ug = 0, vg = 0, uu = 0, uv = 0, vv = 0;
    int count = 0;
    __device__ void add(float x, float y, float level) {
        gray += level;
        u += x;
        v += y;
        ug += x * level;
        vg += y * level;
        uu += x * x;
        uv += x * y;
        vv += y * y;
        ++count;
    }
    __device__ bool fit() {
        if (count < 3)
            return false;
        gray /= count;
        u /= count;
        v /= count;
        ug -= count * u * gray;
        vg -= count * v * gray;
        uu -= count * u * u;
        uv -= count * u * v;
        vv -= count * v * v;
        const float det = uu * vv - uv * uv;
        if (det <= 1.e-5f)
            return false;
        const float gx = (ug * vv - vg * uv) / det;
        vg = (vg * uu - ug * uv) / det;
        ug = gx;
        gray -= ug * u + vg * v;
        return true;
    }
    __device__ float at(float x, float y) const { return gray + x * ug + y * vg; }
};
__device__ bool decode_bits(Image im, const Projection& m, int side, int contrast, Word& bits) {
    const int cells = side + 2;
    const float inv = 1.f / cells;
    GrayPlane planes[2];
    for (int k = 0; k < 2; ++k) {
        const float edge = (k ? -.5f : .5f) * inv;
        for (int j = 0; j < cells; ++j) {
            const float t = (j + .5f) * inv;
            const float xs[4] = {t, t, edge, 1 - edge}, ys[4] = {edge, 1 - edge, t, t};
            for (int z = 0; z < 4; ++z) {
                const float gray = m.sample(im, xs[z], ys[z]);
                if (gray < 0)
                    continue;
                planes[k].add(xs[z] - .5f, ys[z] - .5f, gray);
            }
        }
    }
    if (!planes[0].fit() || !planes[1].fit() || planes[1].at(0, 0) - planes[0].at(0, 0) < contrast)
        return false;
    bits = 0;
    for (int row = 0; row < side; ++row)
        for (int col = 0; col < side; ++col) {
            const float u = (col + 1.5f) * inv, v = (row + 1.5f) * inv;
            const float threshold =
                    .5f * (planes[0].at(u - .5f, v - .5f) + planes[1].at(u - .5f, v - .5f));
            const float gray = m.sample(im, u, v);
            if (gray < 0)
                return false;
            bits |= Word(gray > threshold) << (row * side + col);
        }
    return true;
}

// After decoding gives the candidate a known scale, search for a coherent
// black-to-white edge across its central 86%. One lane examines each normal.
// The winning line maximizes support across normals, so isolated clutter,
// blur and a displaced corner have little influence on the two endpoints.
__device__ float border_gradient(Image im, float2 p, float2 n, float off) {
    const float dark = bilinear(im, p.x + (off - 1.f) * n.x, p.y + (off - 1.f) * n.y);
    const float bright = bilinear(im, p.x + (off + 1.f) * n.x, p.y + (off + 1.f) * n.y);
    return dark < 0 || bright < 0 ? 0.f : fmaxf(0.f, bright - dark);
}
__device__ Line decoded_edge(Image im, float2 a, float2 b, int cells, float radius_cells) {
    const int lane = threadIdx.x & 31;
    const float2 d = sub(b, a);
    const float len = length(d);
    const float2 dir = make_float2(d.x / len, d.y / len), n = make_float2(dir.y, -dir.x);
    const float t = .07f + (.86f / 31) * lane;
    const float2 p = make_float2(a.x + t * d.x, a.y + t * d.y);
    const int radius = min(48, max(2, int(ceilf(radius_cells * len / cells))));
    float peak = 0, displace = 0;
    for (int k = -radius; k <= radius; ++k) {
        const float g = border_gradient(im, p, n, k);
        const float weight = g * (1.f - .15f * abs(k) / radius);
        if (weight > peak) {
            peak = weight;
            displace = k;
        }
    }
    float sum = 0, sum_pos = 0;
    for (int k = -4; k <= 4; ++k) {
        const float v = displace + .5f * k, g = border_gradient(im, p, n, v);
        sum += g * g;
        sum_pos += v * g * g;
    }
    if (sum > 0)
        displace = sum_pos / sum;
    // Opposite-half pairs provide deterministic line hypotheses. Rank them
    // by the strength of other samples near the predicted offset.
    const int other = lane ^ 16;
    const float t2 = __shfl_sync(0xffffffffU, t, other);
    const float v2 = __shfl_sync(0xffffffffU, displace, other);
    const float peak2 = __shfl_sync(0xffffffffU, peak, other);
    float slope = (v2 - displace) / (t2 - t), intercept = displace - t * slope;
    float score = 0;
    for (int k = 0; k < 32; ++k) {
        const float tk = __shfl_sync(0xffffffffU, t, k);
        const float vk = __shfl_sync(0xffffffffU, displace, k);
        const float wk = __shfl_sync(0xffffffffU, peak, k);
        score += wk * fmaxf(0.f, 1.f - fabsf(vk - intercept - slope * tk) / 2.5f);
    }
    if (peak < 1 || peak2 < 1)
        score = 0;
    float max_score = score;
    for (int off = 16; off; off /= 2)
        max_score = fmaxf(max_score, __shfl_xor_sync(0xffffffffU, max_score, off));
    const int winner = __ffs(__ballot_sync(0xffffffffU, score == max_score)) - 1;
    intercept = __shfl_sync(0xffffffffU, intercept, winner);
    slope = __shfl_sync(0xffffffffU, slope, winner);
    const float resid = displace - (intercept + slope * t);
    const float w = peak * fmaxf(0.f, 1.f - (resid * resid) / (2.5f * 2.5f));
    const float wt = warp_sum(w), s1 = warp_sum(w * t), s2 = warp_sum(w * t * t);
    const float v0 = warp_sum(w * displace), v1 = warp_sum(w * displace * t);
    const float det = wt * s2 - s1 * s1;
    if (wt < 32 || det < .018f * wt * wt)
        return Line{a, dir};
    slope = (wt * v1 - s1 * v0) / det;
    intercept = (v0 - slope * s1) / wt;
    const float2 v = make_float2(d.x + slope * n.x, d.y + slope * n.y);
    const float norm = length(v);
    return Line{make_float2(a.x + intercept * n.x, a.y + intercept * n.y),
                make_float2(v.x / norm, v.y / norm)};
}

__device__ void decoded_corners(Image im, float2* p, int side, Config cfg, MarkerCode code,
                                int& hamming) {
    if (cfg.refine_radius_cells_ == 0)
        return;
    Line lines[4];
    float2 fitted[4];
    for (int j = 0; j < 4; ++j)
        lines[j] = decoded_edge(im, p[j], p[(j + 1) % 4], side + 2, cfg.refine_radius_cells_);
    for (int j = 0; j < 4; ++j) {
        const Line& a = lines[(j + 3) % 4];
        const Line& b = lines[j];
        const float denom = cross(a.direction, b.direction);
        if (fabsf(denom) < .12f)
            return;
        const float t = cross(sub(b.center, a.center), b.direction) / denom;
        fitted[j] = make_float2(a.center.x + t * a.direction.x, a.center.y + t * a.direction.y);
        if (length(sub(fitted[j], p[j])) > .30f * length(sub(p[(j + 1) % 4], p[j])))
            return;
    }
    Projection m;
    Word word;
    if (!project(fitted, m) || !decode_bits(im, m, side, cfg.min_contrast_, word))
        return;
    const int distance = __popcll(word ^ code);
    if (distance > hamming)
        return;
    hamming = distance;
    for (int j = 0; j < 4; ++j)
        p[j] = fitted[j];
}

// Build a dense list so the expensive fit launches a bounded set of warps
// independent of hash capacity. No count travels through the host.
__global__ void collect(const Cluster* clusters, int capacity, int min_points, int* active,
                        Counts* counts) {
    const int i = linear_index();
    if (i < capacity && clusters[i].count >= min_points)
        active[atomicAdd(&counts->active, 1)] = i;
}
__device__ void identify_one(Image im, const Cluster& cluster, Config config,
                             const MarkerCode* codes, int code_count, int side, DeviceTag* tags,
                             Counts* counts) {
    float2 p[4];
    Projection m;
    Word bits;
    if (!quad_from_cluster(cluster, config.decimate_, p) ||
        !refine_quad(im, p, config.decimate_ + 1) || !project(p, m) ||
        !decode_bits(im, m, side, config.min_contrast_, bits))
        return;
    const int lane = threadIdx.x & 31;
    unsigned best = 0xffffffffU;
    for (int j = lane; j < 4 * code_count; j += 32) {
        const unsigned d = __popcll(bits ^ codes[j]);
        if (d <= unsigned(config.max_hamming_ + 1))
            best = min(best, (d << 20) | unsigned(j));
    }
    for (int offset = 16; offset; offset /= 2)
        best = min(best, __shfl_down_sync(0xffffffffU, best, offset));
    best = __shfl_sync(0xffffffffU, best, 0);
    if (best == 0xffffffffU)
        return;
    const int match = best & 0xfffff;
    int hamming = best >> 20;
    if (!lane && config.fine_samples_per_cell_ > 0) {
        float shortest = length(sub(p[0], p[3]));
        for (int j = 1; j < 4; ++j)
            shortest = fminf(shortest, length(sub(p[j], p[j - 1])));
        if (shortest < config.fine_samples_per_cell_ * config.decimate_ * (side + 2))
            atomicExch(&counts->fine, 1);
    }
    if (hamming > config.max_hamming_)
        return;
    decoded_corners(im, p, side, config, codes[match], hamming);
    if (lane)
        return;
    for (int j = 0; j < 4; ++j) {
        const auto a = sub(p[(j + 1) % 4], p[j]), b = sub(p[(j + 2) % 4], p[(j + 1) % 4]);
        if (cross(a, b) < config.min_pixels_per_cell_ * (side + 2) * fmaxf(length(a), length(b)))
            return;
    }
    const int idx = atomicAdd(&counts->tags, 1);
    if (idx >= config.max_tags_) {
        atomicExch(&counts->overflow, 1);
        return;
    }
    DeviceTag& tag = tags[idx];
    tag.id = match / 4;
    tag.hamming = hamming;
    tag.support = cluster.count;
    for (int j = 0; j < 4; ++j) {
        const float2 pt = p[(j + 4 - (match % 4)) % 4];
        tag.xy[2 * j] = pt.x;
        tag.xy[2 * j + 1] = pt.y;
    }
}

__global__ void identify(Image im, const Cluster* clusters, const int* active, Config config,
                         const MarkerCode* codes, int code_count, int side, DeviceTag* tags,
                         Counts* counts) {
    const int warp = linear_index() / 32;
    for (int i = warp; i < counts->active; i += gridDim.x * blockDim.x / 32)
        identify_one(im, clusters[active[i]], config, codes, code_count, side, tags, counts);
}

// Only merge same-ID quads with corresponding oriented corners close on the
// scale of their shortest side. Nearby instances of the same ID remain separate.
bool duplicate_tag(const DeviceTag& a, const Tag& b, float ratio) {
    float shortest_sq = std::numeric_limits<float>::max(), furthest_sq = 0;
    for (int j = 0; j < 4; ++j) {
        int k = (j + 1) % 4;
        float dx = a.xy[2 * j] - a.xy[2 * k], dy = a.xy[2 * j + 1] - a.xy[2 * k + 1];
        shortest_sq = std::min(shortest_sq, dx * dx + dy * dy);
        dx = a.xy[2 * j] - b.corners_[2 * j];
        dy = a.xy[2 * j + 1] - b.corners_[2 * j + 1];
        furthest_sq = std::max(furthest_sq, dx * dx + dy * dy);
    }
    return furthest_sq <= ratio * ratio * shortest_sq;
}
struct Layout {
    int step;
    Size seg, tiles;
    int capacity;

    Layout(Size size, int sample_step, int slots)
        : step(sample_step),
          seg{(size.width + step - 1) / step, (size.height + step - 1) / step},
          tiles{(seg.width + 3) / 4, (seg.height + 3) / 4},
          capacity(slots) {
        if (capacity == 0) {
            const int target = std::min((seg.width * seg.height + 3) / 4, 1 << 24);
            capacity = 16384;
            while (capacity < target)
                capacity *= 2;
        }
    }
};
}  // namespace

struct Detector::Impl {
    Config config;
    Size full;
    Layout coarse, fine;
    int side, code_count;
    Buffer<uchar2> range;
    Buffer<unsigned char> colors;
    Buffer<int> parents;
    Buffer<int> roots;
    Buffer<Cluster> clusters;
    Buffer<int> active;
    Buffer<MarkerCode> codes;
    HostTags tags;
    Buffer<Counts> counts;
    Impl(const DictionaryTable& table, Size size, Config cfg)
        : config(cfg),
          full(size),
          coarse(size,
                 cfg.decimate_ ? cfg.decimate_
                               : (size.width * size.height <= cfg.small_frame_pixels_ ? 1 : 3),
                 cfg.cluster_slots_),
          fine(size, cfg.decimate_ ? cfg.decimate_ : 1, cfg.cluster_slots_),
          side(table.marker_size_),
          code_count(table.code_count_),
          range(fine.tiles.width * fine.tiles.height),
          colors(fine.seg.width * fine.seg.height),
          parents(fine.seg.width * fine.seg.height),
          roots(fine.seg.width * fine.seg.height),
          clusters(fine.capacity),
          active(fine.capacity),
          codes(4 * table.code_count_),
          tags(cfg.max_tags_),
          counts(1) {
        check(cudaMemcpy(codes.get(), table.codes_, 4 * code_count * sizeof(MarkerCode),
                         cudaMemcpyHostToDevice),
              "AprilTag dictionary copy");
        check(cudaStreamSynchronize(nullptr), "AprilTag dictionary initialization");
    }
};

Detector::Detector(const DictionaryTable& table, int width, int height, Config config) {
    if (config.max_hamming_ == -1) {
        config.max_hamming_ =
                (table.name_ && std::strcmp(table.name_, "DICT_APRILTAG_36h11") == 0) ? 2 : 0;
    }
    if (width < 1 || height < 1 || width > 16384 || height > 16384 || !table.codes_ ||
        table.marker_size_ < 1 || table.marker_size_ > 7 || table.code_count_ < 1 ||
        table.code_count_ > 65536 || config.decimate_ < 0 || config.decimate_ > 4 ||
        config.max_cluster_probes_ < 1 || config.max_cluster_probes_ > 4096 ||
        config.small_frame_pixels_ < 0 ||
        !(config.min_pixels_per_cell_ >= 0 && config.min_pixels_per_cell_ <= 8) ||
        !(config.fine_samples_per_cell_ > 0 && config.fine_samples_per_cell_ <= 8) ||
        !(config.refine_radius_cells_ >= 0 && config.refine_radius_cells_ <= 2) ||
        config.min_contrast_ < 1 || config.min_contrast_ > 255 || config.min_cluster_points_ < 4 ||
        config.max_hamming_ < 0 || config.max_hamming_ > 2 ||
        !(config.duplicate_corner_ratio_ >= 0 && config.duplicate_corner_ratio_ < .5f) ||
        config.max_tags_ < 1 || config.max_tags_ > 65536 || config.cluster_slots_ < 0 ||
        config.cluster_slots_ > (1 << 24) ||
        (config.cluster_slots_ & (config.cluster_slots_ - 1))) {
        throw std::invalid_argument(
                "invalid AprilTag dimensions, dictionary, or detector configuration");
    }
    this->impl_ = std::make_unique<Impl>(table, Size{width, height}, config);
}
Detector::~Detector() = default;

std::vector<Tag> Detector::detect(const std::uint8_t* data, std::size_t pitch,
                                  cudaStream_t stream) {
    auto& s = *this->impl_;
    if (!data || pitch < std::size_t(s.full.width) ||
        pitch > std::numeric_limits<std::size_t>::max() / s.full.height)
        throw std::invalid_argument("invalid AprilTag grayscale image view");
    Image im{data, pitch, s.full.width, s.full.height};
    auto scan = [&](const Layout& layout, bool second) {
        const auto seg = layout.seg, tiles = layout.tiles;
        const int n = seg.width * seg.height;
        auto config = s.config;
        config.decimate_ = layout.step;
        if (s.config.decimate_ != 0 || layout.step == 1)
            config.fine_samples_per_cell_ = 0;
        Counts counts{};
        check(cudaMemsetAsync(s.clusters.get(), 0, layout.capacity * sizeof(Cluster), stream),
              "clear AprilTag clusters");
        if (second) {
            // Append fine decodes to the same tag array. Clear only the active cluster list length.
            check(cudaMemsetAsync(&s.counts.get()->active, 0, sizeof(int), stream),
                  "clear AprilTag active count");
        } else {
            check(cudaMemsetAsync(s.counts.get(), 0, sizeof(Counts), stream),
                  "clear AprilTag counts");
        }
        tile_range<<<blocks(tiles.width * tiles.height), kThreads, 0, stream>>>(
                im, seg, tiles, layout.step, s.range.get());
        check(cudaGetLastError(), "AprilTag tile range launch");
        classify<<<blocks(n), kThreads, 0, stream>>>(im, seg, tiles, s.range.get(), layout.step,
                                                     config.min_contrast_, s.colors.get());
        check(cudaGetLastError(), "AprilTag threshold launch");
        seed_runs<<<blocks(n), kThreads, 0, stream>>>(seg, s.colors.get(), s.parents.get());
        check(cudaGetLastError(), "AprilTag row runs launch");
        shorten<<<blocks(n), kThreads, 0, stream>>>(s.parents.get(), s.roots.get(), n);
        check(cudaGetLastError(), "AprilTag row compression launch");
        merge_rows<<<blocks(n), kThreads, 0, stream>>>(seg, s.colors.get(), s.roots.get());
        check(cudaGetLastError(), "AprilTag components launch");
        shorten<<<blocks(n), kThreads, 0, stream>>>(s.roots.get(), s.parents.get(), n);
        check(cudaGetLastError(), "AprilTag component compression launch");
        cluster_edges<<<blocks(n), kThreads, 0, stream>>>(
                seg, s.colors.get(), s.parents.get(), s.clusters.get(), layout.capacity,
                std::min(layout.capacity, config.max_cluster_probes_), s.counts.get());
        check(cudaGetLastError(), "AprilTag cluster launch");
        collect<<<blocks(layout.capacity), kThreads, 0, stream>>>(s.clusters.get(), layout.capacity,
                                                                  s.config.min_cluster_points_,
                                                                  s.active.get(), s.counts.get());
        check(cudaGetLastError(), "AprilTag active clusters launch");
        identify<<<std::min(128, blocks(layout.capacity)), kThreads, 0, stream>>>(
                im, s.clusters.get(), s.active.get(), config, s.codes.get(), s.code_count, s.side,
                s.tags.device, s.counts.get());
        check(cudaGetLastError(), "AprilTag decode launch");
        check(cudaMemcpyAsync(&counts, s.counts.get(), sizeof(Counts), cudaMemcpyDeviceToHost,
                              stream),
              "AprilTag count copy");
        check(cudaStreamSynchronize(stream), "AprilTag pipeline completion");
        if (counts.overflow)
            throw std::overflow_error("AprilTag hash probe budget or detection capacity exceeded");
        return counts;
    };
    try {
        auto counts = scan(s.coarse, false);
        if (s.config.decimate_ == 0 && counts.fine)
            counts = scan(s.fine, true);
        std::vector<Tag> result;
        result.reserve(counts.tags);
        if (counts.tags) {
            std::sort(s.tags.host, s.tags.host + counts.tags,
                      [](const DeviceTag& a, const DeviceTag& b) {
                          if (a.id != b.id)
                              return a.id < b.id;
                          if (a.hamming != b.hamming)
                              return a.hamming < b.hamming;
                          if (a.support != b.support)
                              return a.support > b.support;
                          return std::lexicographical_compare(a.xy, a.xy + 8, b.xy, b.xy + 8);
                      });
            for (int i = 0; i < counts.tags; ++i) {
                const auto& current = s.tags.host[i];
                auto end = result.rbegin();
                while (end != result.rend() && end->id_ == current.id) {
                    if (duplicate_tag(current, *end, s.config.duplicate_corner_ratio_))
                        break;
                    ++end;
                }
                if (end != result.rend() && end->id_ == current.id)
                    continue;
                Tag tag{};
                tag.id_ = current.id;
                tag.hamming_ = current.hamming;
                std::copy_n(current.xy, 8, tag.corners_.begin());
                result.push_back(tag);
            }
        }
        return result;
    } catch (...) {
        cudaStreamSynchronize(stream);
        throw;
    }
}
}  // namespace aruco3cuda::apriltag_gpu
