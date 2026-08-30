// SPDX-License-Identifier: Apache-2.0
//
// Implements SHA-256 from the FIPS 180-4 specification. It is not a copy of any other
// implementation.
#include "aruco3cuda/util/sha256.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <ios>
#include <string>

namespace aruco3cuda::util {
namespace {

constexpr std::size_t kBlockBytes = 64;

/// Round constants from FIPS 180-4, derived from the fractional parts of the cube roots of the
/// first 64 primes.
constexpr std::uint32_t kRoundConstants[64] = {
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U,
        0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU,
        0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU,
        0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
        0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
        0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
        0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U,
        0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
        0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U,
        0xc67178f2U};

inline std::uint32_t rotate_right(std::uint32_t value, unsigned int bits) {
    return (value >> bits) | (value << (32U - bits));
}

inline std::uint32_t load_big_endian(const std::uint8_t* p) {
    return (static_cast<std::uint32_t>(p[0]) << 24) | (static_cast<std::uint32_t>(p[1]) << 16) |
           (static_cast<std::uint32_t>(p[2]) << 8) | static_cast<std::uint32_t>(p[3]);
}

inline void store_big_endian(std::uint8_t* p, std::uint32_t value) {
    p[0] = static_cast<std::uint8_t>(value >> 24);
    p[1] = static_cast<std::uint8_t>(value >> 16);
    p[2] = static_cast<std::uint8_t>(value >> 8);
    p[3] = static_cast<std::uint8_t>(value);
}

}  // namespace

Sha256::Sha256() : total_bits_(0), buffer_size_(0) {
    // Initial hash values from FIPS 180-4, derived from the fractional parts of the square roots
    // of the first 8 primes.
    this->state_[0] = 0x6a09e667U;
    this->state_[1] = 0xbb67ae85U;
    this->state_[2] = 0x3c6ef372U;
    this->state_[3] = 0xa54ff53aU;
    this->state_[4] = 0x510e527fU;
    this->state_[5] = 0x9b05688cU;
    this->state_[6] = 0x1f83d9abU;
    this->state_[7] = 0x5be0cd19U;
    std::memset(this->buffer_, 0, sizeof(this->buffer_));
}

void Sha256::process_block(const std::uint8_t* block) {
    std::uint32_t w[64];
    for (std::size_t i = 0; i < 16; ++i) {
        w[i] = load_big_endian(block + i * 4);
    }
    for (std::size_t i = 16; i < 64; ++i) {
        const std::uint32_t s0 =
                rotate_right(w[i - 15], 7) ^ rotate_right(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const std::uint32_t s1 =
                rotate_right(w[i - 2], 17) ^ rotate_right(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    std::uint32_t a = this->state_[0];
    std::uint32_t b = this->state_[1];
    std::uint32_t c = this->state_[2];
    std::uint32_t d = this->state_[3];
    std::uint32_t e = this->state_[4];
    std::uint32_t f = this->state_[5];
    std::uint32_t g = this->state_[6];
    std::uint32_t h = this->state_[7];

    for (std::size_t i = 0; i < 64; ++i) {
        const std::uint32_t s1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
        const std::uint32_t ch = (e & f) ^ ((~e) & g);
        const std::uint32_t temp1 = h + s1 + ch + kRoundConstants[i] + w[i];
        const std::uint32_t s0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
        const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        const std::uint32_t temp2 = s0 + maj;

        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    this->state_[0] += a;
    this->state_[1] += b;
    this->state_[2] += c;
    this->state_[3] += d;
    this->state_[4] += e;
    this->state_[5] += f;
    this->state_[6] += g;
    this->state_[7] += h;
}

void Sha256::update(const void* data, std::size_t size) {
    if (data == nullptr || size == 0) {
        return;
    }
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    this->total_bits_ += static_cast<std::uint64_t>(size) * 8U;

    // Combine with the leftover from the previous call and process in 64-byte units.
    if (this->buffer_size_ > 0) {
        const std::size_t need = kBlockBytes - this->buffer_size_;
        const std::size_t take = (size < need) ? size : need;
        std::memcpy(this->buffer_ + this->buffer_size_, bytes, take);
        this->buffer_size_ += take;
        bytes += take;
        size -= take;
        if (this->buffer_size_ == kBlockBytes) {
            this->process_block(this->buffer_);
            this->buffer_size_ = 0;
        }
    }
    while (size >= kBlockBytes) {
        this->process_block(bytes);
        bytes += kBlockBytes;
        size -= kBlockBytes;
    }
    if (size > 0) {
        std::memcpy(this->buffer_, bytes, size);
        this->buffer_size_ = size;
    }
}

std::string Sha256::finalize() {
    // Padding: an 0x80 byte followed by zeros, with the total bit count placed big endian in the
    // final 8 bytes.
    const std::uint64_t total_bits = this->total_bits_;
    const std::uint8_t one = 0x80U;
    this->update(&one, 1);
    this->total_bits_ = total_bits;  // the padding itself is not part of the length

    const std::uint8_t zero = 0x00U;
    while (this->buffer_size_ != 56) {
        this->update(&zero, 1);
        this->total_bits_ = total_bits;
    }

    std::uint8_t length_bytes[8];
    for (std::size_t i = 0; i < 8; ++i) {
        length_bytes[i] = static_cast<std::uint8_t>(total_bits >> (56U - i * 8U));
    }
    std::memcpy(this->buffer_ + 56, length_bytes, 8);
    this->process_block(this->buffer_);
    this->buffer_size_ = 0;

    std::uint8_t digest[32];
    for (std::size_t i = 0; i < 8; ++i) {
        store_big_endian(digest + i * 4, this->state_[i]);
    }

    std::string hex(64, '0');
    for (std::size_t i = 0; i < 32; ++i) {
        std::snprintf(&hex[i * 2], 3, "%02x", digest[i]);
    }
    return hex;
}

std::string sha256_bytes(const void* data, std::size_t size) {
    Sha256 hasher;
    hasher.update(data, size);
    return hasher.finalize();
}

bool sha256_file(const std::string& path, std::string* out_hex) {
    if (out_hex == nullptr) {
        return false;
    }
    // ifstream rather than FILE* for RAII; the destructor performs the close.
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return false;
    }
    Sha256 hasher;
    char chunk[16384];
    while (input.read(chunk, sizeof(chunk)) || input.gcount() > 0) {
        hasher.update(chunk, static_cast<std::size_t>(input.gcount()));
        if (!input) {
            break;
        }
    }
    if (input.bad()) {
        return false;
    }
    *out_hex = hasher.finalize();
    return true;
}

}  // namespace aruco3cuda::util
