// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_UTIL_SHA256_HPP
#define ARUCO3CUDA_UTIL_SHA256_HPP

#include <cstddef>
#include <cstdint>
#include <string>

namespace aruco3cuda::util {

/// Computes SHA-256 incrementally. An original implementation based on the FIPS 180-4
/// specification.
///
/// Intended use:
///   A checksum that ties the input images used for an evaluation to the resulting JSON.
///   It is meant for confirming identity, not for cryptographic purposes.
///
/// Ownership:
///   The instance owns its internal state, which is released in the destructor. Memory
///   taken as an argument is only read, never copied or retained.
///   **This ownership applies to all public member functions below.**
///
/// Synchronization:
///   Host only, with no synchronization point. It carries in-progress computation state,
///   so a single instance must not be used from several threads at once.
///   **This synchronization applies to all public member functions below.**
///
/// Example input:
///   Sha256 hasher;
///   hasher.update("abc", 3);
///   const std::string digest = hasher.finalize();
/// Example output:
///   digest == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
class Sha256 {
public:
    /// Constructs the object with the initial hash values set.
    ///
    /// Synchronization: host only, with no synchronization point.
    /// Ownership: the instance owns its internal buffer, which is released in the
    ///            destructor.
    Sha256();

    /// Consumes the first size bytes of data. May be called repeatedly.
    ///
    /// Feeding the input in pieces gives the same result as feeding it all at once.
    ///
    /// @param data The start of the memory to consume. May be nullptr, in which case
    ///             nothing happens. The caller owns the memory, and this function
    ///             neither copies nor retains it.
    /// @param size The number of bytes to consume. May be 0, in which case nothing
    ///             happens.
    /// @return Nothing.
    ///
    /// Example input: update("ab", 2) followed by update("c", 1)
    /// Example output: finalize() returns the hash of "abc"
    void update(const void* data, std::size_t size);

    /// Returns the hash of everything consumed.
    ///
    /// Padding is applied and the computation is finalized, so the same instance must
    /// not be reused after this call. Reusing it returns an incorrect value.
    ///
    /// @return A 64-character lowercase hexadecimal string.
    ///
    /// Example input: called after update("abc", 3)
    /// Example output: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    std::string finalize();

private:
    void process_block(const std::uint8_t* block);

    std::uint32_t state_[8];
    std::uint64_t total_bits_;
    std::uint8_t buffer_[64];
    std::size_t buffer_size_;
};

/// Computes the SHA-256 of an entire file.
///
/// @param path The file to read.
/// @param out_hex On success, receives the 64-character hexadecimal string.
/// @return true if the file could be read. false if it could not be opened, in which
///         case out_hex is left unchanged.
bool sha256_file(const std::string& path, std::string* out_hex);

/// Computes the SHA-256 of a byte sequence in memory.
///
/// @param data The start of the memory. May be nullptr, in which case it is treated as
///             empty input. The caller owns the memory, and this function neither copies
///             nor retains it.
/// @param size The number of bytes.
/// @return A 64-character lowercase hexadecimal string.
///
/// Synchronization: host only, with no synchronization point.
///
/// Example input: the 3 bytes of "abc"
/// Example output: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
std::string sha256_bytes(const void* data, std::size_t size);

}  // namespace aruco3cuda::util

#endif  // ARUCO3CUDA_UTIL_SHA256_HPP
