// SPDX-License-Identifier: Apache-2.0
//
// SHA-256 を FIPS 180-4 の公開 test vector で検証する。
#include "aruco3cuda/util/sha256.hpp"

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <ios>
#include <string>

namespace {

std::string hash_text(const std::string& text) {
    return aruco3cuda::util::sha256_bytes(text.data(), text.size());
}

// 正常系: 公開されている既知の値と一致する。
TEST(Sha256Test, matches_published_vectors) {
    EXPECT_EQ(hash_text(""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    EXPECT_EQ(hash_text("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    EXPECT_EQ(hash_text("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
              "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
}

// 境界値: block 境界の前後で正しく padding される。
TEST(Sha256Test, handles_block_boundary_lengths) {
    // 55, 56, 57, 63, 64, 65 byte は padding 経路が分岐する長さ。
    const int lengths[] = {55, 56, 57, 63, 64, 65};
    for (const int length : lengths) {
        const std::string input(static_cast<std::size_t>(length), 'a');
        const std::string digest = hash_text(input);
        EXPECT_EQ(digest.size(), 64U) << "length=" << length;
        EXPECT_NE(digest, hash_text(input + "a")) << "length=" << length;
    }
}

// 正常系: 分割して update しても結果が変わらない。
TEST(Sha256Test, incremental_update_matches_single_update) {
    const std::string text = "The quick brown fox jumps over the lazy dog";
    aruco3cuda::util::Sha256 hasher;
    hasher.update(text.data(), 10);
    hasher.update(text.data() + 10, text.size() - 10);
    EXPECT_EQ(hasher.finalize(), hash_text(text));
}

// 異常系: 存在しない file では false を返し出力を変更しない。
TEST(Sha256Test, file_hash_reports_failure_for_missing_file) {
    std::string digest = "unchanged";
    EXPECT_FALSE(aruco3cuda::util::sha256_file("/nonexistent/path/xyz", &digest));
    EXPECT_EQ(digest, "unchanged");
}

// 異常系: 出力先が nullptr の場合も安全に失敗する。
TEST(Sha256Test, file_hash_rejects_null_output) {
    EXPECT_FALSE(aruco3cuda::util::sha256_file("/etc/hostname", nullptr));
}

// 正常系: file の内容と memory 上の内容で同じ結果になる。
TEST(Sha256Test, file_hash_matches_memory_hash) {
    const std::string path = "/tmp/aruco3cuda_sha256_test.bin";
    const std::string content = "aruco3-cuda reference corpus";
    {
        std::ofstream output(path, std::ios::binary);
        ASSERT_TRUE(output.is_open());
        output.write(content.data(), static_cast<std::streamsize>(content.size()));
    }

    std::string digest;
    ASSERT_TRUE(aruco3cuda::util::sha256_file(path, &digest));
    EXPECT_EQ(digest, hash_text(content));
    std::remove(path.c_str());
}

}  // namespace
