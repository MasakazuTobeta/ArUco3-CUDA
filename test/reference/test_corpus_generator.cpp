// SPDX-License-Identifier: Apache-2.0
//
// 合成 corpus 生成器を検証する。
//
// 重要な性質は 2 つある。
//   1. seed を固定すれば同じ画像が再生成できる。
//   2. manifest の四隅が真値であり、CPU 基準実装の検出結果と一致する。
// 2 は corpus 生成器と CPU 基準 runner を相互に検証することにもなる。
#include "corpus_generator.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#include "reference_runner.hpp"

namespace {

using aruco3cuda::corpusgen::CorpusConfig;
using aruco3cuda::corpusgen::GeneratedScene;
using aruco3cuda::corpusgen::SceneSpec;

class CorpusGeneratorTest : public ::testing::Test {
protected:
    void SetUp() override {
        this->config_.output_dir_ = "/tmp/aruco3cuda_corpus_test";
        this->config_.seed_ = 12345U;
    }
    void TearDown() override {
        std::error_code error;
        std::filesystem::remove_all(this->config_.output_dir_, error);
    }

    CorpusConfig config_;
};

SceneSpec clean_spec(const std::string& name) {
    SceneSpec spec;
    spec.name_ = name;
    spec.width_px_ = 1280;
    spec.height_px_ = 720;
    spec.marker_count_ = 4;
    spec.marker_side_px_ = 120.0;
    return spec;
}

// 正常系: preset が既知の名前でのみ成功する。
TEST(CorpusPresetTest, builds_known_presets_only) {
    std::vector<SceneSpec> specs;
    for (const std::string& name : aruco3cuda::corpusgen::known_presets()) {
        EXPECT_TRUE(aruco3cuda::corpusgen::build_preset(name, &specs)) << name;
        EXPECT_FALSE(specs.empty()) << name;
    }
    EXPECT_FALSE(aruco3cuda::corpusgen::build_preset("does_not_exist", &specs));
    EXPECT_FALSE(aruco3cuda::corpusgen::build_preset("smoke", nullptr));
}

// 正常系: preset の scene 名が重複しない。重複すると file が上書きされる。
TEST(CorpusPresetTest, scene_names_are_unique_within_preset) {
    for (const std::string& name : aruco3cuda::corpusgen::known_presets()) {
        std::vector<SceneSpec> specs;
        ASSERT_TRUE(aruco3cuda::corpusgen::build_preset(name, &specs));
        std::vector<std::string> names;
        names.reserve(specs.size());
        for (const SceneSpec& spec : specs) {
            names.push_back(spec.name_);
        }
        std::sort(names.begin(), names.end());
        EXPECT_EQ(std::adjacent_find(names.begin(), names.end()), names.end())
                << "preset=" << name << " に重複した scene 名がある";
    }
}

// 正常系: 同じ seed からは byte 単位で同じ画像が得られる。
TEST_F(CorpusGeneratorTest, same_seed_reproduces_identical_image) {
    const SceneSpec spec = clean_spec("determinism");
    GeneratedScene first;
    GeneratedScene second;
    std::string error;
    ASSERT_TRUE(aruco3cuda::corpusgen::generate_scene(this->config_, spec, 0, &first, &error))
            << error;
    ASSERT_TRUE(aruco3cuda::corpusgen::generate_scene(this->config_, spec, 0, &second, &error))
            << error;
    EXPECT_EQ(first.sha256_, second.sha256_);
    ASSERT_EQ(first.markers_.size(), second.markers_.size());
    for (std::size_t i = 0; i < first.markers_.size(); ++i) {
        EXPECT_EQ(first.markers_[i].id_, second.markers_[i].id_);
        EXPECT_EQ(first.markers_[i].corners_, second.markers_[i].corners_);
    }
}

// 正常系: seed が違えば内容が変わる。
TEST_F(CorpusGeneratorTest, different_seed_changes_content) {
    const SceneSpec spec = clean_spec("seed_variation");
    GeneratedScene first;
    std::string error;
    ASSERT_TRUE(aruco3cuda::corpusgen::generate_scene(this->config_, spec, 0, &first, &error))
            << error;

    CorpusConfig other = this->config_;
    other.seed_ = this->config_.seed_ + 1U;
    GeneratedScene second;
    ASSERT_TRUE(aruco3cuda::corpusgen::generate_scene(other, spec, 0, &second, &error)) << error;
    EXPECT_NE(first.sha256_, second.sha256_);
}

// 境界値: scene_index が違えば独立した乱数列になる。
// scene を追加しても既存 scene が変わらないことの裏付けになる。
TEST_F(CorpusGeneratorTest, scene_index_produces_independent_randomness) {
    SceneSpec spec = clean_spec("index_a");
    GeneratedScene first;
    std::string error;
    ASSERT_TRUE(aruco3cuda::corpusgen::generate_scene(this->config_, spec, 0, &first, &error))
            << error;
    spec.name_ = "index_b";
    GeneratedScene second;
    ASSERT_TRUE(aruco3cuda::corpusgen::generate_scene(this->config_, spec, 1, &second, &error))
            << error;
    EXPECT_NE(first.sha256_, second.sha256_);
}

// 正常系: manifest の四隅が CPU 基準実装の検出結果と一致する。
// corpus 生成器と CPU 基準 runner の相互検証になる。
TEST_F(CorpusGeneratorTest, ground_truth_corners_match_cpu_detection) {
    const SceneSpec spec = clean_spec("ground_truth");
    GeneratedScene scene;
    std::string error;
    ASSERT_TRUE(aruco3cuda::corpusgen::generate_scene(this->config_, spec, 0, &scene, &error))
            << error;
    ASSERT_EQ(scene.markers_.size(), 4U);

    aruco3cuda::reference::ReferenceConfig detector_config;
    aruco3cuda::reference::ReferenceResult result;
    ASSERT_TRUE(aruco3cuda::reference::detect_image(scene.path_, detector_config, &result, &error))
            << error;
    ASSERT_EQ(result.detections_.size(), scene.markers_.size());

    // ground truth を ID で引けるようにする。preset は ID の重複を避けていないため、
    // 同じ ID が複数ある場合は最も近い候補と対応付ける。
    for (const auto& detection : result.detections_) {
        double best_error = 1e9;
        for (const auto& truth : scene.markers_) {
            if (truth.id_ != detection.id_) {
                continue;
            }
            double worst_corner = 0.0;
            for (std::size_t c = 0; c < 4; ++c) {
                const double dx = detection.corners_[c * 2] - truth.corners_[c * 2];
                const double dy = detection.corners_[c * 2 + 1] - truth.corners_[c * 2 + 1];
                worst_corner = std::max(worst_corner, std::sqrt(dx * dx + dy * dy));
            }
            best_error = std::min(best_error, worst_corner);
        }
        // 劣化のない合成画像であり、subpixel 補正込みで 1.5 pixel 以内に収まる。
        EXPECT_LT(best_error, 1.5) << "id=" << detection.id_;
    }
}

// 正常系: side_ratio が manifest へ記録される。
TEST_F(CorpusGeneratorTest, records_side_ratio) {
    SceneSpec spec = clean_spec("side_ratio");
    spec.marker_side_px_ = 64.0;
    GeneratedScene scene;
    std::string error;
    ASSERT_TRUE(aruco3cuda::corpusgen::generate_scene(this->config_, spec, 0, &scene, &error))
            << error;
    ASSERT_FALSE(scene.markers_.empty());
    // 64 / 1280 = 0.05
    EXPECT_NEAR(scene.markers_[0].side_ratio_, 0.05, 1e-9);
}

// 正常系: 検出下限の式が導出どおりの値を返す。
// 下限は tau_i そのものではなく S + L * tau_i である。
TEST(CorpusThresholdTest, minimum_detectable_side_follows_derivation) {
    EXPECT_NEAR(aruco3cuda::corpusgen::minimum_detectable_side_px(32, 1280, 0.05), 96.0, 1e-9);
    EXPECT_NEAR(aruco3cuda::corpusgen::minimum_detectable_side_px(32, 1920, 0.05), 128.0, 1e-9);
    // tau_i = 0 では縮小が発生せず、下限は S に等しい。
    EXPECT_NEAR(aruco3cuda::corpusgen::minimum_detectable_side_px(32, 1280, 0.0), 32.0, 1e-9);
}

// 正常系: 境界にかかる配置では fully_inside_ が false になる。
TEST_F(CorpusGeneratorTest, marks_markers_outside_image_bounds) {
    SceneSpec spec = clean_spec("border_clip");
    spec.marker_count_ = 1;
    spec.marker_side_px_ = 200.0;
    spec.allow_border_clip_ = true;
    GeneratedScene scene;
    std::string error;
    ASSERT_TRUE(aruco3cuda::corpusgen::generate_scene(this->config_, spec, 0, &scene, &error))
            << error;
    ASSERT_EQ(scene.markers_.size(), 1U);
    EXPECT_FALSE(scene.markers_[0].fully_inside_);
}

// 境界値: マーカー 0 個でも生成でき、ground truth は空になる。
TEST_F(CorpusGeneratorTest, generates_scene_without_markers) {
    SceneSpec spec = clean_spec("empty");
    spec.marker_count_ = 0;
    GeneratedScene scene;
    std::string error;
    ASSERT_TRUE(aruco3cuda::corpusgen::generate_scene(this->config_, spec, 0, &scene, &error))
            << error;
    EXPECT_TRUE(scene.markers_.empty());

    aruco3cuda::reference::ReferenceConfig detector_config;
    aruco3cuda::reference::ReferenceResult result;
    ASSERT_TRUE(aruco3cuda::reference::detect_image(scene.path_, detector_config, &result, &error))
            << error;
    EXPECT_TRUE(result.detections_.empty());
}

// 異常系: 不正な spec と出力先を拒否する。
TEST_F(CorpusGeneratorTest, rejects_invalid_input) {
    GeneratedScene scene;
    std::string error;
    SceneSpec spec = clean_spec("invalid");
    spec.width_px_ = 0;
    EXPECT_FALSE(aruco3cuda::corpusgen::generate_scene(this->config_, spec, 0, &scene, &error));
    EXPECT_FALSE(error.empty());

    CorpusConfig bad_dictionary = this->config_;
    bad_dictionary.dictionary_name_ = "DICT_NOPE";
    EXPECT_FALSE(aruco3cuda::corpusgen::generate_scene(bad_dictionary, clean_spec("x"), 0, &scene,
                                                       &error));
    EXPECT_FALSE(aruco3cuda::corpusgen::generate_scene(this->config_, clean_spec("y"), 0, nullptr,
                                                       &error));
    EXPECT_FALSE(aruco3cuda::corpusgen::generate_scene(this->config_, clean_spec("z"), 0, &scene,
                                                       nullptr));
}

// 正常系: manifest が生成条件と ground truth を含む。
TEST_F(CorpusGeneratorTest, manifest_contains_conditions_and_ground_truth) {
    const SceneSpec spec = clean_spec("manifest");
    GeneratedScene scene;
    std::string error;
    ASSERT_TRUE(aruco3cuda::corpusgen::generate_scene(this->config_, spec, 0, &scene, &error))
            << error;

    std::ostringstream out;
    aruco3cuda::corpusgen::write_manifest_json(out, this->config_, "smoke", {scene});
    const std::string manifest = out.str();

    EXPECT_NE(manifest.find("\"schema_version\""), std::string::npos);
    EXPECT_NE(manifest.find("\"seed\": 12345"), std::string::npos);
    EXPECT_NE(manifest.find("\"ground_truth\""), std::string::npos);
    EXPECT_NE(manifest.find("\"marker_side_px\""), std::string::npos);
    EXPECT_NE(manifest.find(scene.sha256_), std::string::npos);
}

}  // namespace
