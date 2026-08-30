// SPDX-License-Identifier: Apache-2.0
//
// Verifies the synthetic corpus generator.
//
// Two properties matter:
//   1. Fixing the seed reproduces the same image.
//   2. The corners in the manifest are ground truth and agree with what the CPU
//      reference implementation detects.
// Property 2 also cross-validates the corpus generator and the CPU reference
// runner against each other.
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

#include <unistd.h>

#include "reference_runner.hpp"

namespace {

using aruco3cuda::corpusgen::CorpusConfig;
using aruco3cuda::corpusgen::GeneratedScene;
using aruco3cuda::corpusgen::SceneSpec;

class CorpusGeneratorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Give each test its own output directory. Under ctest -j several
        // tests run at once, and a shared directory would be wiped by another
        // test's TearDown.
        const ::testing::TestInfo* info = ::testing::UnitTest::GetInstance()->current_test_info();
        this->config_.output_dir_ = std::string("/tmp/aruco3cuda_corpus_test_") +
                                    (info != nullptr ? info->name() : "unknown") + "_" +
                                    std::to_string(static_cast<long>(::getpid()));
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

// Happy path: presets succeed only for known names.
TEST(CorpusPresetTest, builds_known_presets_only) {
    std::vector<SceneSpec> specs;
    for (const std::string& name : aruco3cuda::corpusgen::known_presets()) {
        EXPECT_TRUE(aruco3cuda::corpusgen::build_preset(name, &specs)) << name;
        EXPECT_FALSE(specs.empty()) << name;
    }
    EXPECT_FALSE(aruco3cuda::corpusgen::build_preset("does_not_exist", &specs));
    EXPECT_FALSE(aruco3cuda::corpusgen::build_preset("smoke", nullptr));
}

// Happy path: scene names within a preset are unique. Duplicates would
// overwrite each other's files.
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
                << "preset=" << name << " has duplicate scene names";
    }
}

// Happy path: the same seed yields a byte-for-byte identical image.
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

// Happy path: a different seed changes the content.
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

// Boundary: a different scene_index yields an independent random sequence.
// This backs up the claim that adding a scene does not change existing ones.
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

// Happy path: the corners in the manifest agree with what the CPU reference
// implementation detects. This cross-validates the corpus generator and the CPU
// reference runner.
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

    // Look up ground truth by ID. Presets do not avoid duplicate IDs, so when
    // several markers share an ID we pair each detection with the closest
    // candidate.
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
        // The synthetic image has no degradation, so with subpixel refinement
        // the error stays within 1.5 pixels.
        EXPECT_LT(best_error, 1.5) << "id=" << detection.id_;
    }
}

// Happy path: side_ratio is recorded in the manifest.
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

// Happy path: the detection-threshold formula returns the derived value.
// The lower bound is S + L * tau_i, not tau_i itself.
TEST(CorpusThresholdTest, minimum_detectable_side_follows_derivation) {
    EXPECT_NEAR(aruco3cuda::corpusgen::minimum_detectable_side_px(32, 1280, 0.05), 96.0, 1e-9);
    EXPECT_NEAR(aruco3cuda::corpusgen::minimum_detectable_side_px(32, 1920, 0.05), 128.0, 1e-9);
    // At tau_i = 0 no downscaling happens, so the lower bound equals S.
    EXPECT_NEAR(aruco3cuda::corpusgen::minimum_detectable_side_px(32, 1280, 0.0), 32.0, 1e-9);
}

// Happy path: a placement that crosses the image border sets fully_inside_ to
// false.
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

// Boundary: a scene with zero markers still generates, with empty ground
// truth.
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

// Happy path: generation succeeds and ground truth is preserved for each
// degradation condition listed in the evaluation plan. Distortion, lighting,
// blur, and occlusion are representative conditions that each need to be
// checked individually.
TEST_F(CorpusGeneratorTest, generates_each_degradation_condition) {
    struct Condition {
        const char* name;
        double rotation_deg;
        double perspective;
        double blur_sigma_px;
        double noise_sigma_levels;
        double illumination;
        double occlusion;
    };
    const Condition conditions[] = {
            {"rotation", 37.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            {"perspective", 0.0, 0.6, 0.0, 0.0, 0.0, 0.0},
            {"blur", 0.0, 0.0, 2.0, 0.0, 0.0, 0.0},
            {"noise", 0.0, 0.0, 0.0, 12.0, 0.0, 0.0},
            {"illumination", 0.0, 0.0, 0.0, 0.0, 0.8, 0.0},
            {"occlusion", 0.0, 0.0, 0.0, 0.0, 0.0, 0.25},
            {"combined", 21.0, 0.4, 1.0, 6.0, 0.5, 0.1},
    };
    for (const Condition& condition : conditions) {
        SceneSpec spec = clean_spec(condition.name);
        spec.marker_count_ = 2;
        spec.marker_side_px_ = 160.0;
        spec.rotation_deg_ = condition.rotation_deg;
        spec.perspective_strength_ = condition.perspective;
        spec.blur_sigma_px_ = condition.blur_sigma_px;
        spec.noise_sigma_levels_ = condition.noise_sigma_levels;
        spec.illumination_strength_ = condition.illumination;
        spec.occlusion_ratio_ = condition.occlusion;

        GeneratedScene scene;
        std::string error;
        ASSERT_TRUE(aruco3cuda::corpusgen::generate_scene(this->config_, spec, 0, &scene, &error))
                << condition.name << ": " << error;
        EXPECT_EQ(scene.markers_.size(), 2U) << condition.name;
        EXPECT_FALSE(scene.sha256_.empty()) << condition.name;
        // Even with degradation applied, ground truth is determined by the
        // generation parameters and does not change.
        for (const auto& marker : scene.markers_) {
            EXPECT_GT(marker.side_ratio_, 0.0) << condition.name;
        }
    }
}

// Happy path: the same degradation condition reproduces the same image from the
// same seed. Noise and distortion consume random numbers, so determinism is
// worth checking precisely under degradation.
TEST_F(CorpusGeneratorTest, degraded_scene_is_reproducible) {
    SceneSpec spec = clean_spec("degraded_repro");
    spec.rotation_deg_ = 17.0;
    spec.perspective_strength_ = 0.5;
    spec.blur_sigma_px_ = 1.5;
    spec.noise_sigma_levels_ = 8.0;
    spec.illumination_strength_ = 0.6;

    GeneratedScene first;
    GeneratedScene second;
    std::string error;
    ASSERT_TRUE(aruco3cuda::corpusgen::generate_scene(this->config_, spec, 3, &first, &error))
            << error;
    ASSERT_TRUE(aruco3cuda::corpusgen::generate_scene(this->config_, spec, 3, &second, &error))
            << error;
    EXPECT_EQ(first.sha256_, second.sha256_);
}

// Failure path: out-of-range settings generate nothing.
TEST_F(CorpusGeneratorTest, rejects_out_of_range_settings) {
    GeneratedScene scene;
    std::string error;

    CorpusConfig small_canonical = this->config_;
    small_canonical.canonical_marker_px_ = 4;
    EXPECT_FALSE(aruco3cuda::corpusgen::generate_scene(small_canonical, clean_spec("a"), 0, &scene,
                                                       &error));
    EXPECT_NE(error.find("canonical_marker_px"), std::string::npos) << error;

    SceneSpec too_much_blur = clean_spec("b");
    too_much_blur.blur_sigma_px_ = 1e9;
    EXPECT_FALSE(
            aruco3cuda::corpusgen::generate_scene(this->config_, too_much_blur, 0, &scene, &error));
    EXPECT_NE(error.find("blur_sigma_px"), std::string::npos) << error;

    SceneSpec oversized_marker = clean_spec("c");
    oversized_marker.marker_side_px_ = 100000.0;
    EXPECT_FALSE(aruco3cuda::corpusgen::generate_scene(this->config_, oversized_marker, 0, &scene,
                                                       &error));

    SceneSpec unnamed = clean_spec("d");
    unnamed.name_.clear();
    EXPECT_FALSE(aruco3cuda::corpusgen::generate_scene(this->config_, unnamed, 0, &scene, &error));
}

// Failure path: rejects an invalid spec and an invalid output destination.
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

// Happy path: the manifest contains the generation conditions and the ground
// truth.
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
