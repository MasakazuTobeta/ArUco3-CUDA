// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_CORPUSGEN_CORPUS_GENERATOR_HPP
#define ARUCO3CUDA_CORPUSGEN_CORPUS_GENERATOR_HPP

#include <array>
#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

namespace aruco3cuda::corpusgen {

/// Generation conditions for one synthetic image.
///
/// Every condition is stated explicitly rather than relying on defaults. The
/// premise is that the same spec and the same seed regenerate the same image.
struct SceneSpec {
    std::string name_;
    int width_px_ = 1280;
    int height_px_ = 720;
    int marker_count_ = 1;
    /// Target length of one marker side, in pixels. Perspective distortion makes the
    /// actual side length vary around it.
    double marker_side_px_ = 128.0;
    /// Rotation within the image plane, in degrees.
    double rotation_deg_ = 0.0;
    /// Strength of the perspective distortion. 0 means none; 1 displaces the corners
    /// by up to 25% of the side length.
    double perspective_strength_ = 0.0;
    /// Standard deviation of the Gaussian blur, in pixels. 0 disables it.
    double blur_sigma_px_ = 0.0;
    /// Standard deviation of the additive Gaussian noise, in levels (0 to 255).
    /// 0 disables it.
    double noise_sigma_levels_ = 0.0;
    /// Strength of the illumination gradient. 0 means uniform; 1 darkens the image
    /// edges by up to 50%.
    double illumination_strength_ = 0.0;
    /// Area ratio of the occlusion covering the marker. 0 means no occlusion.
    double occlusion_ratio_ = 0.0;
    /// Whether a placement that runs past the image boundary is allowed.
    bool allow_border_clip_ = false;
};

/// Ground truth for one generated marker.
struct MarkerGroundTruth {
    int id_ = -1;
    /// Holds the four corners in the order x0, y0, ... y3.
    /// The order corresponds to (0,0), (S,0), (S,S), (0,S) in marker coordinates.
    std::array<double, 8> corners_{};
    double side_px_ = 0.0;
    /// Side length divided by the longest image side. It compares directly against
    /// the ArUco3 minMarkerLengthRatioOriginalImg, so it tells which tau_i still
    /// keeps this marker a detection target.
    double side_ratio_ = 0.0;
    /// Whether all four corners fall inside the image.
    bool fully_inside_ = true;
    double occlusion_ratio_ = 0.0;
};

/// Result of generating one image.
struct GeneratedScene {
    std::string name_;
    std::string path_;
    std::string sha256_;
    int width_px_ = 0;
    int height_px_ = 0;
    SceneSpec spec_;
    std::vector<MarkerGroundTruth> markers_;
};

/// Generation settings for the corpus as a whole.
struct CorpusConfig {
    std::string dictionary_name_ = "DICT_ARUCO_MIP_36h12";
    std::string output_dir_ = "data/corpus";
    std::uint64_t seed_ = 20260827U;
    /// Canonical resolution used when rendering the marker image.
    /// Kept well above the size the marker is actually placed at, so that the
    /// resampling is always a downscale.
    int canonical_marker_px_ = 512;
    int marker_border_bits_ = 1;
};

/// Validates that the corpus settings and the scene spec lie in a valid range.
///
/// An out-of-range value either trips an OpenCV assert or leads to an enormous
/// memory allocation. generate_scene performs this validation first.
///
/// @param config Settings for the corpus as a whole. Only read, never retained.
/// @param spec Conditions of the scene in question. Only read, never retained.
/// @param out_error On failure, receives a reason containing "field=value". Must not
///        be nullptr. The storage is owned by the caller.
/// @return true when every field is valid.
///
/// Ownership: none of the argument storage is retained.
/// Synchronization: host only, with no synchronization point. Calls neither OpenCV
/// nor CUDA.
///
/// Example input: a config with canonical_marker_px_ = 0
/// Example output: false, with out_error holding a string containing
///                 "canonical_marker_px=0"
bool validate_scene(const CorpusConfig& config, const SceneSpec& spec, std::string* out_error);

/// Returns the smallest side length, in pixels, that remains a detection target
/// under the given ArUco3 settings.
///
/// Derivation:
///   The downscale factor is fxfy = S / (S + L * tau_i). The side length after the
///   downscale must be at least S, so side_px * fxfy >= S, which rearranges to
///   side_px >= S + L * tau_i. In terms of the ratio, side_ratio >= S / L + tau_i,
///   which is not tau_i itself.
///
/// @param min_side_length_canonical_img_px S, the OpenCV minSideLengthCanonicalImg.
/// @param longest_image_side_px L, the longest side of the image.
/// @param min_marker_length_ratio_original_img tau_i.
/// @return The lower bound on the side length, in pixels. Exactly at the boundary a
///         marker may still go undetected because of the resampling, so the corpus
///         leaves some margin.
double minimum_detectable_side_px(int min_side_length_canonical_img_px, int longest_image_side_px,
                                  double min_marker_length_ratio_original_img);

/// Builds the list of scene specs for a preset name.
///
/// @param preset One of "smoke", "basic", or "full". Only read, never retained.
/// @param out_specs On success, receives the list of specs, replacing any existing
///        contents. The storage is owned by the caller. Must not be nullptr.
/// @return true for a known preset.
///
/// Ownership: none of the argument storage is retained.
/// Synchronization: host only, with no synchronization point.
///
/// Example input: "smoke"
/// Example output: true, with out_specs holding 3 SceneSpec entries
bool build_preset(const std::string& preset, std::vector<SceneSpec>* out_specs);

/// Returns the list of supported preset names.
///
/// @return The list of preset names.
///
/// Ownership: the result is a value and is owned by the caller.
/// Synchronization: host only, with no synchronization point.
///
/// Example input: no arguments
/// Example output: {"smoke", "basic", "full"}
std::vector<std::string> known_presets();

/// Generates one image and saves it.
///
/// @param config Settings for the corpus as a whole.
/// @param spec Conditions of this image.
/// @param scene_index Sequential number of the image. Used to derive the random seed.
/// @param out_scene On success, receives the result.
/// @param out_error On failure, receives the reason.
/// @return true on success.
///
/// Note:
///   The random seed is derived from seed and scene_index. Deriving it from the
///   sequential number keeps the scenes independent, so adding a scene does not
///   change the content of the existing ones.
bool generate_scene(const CorpusConfig& config, const SceneSpec& spec, std::size_t scene_index,
                    GeneratedScene* out_scene, std::string* out_error);

/// Writes the manifest as JSON.
///
/// The bulky images are not committed to the repository; this manifest records
/// where they are stored along with their checksums.
void write_manifest_json(std::ostream& out, const CorpusConfig& config, const std::string& preset,
                         const std::vector<GeneratedScene>& scenes);

}  // namespace aruco3cuda::corpusgen

#endif  // ARUCO3CUDA_CORPUSGEN_CORPUS_GENERATOR_HPP
