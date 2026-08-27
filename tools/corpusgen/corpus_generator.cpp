// SPDX-License-Identifier: Apache-2.0
#include "corpus_generator.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <ostream>
#include <random>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "aruco3cuda/util/json_writer.hpp"
#include "aruco3cuda/util/sha256.hpp"

namespace aruco3cuda::corpusgen {
namespace {

using aruco3cuda::util::JsonWriter;

constexpr double kPi = 3.14159265358979323846;

const std::map<std::string, int>& dictionary_id_table() {
    static const std::map<std::string, int> kTable = {
            {"DICT_4X4_50", cv::aruco::DICT_4X4_50},
            {"DICT_5X5_250", cv::aruco::DICT_5X5_250},
            {"DICT_6X6_250", cv::aruco::DICT_6X6_250},
            {"DICT_7X7_250", cv::aruco::DICT_7X7_250},
            {"DICT_ARUCO_ORIGINAL", cv::aruco::DICT_ARUCO_ORIGINAL},
            {"DICT_ARUCO_MIP_36h12", cv::aruco::DICT_ARUCO_MIP_36h12},
    };
    return kTable;
}

/// scene ごとに独立した乱数列を作る。
///
/// seed だけを共有し scene_index で分岐させることで、scene を追加しても
/// 既存 scene の乱数列が変わらないようにする。
std::mt19937_64 make_rng(std::uint64_t seed, std::size_t scene_index) {
    // 黄金比に基づく定数で混ぜる。連続する index が近い状態にならないようにする。
    constexpr std::uint64_t kMix = 0x9E3779B97F4A7C15ULL;
    return std::mt19937_64(seed ^ (kMix * static_cast<std::uint64_t>(scene_index + 1U)));
}

double uniform(std::mt19937_64& rng, double low, double high) {
    std::uniform_real_distribution<double> distribution(low, high);
    return distribution(rng);
}

/// マーカー座標系の四隅を、指定した中心・辺長・回転・射影歪みで画像平面へ写す。
std::array<cv::Point2d, 4> make_destination_quad(std::mt19937_64& rng, const SceneSpec& spec,
                                                 double center_x, double center_y) {
    const double half = spec.marker_side_px_ * 0.5;
    // マーカー座標系での四隅。順序は (0,0)、(S,0)、(S,S)、(0,S)。
    const std::array<cv::Point2d, 4> local = {cv::Point2d(-half, -half), cv::Point2d(half, -half),
                                              cv::Point2d(half, half), cv::Point2d(-half, half)};

    const double angle = spec.rotation_deg_ * kPi / 180.0;
    const double cos_a = std::cos(angle);
    const double sin_a = std::sin(angle);

    std::array<cv::Point2d, 4> quad{};
    for (std::size_t i = 0; i < 4; ++i) {
        double x = local[i].x * cos_a - local[i].y * sin_a;
        double y = local[i].x * sin_a + local[i].y * cos_a;
        if (spec.perspective_strength_ > 0.0) {
            // 四隅を独立にずらして射影歪みを作る。最大で辺長の 25%。
            const double range = spec.marker_side_px_ * 0.25 * spec.perspective_strength_;
            x += uniform(rng, -range, range);
            y += uniform(rng, -range, range);
        }
        quad[i] = cv::Point2d(center_x + x, center_y + y);
    }
    return quad;
}

bool quad_inside_image(const std::array<cv::Point2d, 4>& quad, int width_px, int height_px) {
    for (const cv::Point2d& point : quad) {
        if (point.x < 0.0 || point.y < 0.0 || point.x > width_px - 1.0 ||
            point.y > height_px - 1.0) {
            return false;
        }
    }
    return true;
}

/// 決定的な Gaussian noise を加える。OpenCV の大域 RNG を使わない。
void add_noise(cv::Mat& image, std::mt19937_64& rng, double sigma_levels) {
    if (sigma_levels <= 0.0) {
        return;
    }
    std::normal_distribution<double> distribution(0.0, sigma_levels);
    for (int row = 0; row < image.rows; ++row) {
        auto* pixels = image.ptr<std::uint8_t>(row);
        for (int col = 0; col < image.cols; ++col) {
            const double value = static_cast<double>(pixels[col]) + distribution(rng);
            pixels[col] = cv::saturate_cast<std::uint8_t>(value);
        }
    }
}

/// 中心から端へ向かって暗くなる照度勾配を掛ける。
void apply_illumination(cv::Mat& image, double strength) {
    if (strength <= 0.0) {
        return;
    }
    const double center_x = (image.cols - 1) * 0.5;
    const double center_y = (image.rows - 1) * 0.5;
    const double max_distance = std::sqrt(center_x * center_x + center_y * center_y);
    for (int row = 0; row < image.rows; ++row) {
        auto* pixels = image.ptr<std::uint8_t>(row);
        for (int col = 0; col < image.cols; ++col) {
            const double dx = col - center_x;
            const double dy = row - center_y;
            const double ratio = std::sqrt(dx * dx + dy * dy) / max_distance;
            const double gain = 1.0 - 0.5 * strength * ratio;
            pixels[col] = cv::saturate_cast<std::uint8_t>(pixels[col] * gain);
        }
    }
}

}  // namespace

bool validate_scene(const CorpusConfig& config, const SceneSpec& spec, std::string* out_error) {
    if (out_error == nullptr) {
        return false;
    }
    struct IntRange {
        const char* name;
        int value;
        int minimum;
        int maximum;
    };
    // canonical 解像度の上限は memory 使用量から決める。1 marker あたり
    // canonical_marker_px^2 byte を確保するため、8192 で 64 MiB になる。
    const IntRange int_ranges[] = {
            {"width_px", spec.width_px_, 1, 65536},
            {"height_px", spec.height_px_, 1, 65536},
            {"marker_count", spec.marker_count_, 0, 4096},
            {"canonical_marker_px", config.canonical_marker_px_, 8, 8192},
            {"marker_border_bits", config.marker_border_bits_, 1, 16},
    };
    for (const IntRange& range : int_ranges) {
        if (range.value < range.minimum || range.value > range.maximum) {
            *out_error = std::string("設定値が範囲外: ") + range.name + "=" +
                         std::to_string(range.value) + " (有効範囲 " +
                         std::to_string(range.minimum) + " から " + std::to_string(range.maximum) +
                         ")";
            return false;
        }
    }
    // OpenCV の generateImageMarker は辺長が border 込みの cell 数以上を要求する。
    if (config.canonical_marker_px_ < config.marker_border_bits_ * 2 + 1) {
        *out_error = "設定値が矛盾: canonical_marker_px が marker_border_bits に対して小さすぎる";
        return false;
    }

    struct DoubleRange {
        const char* name;
        double value;
        double minimum;
        double maximum;
    };
    const DoubleRange double_ranges[] = {
            {"marker_side_px", spec.marker_side_px_, 1.0, 65536.0},
            {"rotation_deg", spec.rotation_deg_, -360.0, 360.0},
            {"perspective_strength", spec.perspective_strength_, 0.0, 1.0},
            {"blur_sigma_px", spec.blur_sigma_px_, 0.0, 256.0},
            {"noise_sigma_levels", spec.noise_sigma_levels_, 0.0, 255.0},
            {"illumination_strength", spec.illumination_strength_, 0.0, 1.0},
            {"occlusion_ratio", spec.occlusion_ratio_, 0.0, 1.0},
    };
    for (const DoubleRange& range : double_ranges) {
        // NaN は比較が全て false になるため、この書き方で同時に弾ける。
        if (!(range.value >= range.minimum) || !(range.value <= range.maximum)) {
            *out_error = std::string("設定値が範囲外: ") + range.name + "=" +
                         std::to_string(range.value) + " (有効範囲 " +
                         std::to_string(range.minimum) + " から " + std::to_string(range.maximum) +
                         ")";
            return false;
        }
    }
    if (spec.marker_count_ > 0 &&
        (spec.marker_side_px_ > spec.width_px_ || spec.marker_side_px_ > spec.height_px_)) {
        *out_error = "設定値が矛盾: marker_side_px が画像寸法を超えている";
        return false;
    }
    if (spec.name_.empty()) {
        *out_error = "scene 名が空である。出力 file 名を決められない";
        return false;
    }
    return true;
}

double minimum_detectable_side_px(int min_side_length_canonical_img_px, int longest_image_side_px,
                                  double min_marker_length_ratio_original_img) {
    return static_cast<double>(min_side_length_canonical_img_px) +
           static_cast<double>(longest_image_side_px) * min_marker_length_ratio_original_img;
}

std::vector<std::string> known_presets() {
    return {"smoke", "basic", "full"};
}

bool build_preset(const std::string& preset, std::vector<SceneSpec>* out_specs) {
    if (out_specs == nullptr) {
        return false;
    }
    std::vector<SceneSpec> specs;

    // 評価計画の入力条件に対応する。解像度、マーカー数、辺長、劣化条件を分けて並べる。
    const std::vector<std::pair<int, int>> resolutions_full = {
            {640, 480}, {1280, 720}, {1920, 1080}, {3840, 2160}};
    const std::vector<std::pair<int, int>> resolutions_basic = {{640, 480}, {1280, 720}};

    auto add_clean = [&specs](const std::string& prefix, int width, int height, int count,
                              double side) {
        SceneSpec spec;
        spec.name_ = prefix + "_" + std::to_string(width) + "x" + std::to_string(height) + "_n" +
                     std::to_string(count) + "_s" + std::to_string(static_cast<int>(side));
        spec.width_px_ = width;
        spec.height_px_ = height;
        spec.marker_count_ = count;
        spec.marker_side_px_ = side;
        specs.push_back(spec);
    };

    if (preset == "smoke") {
        // 既定の S = 32、tau_i = 0.05 では、1280x720 における下限辺長は
        // 32 + 1280 * 0.05 = 96 pixel になる。境界ちょうどでは再標本化の影響で
        // 検出できないため、smoke では余裕のある 128 pixel を使う。
        add_clean("clean", 1280, 720, 1, 128);
        add_clean("clean", 1280, 720, 4, 128);
        SceneSpec degraded;
        degraded.name_ = "degraded_1280x720";
        degraded.marker_count_ = 4;
        degraded.marker_side_px_ = 128;
        degraded.rotation_deg_ = 23.0;
        degraded.perspective_strength_ = 0.3;
        degraded.blur_sigma_px_ = 1.2;
        degraded.noise_sigma_levels_ = 5.0;
        degraded.illumination_strength_ = 0.4;
        specs.push_back(degraded);
    } else if (preset == "basic" || preset == "full") {
        const auto& resolutions = (preset == "basic") ? resolutions_basic : resolutions_full;
        const std::vector<int> counts = {0, 1, 4, 16};
        const std::vector<double> sides = (preset == "basic")
                                                  ? std::vector<double>{32, 128}
                                                  : std::vector<double>{16, 32, 64, 128, 256};
        for (const auto& resolution : resolutions) {
            for (const int count : counts) {
                for (const double side : sides) {
                    if (count == 0 && side != sides.front()) {
                        continue;  // マーカー 0 個の場合は辺長を変えても同じ画像になる
                    }
                    // マーカーが画像へ収まらない組み合わせは飛ばす。
                    const double needed = side * std::sqrt(static_cast<double>(std::max(count, 1)));
                    if (count > 0 &&
                        (needed > resolution.first * 0.9 || needed > resolution.second * 0.9)) {
                        continue;
                    }
                    add_clean("clean", resolution.first, resolution.second, count, side);
                }
            }
        }
        // 劣化条件。基準となる解像度とマーカー数を固定し、条件を 1 つずつ変える。
        struct Degradation {
            const char* name;
            double rotation_deg;
            double perspective;
            double blur_sigma_px;
            double noise_sigma_levels;
            double illumination;
            double occlusion;
            bool border_clip;
        };
        const std::vector<Degradation> degradations = {
                {"rotation", 37.0, 0.0, 0.0, 0.0, 0.0, 0.0, false},
                {"perspective", 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, false},
                {"blur", 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, false},
                {"noise", 0.0, 0.0, 0.0, 12.0, 0.0, 0.0, false},
                {"illumination", 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, false},
                {"occlusion", 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, false},
                {"border", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, true},
                {"combined", 21.0, 0.4, 1.0, 6.0, 0.5, 0.1, false},
        };
        for (const auto& resolution : resolutions) {
            for (const Degradation& degradation : degradations) {
                SceneSpec spec;
                spec.name_ = std::string(degradation.name) + "_" +
                             std::to_string(resolution.first) + "x" +
                             std::to_string(resolution.second);
                spec.width_px_ = resolution.first;
                spec.height_px_ = resolution.second;
                spec.marker_count_ = 4;
                spec.marker_side_px_ = 96;
                spec.rotation_deg_ = degradation.rotation_deg;
                spec.perspective_strength_ = degradation.perspective;
                spec.blur_sigma_px_ = degradation.blur_sigma_px;
                spec.noise_sigma_levels_ = degradation.noise_sigma_levels;
                spec.illumination_strength_ = degradation.illumination;
                spec.occlusion_ratio_ = degradation.occlusion;
                spec.allow_border_clip_ = degradation.border_clip;
                specs.push_back(spec);
            }
        }
    } else {
        return false;
    }

    *out_specs = specs;
    return true;
}

bool generate_scene(const CorpusConfig& config, const SceneSpec& spec, std::size_t scene_index,
                    GeneratedScene* out_scene, std::string* out_error) {
    if (out_scene == nullptr || out_error == nullptr) {
        return false;
    }
    const auto dictionary_entry = dictionary_id_table().find(config.dictionary_name_);
    if (dictionary_entry == dictionary_id_table().end()) {
        *out_error = "未対応の Dictionary: " + config.dictionary_name_;
        return false;
    }
    if (!validate_scene(config, spec, out_error)) {
        return false;
    }

    std::mt19937_64 rng = make_rng(config.seed_, scene_index);
    const cv::aruco::Dictionary dictionary =
            cv::aruco::getPredefinedDictionary(dictionary_entry->second);

    // 背景は一様な明るい灰色。純白にすると noise が片側へのみ振れる。
    cv::Mat scene(spec.height_px_, spec.width_px_, CV_8UC1, cv::Scalar(230));

    // マーカーを重ならないように格子状へ配置し、各区画内で少し揺らす。
    const int columns = static_cast<int>(
            std::ceil(std::sqrt(static_cast<double>(std::max(spec.marker_count_, 1)))));
    const int rows = (spec.marker_count_ + columns - 1) / std::max(columns, 1);
    const double cell_width = static_cast<double>(spec.width_px_) / std::max(columns, 1);
    const double cell_height = static_cast<double>(spec.height_px_) / std::max(rows, 1);

    GeneratedScene result;
    result.name_ = spec.name_;
    result.spec_ = spec;
    result.width_px_ = spec.width_px_;
    result.height_px_ = spec.height_px_;

    // canonical 画像の外周を pixel の境界で表す。pixel 中心が 0 のとき、
    // 外周は -0.5 から size - 0.5 になる。ground truth はこの規約で定義する。
    const float canonical_edge = static_cast<float>(config.canonical_marker_px_) - 0.5F;
    const std::array<cv::Point2f, 4> source_quad = {
            cv::Point2f(-0.5F, -0.5F), cv::Point2f(canonical_edge, -0.5F),
            cv::Point2f(canonical_edge, canonical_edge), cv::Point2f(-0.5F, canonical_edge)};

    for (int index = 0; index < spec.marker_count_; ++index) {
        const int marker_id =
                static_cast<int>(rng() % static_cast<std::uint64_t>(dictionary.bytesList.rows));
        cv::Mat marker;
        dictionary.generateImageMarker(marker_id, config.canonical_marker_px_, marker,
                                       config.marker_border_bits_);

        const int cell_col = index % std::max(columns, 1);
        const int cell_row = index / std::max(columns, 1);
        double center_x = (cell_col + 0.5) * cell_width;
        double center_y = (cell_row + 0.5) * cell_height;
        // 区画内で位置を揺らす。マーカーが区画からはみ出さない範囲に留める。
        const double jitter_x = std::max(0.0, (cell_width - spec.marker_side_px_ * 1.5) * 0.5);
        const double jitter_y = std::max(0.0, (cell_height - spec.marker_side_px_ * 1.5) * 0.5);
        center_x += uniform(rng, -jitter_x, jitter_x);
        center_y += uniform(rng, -jitter_y, jitter_y);

        if (spec.allow_border_clip_ && index == 0) {
            // 画像境界にかかる配置を意図的に作る。
            center_x = spec.marker_side_px_ * 0.35;
            center_y = spec.height_px_ * 0.5;
        }

        const std::array<cv::Point2d, 4> quad =
                make_destination_quad(rng, spec, center_x, center_y);

        std::array<cv::Point2f, 4> destination_quad{};
        for (std::size_t i = 0; i < 4; ++i) {
            destination_quad[i] =
                    cv::Point2f(static_cast<float>(quad[i].x), static_cast<float>(quad[i].y));
        }

        const cv::Mat homography =
                cv::getPerspectiveTransform(source_quad.data(), destination_quad.data());

        cv::Mat warped;
        cv::warpPerspective(marker, warped, homography, scene.size(), cv::INTER_AREA,
                            cv::BORDER_CONSTANT, cv::Scalar(0));
        cv::Mat mask_source(marker.size(), CV_8UC1, cv::Scalar(255));
        cv::Mat mask;
        cv::warpPerspective(mask_source, mask, homography, scene.size(), cv::INTER_NEAREST,
                            cv::BORDER_CONSTANT, cv::Scalar(0));
        warped.copyTo(scene, mask);

        MarkerGroundTruth truth;
        truth.id_ = marker_id;
        truth.side_px_ = spec.marker_side_px_;
        truth.side_ratio_ = spec.marker_side_px_ /
                            static_cast<double>(std::max(spec.width_px_, spec.height_px_));
        for (std::size_t i = 0; i < 4; ++i) {
            truth.corners_[i * 2] = quad[i].x;
            truth.corners_[i * 2 + 1] = quad[i].y;
        }
        truth.fully_inside_ = quad_inside_image(quad, spec.width_px_, spec.height_px_);

        if (spec.occlusion_ratio_ > 0.0) {
            // マーカーの一部を矩形で覆う。面積比から矩形の幅を決める。
            const double width = spec.marker_side_px_ * spec.occlusion_ratio_;
            const cv::Rect2d occluder(center_x - spec.marker_side_px_ * 0.5,
                                      center_y - spec.marker_side_px_ * 0.5, width,
                                      spec.marker_side_px_);
            cv::rectangle(scene, occluder, cv::Scalar(128), cv::FILLED);
            truth.occlusion_ratio_ = spec.occlusion_ratio_;
        }
        result.markers_.push_back(truth);
    }

    // 劣化は配置の後に画像全体へ適用する。順序を固定して再現性を保つ。
    if (spec.blur_sigma_px_ > 0.0) {
        cv::GaussianBlur(scene, scene, cv::Size(0, 0), spec.blur_sigma_px_, spec.blur_sigma_px_,
                         cv::BORDER_REPLICATE);
    }
    apply_illumination(scene, spec.illumination_strength_);
    add_noise(scene, rng, spec.noise_sigma_levels_);

    // 出力先が無ければ作る。imwrite は directory を作らない。
    std::error_code directory_error;
    std::filesystem::create_directories(config.output_dir_, directory_error);
    if (directory_error) {
        *out_error = "出力 directory を作成できない: " + config.output_dir_;
        return false;
    }

    result.path_ = config.output_dir_ + "/" + spec.name_ + ".png";
    // PNG は可逆であり、圧縮 level を固定すれば byte 単位で再現できる。
    const std::vector<int> encode_params = {cv::IMWRITE_PNG_COMPRESSION, 6};
    if (!cv::imwrite(result.path_, scene, encode_params)) {
        *out_error = "画像を保存できない: " + result.path_;
        return false;
    }
    if (!aruco3cuda::util::sha256_file(result.path_, &result.sha256_)) {
        *out_error = "checksum を計算できない: " + result.path_;
        return false;
    }

    *out_scene = result;
    return true;
}

void write_manifest_json(std::ostream& out, const CorpusConfig& config, const std::string& preset,
                         const std::vector<GeneratedScene>& scenes) {
    JsonWriter writer(out);
    writer.begin_object();
    writer.member_int("schema_version", 1);
    writer.member_string("producer", "aruco3cuda_corpusgen");
    writer.member_string("preset", preset);
    writer.member_string("dictionary", config.dictionary_name_);
    writer.member_int("seed", static_cast<long long>(config.seed_));
    writer.member_int("canonical_marker_px", config.canonical_marker_px_);
    writer.member_int("marker_border_bits", config.marker_border_bits_);

    writer.key("scenes");
    writer.begin_array();
    for (const GeneratedScene& scene : scenes) {
        writer.begin_object();
        writer.member_string("name", scene.name_);
        writer.member_string("path", scene.path_);
        writer.member_string("sha256", scene.sha256_);
        writer.member_int("width_px", scene.width_px_);
        writer.member_int("height_px", scene.height_px_);

        writer.key("conditions");
        writer.begin_object();
        writer.member_int("marker_count", scene.spec_.marker_count_);
        writer.member_double("marker_side_px", scene.spec_.marker_side_px_, 3);
        writer.member_double("rotation_deg", scene.spec_.rotation_deg_, 3);
        writer.member_double("perspective_strength", scene.spec_.perspective_strength_, 3);
        writer.member_double("blur_sigma_px", scene.spec_.blur_sigma_px_, 3);
        writer.member_double("noise_sigma_levels", scene.spec_.noise_sigma_levels_, 3);
        writer.member_double("illumination_strength", scene.spec_.illumination_strength_, 3);
        writer.member_double("occlusion_ratio", scene.spec_.occlusion_ratio_, 3);
        writer.member_bool("allow_border_clip", scene.spec_.allow_border_clip_);
        writer.end_object();

        writer.key("ground_truth");
        writer.begin_array();
        for (const MarkerGroundTruth& marker : scene.markers_) {
            writer.begin_object();
            writer.member_int("id", marker.id_);
            writer.member_double("side_px", marker.side_px_, 3);
            writer.member_double("side_ratio", marker.side_ratio_, 6);
            writer.member_bool("fully_inside", marker.fully_inside_);
            writer.member_double("occlusion_ratio", marker.occlusion_ratio_, 3);
            writer.key("corners");
            writer.begin_array();
            for (std::size_t i = 0; i < 4; ++i) {
                writer.begin_array();
                writer.value_double(marker.corners_[i * 2], 4);
                writer.value_double(marker.corners_[i * 2 + 1], 4);
                writer.end_array();
            }
            writer.end_array();
            writer.end_object();
        }
        writer.end_array();
        writer.end_object();
    }
    writer.end_array();
    writer.end_object();
    out << '\n';
}

}  // namespace aruco3cuda::corpusgen
