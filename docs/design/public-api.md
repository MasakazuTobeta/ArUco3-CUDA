# 公開 API 草案

## 目的

CUDA 検出器の公開 API の型、所有権、同期動作、エラー通知方法を定義し、OpenCV 依存を core から分離した構成を示します。

## 対象範囲

`core` が公開する型と関数、設定値、結果表現、`adapter/opencv` の変換 API を対象とします。姿勢推定 API、Python binding、ROS2 interface は対象外です。

## 現状

`Status`、`MemorySpace`、`ImageViewU8`、`CornerRefineMethod`、`DetectorConfig` と、それぞれの境界検証は実装済みです。`Dictionary` は `include/aruco3cuda/dictionary.hpp` として実装済みです。`DeviceDetections`、`HostDetections`、`Detector` は未実装であり、本書の該当箇所は草案です。

既定値は OpenCV 4.x の `DetectorParameters` の観測仕様に合わせています。ArUco3 検出戦略に関わる 2 項目のみ、評価目的に合わせて既定を変えています。OpenCV と同じ既定値が必要な場合は `DetectorConfig::opencv_defaults()` を使います。

## 目標

### 設計原則

- core は OpenCV の型へ依存しません。`cv::Mat` と `cv::cuda::GpuMat` の変換は adapter の責務とします。
- 全ての非同期 API は caller 所有の `cudaStream_t` を受け取ります。既定 stream への暗黙依存を作りません。
- 失敗は戻り値の `Status` で通知します。core は例外を送出しません。adapter は OpenCV の慣習に合わせて例外へ変換できます。
- 作業領域は `Detector` が所有し、フレームごとの確保を行いません。
- 単位を持つ値は名前に単位を含めます。

### 名前空間と file 構成

```
include/aruco3cuda/
  status.hpp       // Status、last_cuda_error_message      実装済み
  version.hpp      // version 情報                          実装済み
  types.hpp        // MemorySpace、ImageViewU8、境界検証     実装済み
  config.hpp       // CornerRefineMethod、DetectorConfig    実装済み
  dictionary.hpp   // DictionaryTable、照合                  実装済み
  device_probe.hpp // device の性質取得                      実装済み
  detections.hpp   // DeviceDetections、HostDetections       未実装
  detector.hpp     // Detector                               未実装
include/aruco3cuda/opencv/
  adapter.hpp      // cv::Mat / cv::cuda::GpuMat との変換    未実装
```

### 基本型

```cpp
// SPDX-License-Identifier: Apache-2.0
namespace aruco3cuda {

/// 入力 buffer が存在する memory 空間。統合 GPU でも費用が異なるため明示する。
enum class MemorySpace : int {
  kHostPageable = 0,
  kHostPinned = 1,
  kManaged = 2,
  kDevice = 3,
};

/// 公開 API の結果状態。core は例外を送出しない。
enum class Status : int {
  kOk = 0,
  kInvalidImage,          ///< pointer、寸法、pitch が不正
  kInvalidConfig,         ///< 設定値が範囲外、または相互に矛盾
  kUnsupportedDictionary,
  kCandidateOverflow,     ///< 候補数が上限を超えた。結果は打ち切られている
  kMarkerOverflow,        ///< 検出数が上限を超えた。結果は打ち切られている
  kCudaError,             ///< CUDA API または kernel 起動の失敗
  kNotInitialized,
};

/// 8-bit grayscale 画像の非所有 view。所有権は呼出側に残る。
struct ImageViewU8 {
  const std::uint8_t* data_ = nullptr;
  int width_px_ = 0;
  int height_px_ = 0;
  std::size_t pitch_bytes_ = 0;   ///< 行間隔。width_px_ と等しいとは限らない
  MemorySpace space_ = MemorySpace::kDevice;
};

}  // namespace aruco3cuda
```

### 設定

```cpp
// SPDX-License-Identifier: Apache-2.0
namespace aruco3cuda {

enum class CornerRefineMethod : int {
  kNone = 0,
  kSubpix = 1,
};

/// 検出設定。既定値は OpenCV 4.x の DetectorParameters に合わせる。
struct DetectorConfig {
  // 適応的二値化
  int adaptive_thresh_win_size_min_ = 3;
  int adaptive_thresh_win_size_max_ = 23;
  int adaptive_thresh_win_size_step_ = 10;
  double adaptive_thresh_constant_ = 7.0;

  // 候補フィルタ
  double min_marker_perimeter_rate_ = 0.03;
  double max_marker_perimeter_rate_ = 4.0;
  double polygonal_approx_accuracy_rate_ = 0.03;
  double min_corner_distance_rate_ = 0.05;
  int min_distance_to_border_px_ = 3;
  double min_marker_distance_rate_ = 0.125;
  float min_group_distance_ = 0.21f;

  // ビット読取りと照合
  int marker_border_bits_ = 1;
  int perspective_remove_pixel_per_cell_ = 4;
  double perspective_remove_ignored_margin_per_cell_ = 0.13;
  double max_erroneous_bits_in_border_rate_ = 0.35;
  double min_otsu_std_dev_ = 5.0;
  double error_correction_rate_ = 0.6;
  float valid_bit_id_threshold_ = 0.49f;

  // 四隅補正
  CornerRefineMethod corner_refine_method_ = CornerRefineMethod::kSubpix;
  int corner_refinement_win_size_px_ = 5;
  int corner_refinement_max_iterations_ = 30;
  double corner_refinement_min_accuracy_ = 0.1;

  // ArUco3 検出戦略
  bool use_aruco3_detection_ = true;
  int min_side_length_canonical_img_px_ = 32;
  float min_marker_length_ratio_original_img_ = 0.05f;

  // CUDA 固有
  int max_candidates_ = 4096;      ///< 候補 buffer の上限。超過時は kCandidateOverflow
  int max_markers_ = 1024;         ///< 検出 buffer の上限。超過時は kMarkerOverflow
  int max_width_px_ = 3840;        ///< workspace 事前確保の基準
  int max_height_px_ = 2160;
  int candidate_block_size_ = 128; ///< 機種別に測定で決める。source の固定値にしない

  /// 相互に矛盾する設定を検出する。Detector 生成前に呼び出せる。
  Status validate() const;
};

}  // namespace aruco3cuda
```

`use_aruco3_detection_` の既定値と `min_marker_length_ratio_original_img_` の既定値は、OpenCV の既定と異なります。OpenCV は `useAruco3Detection = false`、`minMarkerLengthRatioOriginalImg = 0.0f` を既定とし、この組み合わせでは縮小が発生しません。本 project は ArUco3 検出戦略の評価が目的であるため既定を変更し、OpenCV 互換の既定値を返す `DetectorConfig::opencv_defaults()` を別に用意します。

### 結果表現

```cpp
// SPDX-License-Identifier: Apache-2.0
namespace aruco3cuda {

/// device 常駐の検出結果。GPU 常駐 pipeline から同期なしで参照できる。
struct DeviceDetections {
  int* ids_ = nullptr;               ///< [max_markers_]
  float2* corners_ = nullptr;        ///< [max_markers_ * 4]、反時計回りに正規化
  int* rotations_ = nullptr;         ///< [max_markers_]、0 から 3
  float* hamming_distances_ = nullptr;
  int* count_ = nullptr;             ///< device 上の検出数
  int* overflow_flags_ = nullptr;
  int capacity_ = 0;
};

/// host 側へ取り出した検出結果。
struct HostDetections {
  std::vector<int> ids_;
  std::vector<std::array<float, 8>> corners_;  ///< 四隅 x, y を反時計回りに格納
  std::vector<int> rotations_;
  std::vector<float> hamming_distances_;
  bool candidate_overflow_ = false;
  bool marker_overflow_ = false;
};

}  // namespace aruco3cuda
```

### 検出器

```cpp
// SPDX-License-Identifier: Apache-2.0
namespace aruco3cuda {

/// ArUco3 検出戦略の CUDA 実装。
///
/// 所有権:
///   workspace と DeviceDetections の backing memory を本 class が所有する。
///   入力画像の memory は呼出側が所有する。
/// 同期動作:
///   detect_async() は stream へ kernel を発行するだけで host 同期を行わない。
///   download() は stream の完了を待ってから host buffer を埋める。
class Detector {
 public:
  Detector() = default;
  ~Detector();
  Detector(const Detector&) = delete;
  Detector& operator=(const Detector&) = delete;
  Detector(Detector&&) noexcept;
  Detector& operator=(Detector&&) noexcept;

  /// workspace を確保し Dictionary を device へ転送する。
  /// 失敗時は内部状態を変更しない。
  Status initialize(const Dictionary& dictionary, const DetectorConfig& config);

  /// 検出を stream へ発行する。host 同期を行わない。
  /// 入力例: 1920x1080、pitch_bytes_ = 1920、space_ = kDevice
  /// 出力例: detections.count_ に device 上の検出数が書かれる
  Status detect_async(const ImageViewU8& image, cudaStream_t stream);

  /// 直近の detect_async() の device 常駐結果を返す。同期は発生しない。
  const DeviceDetections& device_detections() const;

  /// stream の完了を待って host 結果を得る。ここでのみ同期が発生する。
  Status download(HostDetections& out, cudaStream_t stream);

  /// 段階ごとの CUDA event 計測を有効にする。benchmark 以外では無効を推奨する。
  void set_stage_timing_enabled(bool enabled);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace aruco3cuda
```

### OpenCV adapter

```cpp
// SPDX-License-Identifier: Apache-2.0
namespace aruco3cuda::opencv {

/// cv::aruco::DetectorParameters を DetectorConfig へ写像する。
/// 対応しない parameter が設定されている場合は kInvalidConfig を返す。
Status from_detector_parameters(const cv::aruco::DetectorParameters& params,
                                DetectorConfig& out);

/// cv::aruco::Dictionary から device 用 Dictionary を構築する。
Status from_cv_dictionary(const cv::aruco::Dictionary& dictionary, Dictionary& out);

/// GpuMat から非所有 view を作る。連続性と型を検証する。
Status view_from_gpu_mat(const cv::cuda::GpuMat& mat, ImageViewU8& out);

/// 検出結果を OpenCV の detectMarkers と同じ表現へ変換する。
void to_cv_output(const HostDetections& detections,
                  std::vector<std::vector<cv::Point2f>>& corners,
                  std::vector<int>& ids);

}  // namespace aruco3cuda::opencv
```

adapter は別の CMake target とし、OpenCV を link しない構成でも core を build できるようにします。

### 使用例

```cpp
aruco3cuda::Detector detector;
aruco3cuda::DetectorConfig config;
config.max_width_px_ = 1920;
config.max_height_px_ = 1080;

if (detector.initialize(dictionary, config) != aruco3cuda::Status::kOk) { /* ... */ }

cudaStream_t stream = nullptr;
cudaStreamCreate(&stream);

aruco3cuda::ImageViewU8 view{d_image, 1920, 1080, pitch_bytes, aruco3cuda::MemorySpace::kDevice};
if (detector.detect_async(view, stream) != aruco3cuda::Status::kOk) { /* ... */ }

aruco3cuda::HostDetections result;
if (detector.download(result, stream) != aruco3cuda::Status::kOk) { /* ... */ }
```

## 実装上の判断

- `Status` を戻り値とし core から例外を出さないことで、CUDA callback、destructor、device code からの例外送出を構造的に防ぎます。
- `DeviceDetections` を公開することで、GPU 常駐 pipeline が host 同期なしに結果を利用できます。[評価計画](../evaluation-plan.md) の `CUDA-Resident` 経路はこの API を使用します。
- `pitch_bytes_` を必須にし、ROI と非連続入力を初期から扱えるようにします。
- overflow を戻り値と結果 flag の両方で通知し、無言の切り捨てを行いません。
- `Impl` を pimpl とし、公開 header が CUDA 固有型へ依存する範囲を `cudaStream_t` と `float2` に限定します。

## 決定した事項

### 公開 aggregate の field にも末尾 `_` を付ける

`CONTRIBUTING.md` の命名規則をそのまま適用します。既に `DeviceProbeResult`、`DictionaryTable`、`ReferenceConfig`、`SceneSpec`、`BenchmarkConfig` が同じ規則で実装されており、公開 aggregate だけ規則を変えると、同じ repository 内で 2 つの規則が混在します。呼出側の記述はやや冗長になりますが、規則が 1 つである利点を優先します。

### 検証は Status を返し、理由は任意の out 引数で受け取る

`validate_image_view()` と `DetectorConfig::validate()` は `Status` を返し、失敗理由は `std::string*` へ格納します。`nullptr` を渡してよく、その場合は文字列を組み立てません。検証は毎フレーム呼ばれ得るため、成功経路で確保を発生させない構造にしています。

### 画像の失敗に専用の Status を割り当てる

`kInvalidImage` を `kInvalidArgument` と別に設けます。画像 view の不正は呼出側の入力経路の問題であり、設定や index の誤りとは対処が異なります。

## 未確定事項

- `float2` を公開 header で使用するか、独自の `Point2f` 相当を定義するか。`DeviceDetections` の実装時に決めます。
- `download()` 以外に、部分結果だけを取得する API を用意するか。
- 複数 Dictionary の同時照合を初期 scope に含めるか。
- OpenCV adapter を同一 library に含めるか別 target にするか。[アーキテクチャ](../architecture.md) の未確定事項と同じ。
- `cv::aruco::DetectorParameters` の非対応 parameter を無視するか拒否するか。

## 関連

- [アーキテクチャ](../architecture.md)
- [検出パイプライン設計](detector-pipeline.md)
- [実装計画](../implementation-plan.md)
- [Code Provenance 記録](../code-provenance.md)
