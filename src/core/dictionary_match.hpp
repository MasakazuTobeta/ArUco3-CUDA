// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_CORE_DICTIONARY_MATCH_HPP
#define ARUCO3CUDA_CORE_DICTIONARY_MATCH_HPP

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/dictionary.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/workspace.hpp"
#include "candidate_filter.hpp"
#include "cell_decode.hpp"

namespace aruco3cuda::detail {

/// device 上の Dictionary。
///
/// DictionaryTable の codes_ は host の静的記憶域を指すため、そのままでは
/// kernel から読めない。同じ内容を device へ写した参照を持つ。
///
/// 所有権: codes_ が指す領域の所有権は workspace にある。
/// 同期動作: 単なる参照であり同期点を持たない。
///
/// 入力例: DICT_ARUCO_MIP_36h12 の table
/// 出力例: marker_size_ = 6、code_count_ = 250、codes_ が 1000 要素
struct DeviceDictionary {
    /// [code_count_ * 4]。添字は id * 4 + rotation。host 側の並びと同じ。
    const MarkerCode* codes_ = nullptr;
    int marker_size_ = 0;
    int code_count_ = 0;
    int max_correction_bits_ = 0;
};

/// 候補ごとの照合結果。
///
/// 所有権: 全ての pointer が指す領域の所有権は workspace にある。
/// 同期動作: 単なる参照の集合であり同期点を持たない。
///
/// 入力例: 候補上限 4096
/// 出力例: ids_ が 4096 要素
struct MatchBuffers {
    /// 一致した ID。一致しなければ -1。
    std::int32_t* ids_ = nullptr;
    /// 一致した回転。0 から 3。一致しなければ 0。
    std::int32_t* rotations_ = nullptr;
    /// 採用した ID の距離。一致しなければ全 ID を通した最小距離。border 検証で
    /// 落ちた候補は照合しないため marker_size の 2 乗に 1 を足した値が入る。
    std::int32_t* distances_ = nullptr;
    int capacity_ = 0;
};

/// device 上の Dictionary に必要な workspace の大きさを返す。
///
/// @param table 対象 Dictionary。
/// @return 必要な byte 数。table が不正なら 0。
///
/// 所有権: 引数の領域を保持しない。
/// 同期動作: host 専用であり同期点を持たない。CUDA API を呼ばない。
///
/// 入力例: DICT_ARUCO_MIP_36h12 の table
/// 出力例: 8192
std::size_t device_dictionary_workspace_bytes(const DictionaryTable& table);

/// Dictionary を device へ写す領域を確保し、内容を転送する。
///
/// 転送元は host の静的記憶域であり pageable である。初期化時に 1 度だけ
/// 呼ぶことを想定しており、毎 frame の経路には現れない。
///
/// @param table 対象 Dictionary。
/// @param workspace device 空間の workspace。
/// @param out 成功時に device 側の参照を格納する。領域の所有権は呼出側にある。
/// @param stream 転送を発行する stream。
/// @return kOk。引数が不正なら kInvalidArgument、容量不足なら
///         容量不足なら kInvalidConfig、転送に失敗したら kCudaError。
///
/// 所有権: table が指す codes_ は転送元として読むだけで保持しない。
/// 同期動作: 転送は stream 上で非同期に発行する。呼出側が同期するまで
///           内容は確定しない。
///
/// 入力例: DICT_ARUCO_MIP_36h12 の table、空きのある workspace
/// 出力例: kOk。out->code_count_ = 250
Status upload_dictionary(const DictionaryTable& table, Workspace& workspace, DeviceDictionary* out,
                         cudaStream_t stream);

/// 照合結果に必要な workspace の大きさを返す。
///
/// @param config 候補上限を含む設定。
/// @return 必要な byte 数。config が不正なら 0。
///
/// 所有権: 引数の領域を保持しない。
/// 同期動作: host 専用であり同期点を持たない。CUDA API を呼ばない。
///
/// 入力例: max_candidates_ = 4096 の設定
/// 出力例: 49152
std::size_t match_workspace_bytes(const DetectorConfig& config);

/// 照合結果の領域を確保する。
///
/// @param config 候補上限を含む設定。
/// @param workspace device 空間の workspace。
/// @param out 成功時に buffer 群を格納する。領域の所有権は呼出側にある。
/// @return kOk。引数が不正なら kInvalidArgument、容量不足なら kInvalidConfig。
///
/// 所有権: 引数の領域を保持しない。
/// 同期動作: host 専用であり同期点を持たない。
///
/// 入力例: max_candidates_ = 4096 の設定、空きのある workspace
/// 出力例: kOk。out->capacity_ = 4096
Status reserve_matches(const DetectorConfig& config, Workspace& workspace, MatchBuffers* out);

/// 候補のセル比を Dictionary と照合する。
///
/// OpenCV の `Dictionary::identify` と同じ規則で判定する。セル比を
/// 「黒ではない」「白ではない」の 2 つの mask へ振り分け、ID の昇順に見て
/// 許容距離を満たした最初の ID を採る。最小距離の ID ではない。
///
/// border 検証を通らなかった候補は照合しない。ids_ に -1 を入れる。
///
/// @param ratios セル比と border 検証の結果。
/// @param candidates 候補数を持つ buffer。
/// @param dictionary device 側の Dictionary。
/// @param config 閾値と誤り訂正率を含む設定。
/// @param matches 出力先。
/// @param stream kernel を発行する stream。
/// @return kOk。引数が不正なら kInvalidArgument、設定が不整合なら
///         kInvalidConfig、kernel 起動に失敗したら kCudaError。
///
/// 所有権: 引数の領域を保持しない。
/// 同期動作: kernel を stream 上で非同期に発行する。呼出側が同期するまで
///           結果は確定しない。
///
/// 入力例: 4 候補分のセル比、DICT_ARUCO_MIP_36h12
/// 出力例: kOk。ids_ に 4 件の ID か -1 が入る
Status match_candidates_async(const CellRatioBuffers& ratios, const DeviceCandidates& candidates,
                              const DeviceDictionary& dictionary, const DetectorConfig& config,
                              MatchBuffers* matches, cudaStream_t stream);

}  // namespace aruco3cuda::detail

#endif  // ARUCO3CUDA_CORE_DICTIONARY_MATCH_HPP
