// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_CORE_CANDIDATE_TREE_HPP
#define ARUCO3CUDA_CORE_CANDIDATE_TREE_HPP

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/workspace.hpp"
#include "candidate_filter.hpp"
#include "dictionary_match.hpp"

namespace aruco3cuda::detail {

/// 候補の包含関係と、走査が届いた範囲。
///
/// OpenCV は候補を入れ子の木として持ち、内側から順に識別する。内側で
/// マーカーが見つかったら、それを囲む候補は識別せずに済ませる。黒枠の
/// 外周と内周が両方候補になるため、この打ち切りが無いと同じマーカーが
/// 二重に出る。S9 の重複整理とはこの打ち切りのことである。
///
/// 所有権: 全ての pointer が指す領域の所有権は workspace にある。
/// 同期動作: 単なる参照の集合であり同期点を持たない。内容は発行済みの
///           kernel が完了するまで確定しない。
///
/// 入力例: 候補上限 4096 の設定
/// 出力例: parent_ と depth_ が 4096 要素、stop_depth_ が 1 要素
struct CandidateTreeBuffers {
    /// 自分を囲む候補の index。無ければ -1。
    ///
    /// OpenCV は index が自分より小さい候補のうち**最大の** index を選ぶ。
    /// index は周長の降順に並んでいるため、これは「自分を囲むもののうち
    /// 最も内側 (周長が最小)」を意味する。最小の index ではない。
    std::int32_t* parent_ = nullptr;
    /// 入れ子の段数。何にも囲まれていない候補から見た深さではなく、
    /// 自分が囲んでいる候補の段数である。最も内側が 0 になる。
    std::int32_t* depth_ = nullptr;
    /// 祖先として印を付けたか。1 で印あり。走査の途中経過であり、
    /// 識別を飛ばす条件としては使わない。
    ///
    /// 1 byte で足りるが 32 bit で持つ。印を付けるのに atomicExch を使い、
    /// CUDA には 8 bit 版が無いためである。
    std::int32_t* visited_ = nullptr;
    /// 走査が止まった段数。要素数 1。
    ///
    /// depth_ がこの値未満の候補だけが識別される。OpenCV の while ループが
    /// 抜けたときの depth に等しい。
    std::int32_t* stop_depth_ = nullptr;
    /// 走査が到達した数。要素数 1。検証と debug のために残す。
    std::int32_t* counter_ = nullptr;
    int capacity_ = 0;
};

/// 包含木に必要な workspace の大きさを返す。
///
/// @param config 候補上限を含む設定。
/// @return 必要な byte 数。config が不正なら 0。
///
/// 所有権: 引数の領域を保持しない。
/// 同期動作: host 専用であり同期点を持たない。CUDA API を呼ばない。
///
/// 入力例: max_candidates_ = 4096 の設定
/// 出力例: 49664
std::size_t candidate_tree_workspace_bytes(const DetectorConfig& config);

/// 包含木の領域を確保する。
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
Status reserve_candidate_tree(const DetectorConfig& config, Workspace& workspace,
                              CandidateTreeBuffers* out);

/// 候補の包含関係から親と段数を求める。
///
/// 包含判定は OpenCV の `pointPolygonTest` を `measureDist = false` で呼んだ
/// 場合と同じにする。四隅が 4 つとも相手の内側か境界上にあれば内側とみなす。
/// 交差積は 64 bit 整数で計算する。単精度では座標差の積が丸まり、符号が
/// 変わることがある。
///
/// 段数の伝播は index の降順に逐次で行う。並列にすると、まだ確定していない
/// 段数を読んで結果が変わる。候補数は多くても数千であり、逐次でも問題ない。
///
/// @param grouped 統合後の候補。周長の降順に並んでいることを前提とする。
/// @param tree 出力先。
/// @param stream kernel を発行する stream。
/// @return kOk。引数が不正なら kInvalidArgument、kernel 起動に失敗したら
///         kCudaError。
///
/// 所有権: 引数の領域を保持しない。
/// 同期動作: kernel を stream 上で非同期に発行する。呼出側が同期するまで
///           結果は確定しない。
///
/// 入力例: 2 段に入れ子になった 2 候補
/// 出力例: kOk。parent_ = {-1, 0}、depth_ = {1, 0}
Status build_candidate_tree_async(const DeviceCandidates& grouped, CandidateTreeBuffers* tree,
                                  cudaStream_t stream);

/// 識別の打ち切りが起きる段数を求める。
///
/// OpenCV は段数 0 から順に識別し、到達した候補の数が全体に届いた時点で
/// 止める。到達数は「その段で識別した候補」と「識別できた候補の祖先」の
/// 両方を数える。祖先として数えた候補は、自分の段に来たときにもう一度
/// 数える。この二重計上は打ち切りを早める方向に働くため、取り除いては
/// ならない。
///
/// @param grouped 候補数を持つ buffer。
/// @param matches 照合結果。ids_ が 0 以上の候補を識別できたとみなす。
/// @param tree 親と段数を格納済みの buffer。stop_depth_ と counter_ を書く。
/// @param stream kernel を発行する stream。
/// @return kOk。引数が不正なら kInvalidArgument、kernel 起動に失敗したら
///         kCudaError。
///
/// 所有権: 引数の領域を保持しない。
/// 同期動作: kernel を stream 上で非同期に発行する。呼出側が同期するまで
///           結果は確定しない。
///
/// 入力例: 内側の候補が識別できた 2 段の木
/// 出力例: kOk。stop_depth_ = 1。段数 1 の外側の候補は識別されない
Status resolve_suppression_async(const DeviceCandidates& grouped, const MatchBuffers& matches,
                                 CandidateTreeBuffers* tree, cudaStream_t stream);

}  // namespace aruco3cuda::detail

#endif  // ARUCO3CUDA_CORE_CANDIDATE_TREE_HPP
