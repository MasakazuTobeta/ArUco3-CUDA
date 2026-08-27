// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_CORE_SCAN_HPP
#define ARUCO3CUDA_CORE_SCAN_HPP

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "aruco3cuda/status.hpp"
#include "aruco3cuda/workspace.hpp"

namespace aruco3cuda::detail {

/// 排他 scan の作業領域。
///
/// GPU で「条件を満たす要素だけを詰める」処理は、要素ごとの 1/0 を
/// 排他 scan して書き込み先を決める形になる。atomicAdd で場所を取ると
/// 到着順で並びが変わり、実行ごとに結果の順序が変わってしまう。
///
/// 所有権: pointer が指す領域の所有権は workspace にある。
/// 同期動作: 単なる参照の集合であり同期点を持たない。
///
/// 入力例: 要素数 102480 に対する reserve_scan
/// 出力例: block_count_ = 401
struct ScanBuffers {
    /// block ごとの開始位置。
    std::int32_t* block_offsets_ = nullptr;
    /// 総和。要素数 1。
    std::int32_t* total_ = nullptr;
    int block_count_ = 0;
    int capacity_ = 0;
};

/// 要素数から scan の block 数を返す。
///
/// @param count 走査する要素数。
/// @return block 数。count が 0 以下なら 0。
///
/// 所有権: 資源を保持しない。
/// 同期動作: host 専用であり同期点を持たない。
///
/// 入力例: 102480
/// 出力例: 401
int scan_block_count(int count);

/// scan に必要な workspace の容量を返す。
///
/// @param count 走査する要素数。
/// @return 必要な byte 数。引数が不正なら 0。
///
/// 所有権: 資源を保持しない。
/// 同期動作: host 専用であり同期点を持たない。
///
/// 入力例: 102480
/// 出力例: block 401 個分と総和 1 個分を収める byte 数
std::size_t scan_workspace_bytes(int count);

/// workspace から scan 用の領域を切り出す。
///
/// @param count 走査する要素数。1 以上。
/// @param workspace 切り出し元。呼出側が所有する。
/// @param out 成功時に buffer 一式を格納する。nullptr は不可。
/// @return kOk。容量不足なら kInvalidConfig、引数が不正なら kInvalidArgument。
///
/// 所有権: 切り出した領域の所有権は workspace に残る。
/// 同期動作: host 専用であり同期点を持たない。
///
/// 入力例: 102480 と十分な容量の workspace
/// 出力例: block_count_ = 401 で pointer が入る
Status reserve_scan(int count, Workspace& workspace, ScanBuffers* out);

/// 配列をその場で排他 scan する。
///
/// 加算は整数であり、block へ分けても結果は分割の仕方に依存しない。
/// 同じ入力からは常に同じ出力が得られる。
///
/// @param values 走査する配列。入力を上書きする。nullptr は不可。
/// @param count 要素数。reserve_scan へ渡した値以下である必要がある。
/// @param buffers reserve_scan が返した buffer 一式。nullptr は不可。
/// @param stream 発行先の stream。既定 stream を使う場合は nullptr。
/// @return kOk、または kInvalidArgument、kCudaError。
///
/// 所有権: 引数が指す領域の所有権は呼出側と workspace に残る。
/// 同期動作: stream へ kernel を発行するだけで host 同期を行わない。
///           総和は buffers.total_ へ書かれ、host から読むには同期が要る。
///
/// 入力例: {1, 0, 1, 1}
/// 出力例: {0, 1, 1, 2}、total_ = 3
Status exclusive_scan_async(std::int32_t* values, int count, ScanBuffers* buffers,
                            cudaStream_t stream);

}  // namespace aruco3cuda::detail

#endif  // ARUCO3CUDA_CORE_SCAN_HPP
