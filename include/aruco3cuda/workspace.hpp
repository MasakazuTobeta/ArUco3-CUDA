// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_WORKSPACE_HPP
#define ARUCO3CUDA_WORKSPACE_HPP

#include <cstddef>
#include <string>

#include "aruco3cuda/status.hpp"
#include "aruco3cuda/types.hpp"

namespace aruco3cuda {

/// workspace の使用状況。
///
/// 目的:
///   フレームごとの確保が発生していないことを、呼出側とテストから確認できる
///   ようにする。規約はフレームごとの `cudaMalloc` と `cudaFree` を避けることを
///   求めるが、守れているかは統計を見なければ分からない。
struct WorkspaceStatistics {
    /// 実際に device memory を確保した回数。定常状態では増えない。
    std::size_t allocation_count_ = 0;
    /// 容量不足により確保し直した回数。allocation_count_ に含まれる。
    std::size_t reallocation_count_ = 0;
    /// 現在の容量。単位は byte。
    std::size_t capacity_bytes_ = 0;
    /// 現在の切り出し済み量。単位は byte。
    std::size_t used_bytes_ = 0;
    /// これまでの切り出し量の最大値。単位は byte。
    std::size_t peak_used_bytes_ = 0;
    /// 容量不足で切り出しに失敗した回数。
    std::size_t exhausted_count_ = 0;
};

/// 検出処理の中間 buffer をまとめて保持する arena。
///
/// 目的:
///   段階ごとに個別の確保を行うと、フレームごとに `cudaMalloc` と `cudaFree` が
///   並び、その費用が測定へ混入する。1 つの大きな領域を最初に確保し、
///   段階ごとの buffer はそこから切り出す。
///
/// 設計:
///   切り出しは bump pointer 方式で行い、個別の解放は提供しない。フレームの
///   先頭で reset() を呼び、切り出し位置を戻す。中間 buffer の寿命は
///   フレーム内で閉じるため、この方式で足りる。
///
///   allocate() は容量が足りなくても自動で拡張しない。自動拡張はフレーム
///   ごとの確保を招き、規約が避けよと定める状態を静かに作る。容量は
///   ensure_capacity() で検出器の初期化時に確保する。
///
/// 所有権:
///   確保した device memory はこの class が所有し、destructor で解放する。
///   allocate() が返す pointer は arena 内を指し、呼出側は解放しない。
///   arena が reset() されるか破棄されると無効になる。
///
/// 同期動作:
///   ensure_capacity() と release() は `cudaMalloc` と `cudaFree` を呼ぶ。
///   これらは暗黙の device 同期を伴うため、検出中には呼ばない設計とする。
///   他の member 関数は host 側の計算のみで同期点を持たない。
///   **この所有権と同期動作は全ての public member 関数に適用される。**
///   1 つの instance を複数 thread から同時に使用してはならない。
///
/// 入力例:
///   Workspace workspace;
///   workspace.ensure_capacity(1 << 20, MemorySpace::kDevice);
///   void* buffer = nullptr;
///   workspace.allocate(4096, 256, &buffer);
/// 出力例:
///   statistics().allocation_count_ == 1、used_bytes_ == 4096
class Workspace {
public:
    Workspace() = default;
    ~Workspace();

    Workspace(const Workspace&) = delete;
    Workspace& operator=(const Workspace&) = delete;
    Workspace(Workspace&& other) noexcept;
    Workspace& operator=(Workspace&& other) noexcept;

    /// 指定した容量を確保する。既に足りていれば何もしない。
    ///
    /// 足りない場合は確保し直す。既存の切り出しは全て無効になるため、
    /// 検出中ではなく初期化時に呼ぶこと。
    ///
    /// @param bytes 必要な容量。0 を渡すと何もせず kOk を返す。
    /// @param space 確保する memory 空間。既存の容量と異なる空間を指定した
    ///              場合は確保し直す。
    /// @param out_message 失敗時に理由を格納する。nullptr を渡してよい。
    /// @return kOk。確保に失敗した場合は kCudaError。
    ///
    /// 入力例: bytes = 1048576、space = MemorySpace::kDevice
    /// 出力例: Status::kOk。statistics().capacity_bytes_ が 1048576 以上になる
    Status ensure_capacity(std::size_t bytes, MemorySpace space,
                           std::string* out_message = nullptr);

    /// arena から領域を切り出す。
    ///
    /// @param bytes 必要な byte 数。0 を渡すと *out を nullptr にして kOk を返す。
    /// @param alignment 境界。2 の冪である必要がある。
    /// @param out 切り出した領域の先頭を格納する。nullptr は不可。
    ///            所有権は arena に残り、呼出側は解放しない。
    /// @return kOk。out が nullptr、または alignment が 2 の冪でない場合は
    ///         kInvalidArgument。容量が足りない場合は kInvalidConfig を返す。
    ///         容量は設定から算出するため、不足は設定の誤りを意味する。
    ///
    /// 同期動作: host 側の計算のみで同期点を持たない。CUDA API を呼ばない。
    ///
    /// 入力例: bytes = 4096、alignment = 256
    /// 出力例: Status::kOk。*out が 256 境界の pointer になる
    Status allocate(std::size_t bytes, std::size_t alignment, void** out);

    /// 切り出し位置を先頭へ戻す。確保済みの容量は保持する。
    ///
    /// フレームの先頭で呼ぶ。これまでに allocate() が返した pointer は
    /// 全て無効になる。容量を解放しないため確保は発生しない。
    ///
    /// @return 無し。
    ///
    /// 入力例: allocate(1024, 256, &p) の後に reset()
    /// 出力例: statistics().used_bytes_ が 0 に戻り、次の allocate が同じ位置を返す
    void reset();

    /// 確保済みの容量を解放する。
    ///
    /// 検出器の破棄時に呼ぶ。destructor からも呼ばれる。二重に呼んでよい。
    /// destructor から呼ばれるため例外を送出せず、解放の失敗も戻り値で
    /// 通知しない。失敗は記録済みの CUDA エラーとして残る。
    ///
    /// @return 無し。
    ///
    /// 入力例: ensure_capacity(4096, MemorySpace::kDevice) の後に release()
    /// 出力例: statistics().capacity_bytes_ が 0 になり、allocate が失敗する
    void release();

    /// 使用状況を返す。
    ///
    /// @return 統計への参照。所有権は workspace に残り、次の操作まで有効。
    ///
    /// 入力例: 100 フレーム分の reset と allocate を繰り返した後
    /// 出力例: allocation_count_ が 1 のまま、reallocation_count_ が 0
    const WorkspaceStatistics& statistics() const { return this->statistics_; }

    /// 現在の memory 空間を返す。
    ///
    /// @return 直近の ensure_capacity で指定した空間。未確保の場合の値は
    ///         意味を持たない。
    ///
    /// 入力例: ensure_capacity(4096, MemorySpace::kHostPinned) の後
    /// 出力例: MemorySpace::kHostPinned
    MemorySpace space() const { return this->space_; }

private:
    void* base_ = nullptr;
    std::size_t capacity_bytes_ = 0;
    std::size_t offset_bytes_ = 0;
    MemorySpace space_ = MemorySpace::kDevice;
    WorkspaceStatistics statistics_;
};

/// 境界へ切り上げた値を返す。
///
/// @param value 対象。
/// @param alignment 境界。2 の冪である必要がある。0 を渡すと value を返す。
/// @return alignment の倍数で value 以上の最小値。桁溢れする場合は 0 を返す。
///
/// 所有権: 資源を保持しない。
/// 同期動作: host 専用であり同期点を持たない。
///
/// 入力例: value = 100、alignment = 256
/// 出力例: 256
std::size_t align_up(std::size_t value, std::size_t alignment);

}  // namespace aruco3cuda

#endif  // ARUCO3CUDA_WORKSPACE_HPP
