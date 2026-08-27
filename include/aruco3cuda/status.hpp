// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_STATUS_HPP
#define ARUCO3CUDA_STATUS_HPP

namespace aruco3cuda {

/// 公開 API の結果状態。
///
/// core は例外を送出しない。destructor、CUDA callback、device code から
/// 例外が伝播する構成を避けるため、失敗は全てこの値で通知する。
enum class Status : int {
    kOk = 0,
    kInvalidArgument,        ///< pointer、寸法、index 等が範囲外
    kInvalidImage,           ///< 画像 view の pointer、寸法、stride、memory 空間が不正
    kInvalidConfig,          ///< 設定値が範囲外、または相互に矛盾
    kUnsupportedDictionary,  ///< 未対応の Dictionary
    kCandidateOverflow,      ///< 候補数が上限を超えた。結果は打ち切られている
    kMarkerOverflow,         ///< 検出数が上限を超えた。結果は打ち切られている
    kCudaError,              ///< CUDA API または kernel 起動の失敗
    kNotInitialized,         ///< 初期化前に呼び出された
};

/// Status を識別子文字列へ変換する。log と test の判定に使用する。
///
/// @param status 変換対象。未知の値でも nullptr を返さない。
/// @return 静的記憶域を持つ文字列。
///
/// 所有権: 戻り値は静的記憶域を指す。呼出側は解放も変更もしない。
/// 同期動作: host 専用であり同期点を持たない。再入可能である。
///
/// 入力例: Status::kCudaError
/// 出力例: "kCudaError"
const char* to_string(Status status);

/// 直近に記録された CUDA エラーの説明を返す。
///
/// CUDA API 名、device、処理段階、CUDA 側の error 文字列を含む。
/// thread ごとに独立して保持する。まだ記録が無い場合は空文字列を返す。
///
/// @return thread_local な記憶域を指す文字列。次の CUDA エラー記録まで有効。
///
/// 所有権: 戻り値は library 側の記憶域を指す。呼出側は解放しない。
///         内容は同じ thread で次に CUDA エラーが記録されると上書きされる。
/// 同期動作: host 専用であり同期点を持たない。thread ごとに独立する。
///
/// 入力例: cudaMalloc の失敗を check_cuda が記録した直後に呼ぶ
/// 出力例: "api=cudaMalloc stage=run_device_self_test device=0 stream=(nil) error=..."
const char* last_cuda_error_message();

}  // namespace aruco3cuda

#endif  // ARUCO3CUDA_STATUS_HPP
