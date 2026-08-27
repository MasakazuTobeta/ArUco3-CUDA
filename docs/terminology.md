# 日本語用語辞書

文書、PR、レビューで使用する表現を統一するための辞書です。

| 避けたい表現 | 推奨表現 | 補足 |
| --- | --- | --- |
| GPU化すれば高速 | CUDA 実装の有効条件を評価する | 高速化を前提にしない。 |
| 転送なし | 明示的な copy なし | ユニファイドメモリでも同期や cache の費用は残る。 |
| 処理時間 | カーネル時間 / end-to-end 時間 | 測定範囲を明記する。 |
| 正解 | CPU 基準結果 / ground truth | どちらを指すか明記する。 |
| ArUco3 マーカー | ArUco3 検出戦略 | ArUco3 は新しい Dictionary ではなく高速検出方式を指す。 |
| GPUマシン | CUDA 対応環境 | hardware 条件を具体化する。 |
| ゼロコピー | zero-copy | 使用 API と memory 種別を併記する。 |
| CPU fallback | CPU fallback | 機能縮退または小規模入力向け選択を指す。 |
| crossover | crossover point | CPU と CUDA の優位性が切り替わる条件。 |
| artifact | 成果物 | ログ、測定結果、可視化画像等。 |

## 技術名

CUDA、OpenCV、ArUco、ArUco3、DGX Spark、Jetson Orin、Compute Capability、CMake、C++ は正式名称または一般的な表記を使用します。
