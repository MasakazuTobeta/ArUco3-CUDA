# Contributing

このファイルは、このリポジトリで作業する人間とエージェントの共通ルールです。

## Base Standard

- C++ コーディング規約は MISRA C++ 2023 を基礎とし、本書の規定をリポジトリ固有の上書きとして扱います。
- CUDA C++ でも、host code と device code の責務、所有権、同期点、エラー境界を明示してください。
- 将来 OpenCV へコントリビュートできるよう、OpenCV のコーディング規約と Apache-2.0 互換性を意識してください。
- 本リポジトリへの contribution は、明示的な例外がない限り Apache License 2.0 で提供されます。
- コピーレフト依存は導入しません。依存追加は Apache-2.0、MIT、BSD 等の permissive license を原則とします。
- 公式 ArUco の GPLv3 source code をコピー、翻案、移植しません。実装前に [知的財産・ライセンス方針](docs/ip-and-licensing.md) を確認してください。

## Development Workflow

1. 変更前に影響範囲、依存関係、リスク、検証方法、更新対象文書を整理する。
2. 新しいアルゴリズムは論文、OpenCV CPU 実装の観測可能な挙動、公開仕様から要件を定義する。
3. 可能な限り、期待動作を固定するテストを先に用意する。
4. CPU 基準実装との正確性比較を通した後に性能最適化を行う。
5. 最適化前後で、正確性、カーネル時間、end-to-end 時間、メモリ使用量を記録する。
6. API、設定、評価条件、対応環境が変わる場合は関連文書を同じ変更で更新する。

## C++ / CUDA Coding Rules

- C++17 以上を使用し、RAII を徹底する。
- raw `new` / `delete` を使用しない。
- 型と class は `PascalCase`、関数とローカル変数は `snake_case`、定数は `kPascalCase`、名前空間は小文字を使用する。
- 継続保持するメンバ変数は末尾 `_` とし、参照時は必ず `this->` を付ける。
- `auto` は型が明白な場合に限り使用する。
- 単位を持つ値は名前または型で単位を示す。例: `elapsed_ms`、`marker_length_px`。
- CUDA カーネル内で動的メモリ確保、例外、暗黙の device-wide synchronization を使用しない。
- CUDA API とカーネル起動のエラーを確認する。非同期処理では、失敗を検出する同期点をテストまたは呼出側に設ける。
- フレームごとの `cudaMalloc` / `cudaFree` を避け、再利用可能な作業領域を使用する。
- `cudaStream_t` を公開 API から扱える設計とし、既定 stream への暗黙依存を避ける。
- `__CUDA_ARCH__` による機種固有分岐は局所化し、共通実装を保つ。
- `clang-format` を適用し、静的解析の警告を放置しない。

## Code Provenance Rules

- 実装根拠にした論文、仕様、permissive license の source code を `docs/ip-and-licensing.md` または実装の sidecar 文書へ記録する。
- 公式 ArUco GPLv3 code の構造、表現、コメント、関数分割を移植しない。
- OpenCV の Apache-2.0 code を改変または一部利用する場合は、対象 file、commit、ライセンス、変更内容を記録し、必要な copyright と notice を保持する。
- ArUco3 検出戦略の実装根拠は、2018 年の論文と Apache-2.0 の OpenCV 4.x に限定する。公式 ArUco GPLv3 source code は、実装、レビュー、最適化、test data 作成の根拠に使用しない。
- 定義済み Dictionary は、version と commit を固定した OpenCV 4.x の `predefined_dictionaries.hpp` を正本とする。公式 ArUco GPLv3 配布物から codeword、table、画像を抽出しない。
- OpenCV の定義済み Dictionary を repository 内へ格納する場合は、取得元 file、commit、license、変換手順、変換前後の hash を記録し、必要な notice を保持する。
- `extendDictionary()` 等で生成した custom Dictionary を `DICT_ARUCO_MIP_36h12` と同一であるかのように扱わない。既定 Dictionary と custom Dictionary は識別子、metadata、test を分離する。
- CPU 実装を互換性 oracle として実行しただけの場合も、使用 version と設定を評価結果へ記録する。
- code provenance が説明できない寄稿は取り込まない。
- 特許の有無は source code license と別に扱う。商用公開前に必要な patent clearance を行う。

## Error Handling And Validation

- public API の画像型、寸法、stride、Dictionary、設定値、pointer、stream を境界で検証する。
- host code の失敗は例外または明示的な状態値で通知し、無言で継続しない。
- destructors、CUDA callback、device code から例外を送出しない。
- CUDA のエラーには API 名、device、stream、処理段階を追跡できる文脈を付ける。
- 外部入力、ファイル、環境変数、データセットを信頼せず、サイズ上限と形式を検証する。

## Comment Rules

- program file 内の説明コメントと Doxygen は原則として日本語で記述する。
- public class と関数には、目的、引数、戻り値、所有権、同期動作、入力例、出力例を記載する。
- コメントは処理の言い換えではなく、設計理由、前提、失敗時の扱い、性能上の意図を説明する。
- CUDA カーネルでは、thread / block とデータの対応、競合回避、境界条件を説明する。

## Testing Rules

- 正常系、異常系、境界値を自動テストする。
- CPU 基準実装と ID、回転、四隅座標、未検出結果を比較する。
- 画像寸法、stride、ROI、マーカーサイズ、Dictionary、歪み、照明、ぼけ、遮蔽の代表条件を含める。
- Compute Sanitizer を使用するテスト経路を設ける。
- C0 および C1 カバレッジ 100% を目標とし、未達の場合は対象外理由を記録する。CUDA device code は host coverage と別に、入力分割と境界値で実行経路を確認する。測定は `cmake --preset coverage` の後 `cmake --build build/coverage --target coverage-report` で行い、現状と未達理由は [実装計画](docs/implementation-plan.md) に記録する。
- 意図的に CUDA API を失敗させる test は suite 名へ `DeliberateError` を含め、Compute Sanitizer の実行から除外できるようにする。
- 性能テストを正確性テストの代用にしない。

## Benchmark Rules

- warm-up 回数、測定回数、解像度、マーカー条件、power mode、clock、CUDA / driver / OpenCV version を記録する。
- `T_kernel`、`T_end_to_end`、単一フレーム遅延、複数フレームのスループットを分離する。
- 平均値だけでなく、中央値、p95、p99 を保存する。
- CPU と CUDA で同じ入力と検出条件を使用する。
- GPU 常駐入力と host 入力からの転送込みを別結果として扱う。
- 有利な結果だけを選択せず、CPU が速い条件を含めて crossover point を報告する。

## Documentation

- 文書は原則として日本語で記述し、`docs/terminology.md` の推奨表現を使用する。
- 仕様・構想では、`現状` と `目標` を明確に分ける。
- C++ の同一 stem の program file 合計が 300 行以上の場合は、隣接する sidecar `*.md` を追加する。
- sidecar には `目的`、`対象範囲`、`現状`、`実装上の判断`、`目標`、`関連` を含める。
- 将来の判断に影響する採用理由と制約は ADR または関連文書へ残す。

## Commit And Review

- commit message は Conventional Commits 形式を使用する。
- 1 commit は 1 つの目的にまとめる。
- PR 本文は原則として日本語で記述し、概要、変更点、検証結果、未実施項目、関連文書を含める。
- secret、認証情報、ビルド生成物、測定用の大容量動画・画像を commit しない。
- 新規 source file には `SPDX-License-Identifier: Apache-2.0` を記載する。
