# benchmark_harness

## 目的

測定条件と実行環境を結果と一体で記録し、後から再現・比較できる形で性能を残します。CPU、CUDA、ハイブリッドを同じ枠組みで測り、CPU が有利な条件も含めて crossover point を示せるようにします。

## 対象範囲

遅延と throughput の測定、統計の算出、環境情報の収集、JSONL 出力を対象とします。完全 GPU 経路の実装は Phase 3 以降で追加します。可視化は `aggregate.py` の責務とします。

## 現状

- 実装があるのは `CPU` 経路と `Hybrid` 経路です。
- `CUDA-E2E` と `CUDA-Resident` は識別子だけ定義し、指定すると未実装として失敗します。GPU decode (Phase 3) の完了後に追加します。
- memory 種別は `M-Device` と `M-Pageable` を実装しています。`M-Pinned` と `M-Managed` は WP-4.2 です。
- kernel 時間 (CUDA event) はどの経路でも未測定です。段階時間 (`stages`) は wall-clock であり、host 同期を含みます。両者を区別するため、段階時間で `kernel` を埋めることはしません。
- 測定した結果は [benchmark 報告](../docs/benchmark-report.md) にあります。

## 実装上の判断

### 測定区間に画像の読み込みを含めない

`CPU` 経路は読み込み済みの画像に対して検出だけを繰り返します。1 反復ごとに `cv::imread` と `sha256_file` を呼ぶと、合成 corpus の 1280x720 PNG では測定区間の 58% から 85% が PNG の復号になります。実時間処理では PNG を復号しないため、検出時間の比較として成立しません。

この変更で `p50` の値は従来より小さくなります。`schema_version` を 3 へ上げ、version 2 以前の結果と混ぜて集計できないようにしています。

### hybrid の memory 種別で測定区間を変える

`M-Device` は画像が既に device にある想定で転送を測定区間の外へ置きます。camera から GPU へ直接入る構成の上限にあたります。`M-Pageable` は host の画像を毎 frame 転送し、転送を測定区間へ含めます。`CPU` 経路と同じく host の画像から始める場合の値です。

### 未実装の経路を CPU で代替しない

`--route CUDA-E2E` のように未実装の経路を指定した場合、無言で CPU 経路へ読み替えず失敗します。読み替えると結果の `route` と実際に測った処理が食い違い、後から気付けません。

### 経路と memory 種別を独立した軸にする

DGX Spark と Jetson Orin はいずれも統合 GPU であり、明示的な copy の費用が discrete GPU と大きく異なります。`M-Pageable`、`M-Pinned`、`M-Managed`、`M-Device` を経路とは別の軸として記録します。識別子は [評価計画](../docs/evaluation-plan.md) の表記に揃えます。

### 遅延と throughput を分けて測る

遅延は 1 フレームずつ独立に測り、throughput は連続処理の総時間から求めます。両者は別の指標であり、片方から他方を換算しません。

### kernel 時間を 0 で埋めない

CPU 経路には kernel 時間が存在しません。JSON では `null` とし、0 を書きません。0 を書くと集計時に「非常に速い kernel」と誤読されます。

### 測定条件を core 種別まで固定する

DGX Spark GB10 は Cortex-X925 (性能) と Cortex-A725 (効率) の混成です。同じ条件でも割り当て先の core 種別で 1.64 倍の差が出るため、`--cpu-list` で固定できるようにしています。core 構成と実際の親和性は結果へ記録します。Jetson AGX Orin は Cortex-A78AE 12 個の均一構成であり、この影響を受けません。

### 実行間ばらつきを 1 回の実行内の分位点と混同しない

全解像度を扱う CPU 経路では、process ごとの memory 配置 (ASLR) の違いだけで p50 が 9% 変動します。`setarch -R` で無効化すると実行間で完全に一致することを確認しました。1 回の実行内の p50、p95、p99 はこの変動を捉えません。

このため ASLR の状態を結果へ記録し、`aggregate.py` が同一条件の複数実行をまとめて実行間ばらつきを表示します。測定は独立した process として複数回行います。

### 外れ値を除去しない

統計は全標本から算出します。分位点は nearest-rank 法で求め、補間しません。返る値は必ず実測値のいずれかになります。集計方法が実装依存になると、環境をまたいだ比較が成立しません。詳細は `aruco3cuda::util::compute_statistics` を参照してください。

`--save-samples` を指定すると全標本を結果へ含められます。分布そのものを保存する必要がある場合に使用します。

### 縮小率を測定条件へ必ず記録する

ArUco3 の実効縮小率 `fxfy` を条件として記録します。`minMarkerLengthRatioOriginalImg` の既定値は 0.0 であり、この場合 `useAruco3Detection` を有効にしても縮小が発生しません。この値が残らないと、ArUco3 の効果を測ったのかどうかを後から判別できません。

### 環境情報を可能な限り library から取得する

GPU 名、Compute Capability、統合 GPU かどうかは CUDA から取得します。`nvidia-smi` が無い container でも記録できます。driver version と Jetson の power mode は library から取得できないため外部 command を使い、取得できない場合は空文字列のままとします。推測で埋めません。

## 目標

- CUDA 経路を追加し、CUDA event による kernel 時間と wall-clock を分離して記録する。
- peak device memory とフレームごとの allocation 数を記録する。
- 解像度、マーカー数、辺長を掃引した測定を 1 回の実行で行えるようにする。
- clock と power mode を固定した状態での測定手順を確立する。

## 関連

- [評価計画](../docs/evaluation-plan.md)
- [実装計画](../docs/implementation-plan.md)
- [reference_runner](../reference/reference_runner.md)
