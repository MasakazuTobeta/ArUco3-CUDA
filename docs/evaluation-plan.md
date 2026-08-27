# 評価計画

## 目的

CUDA 実装の正確性、性能、memory 使用量、移植性を再現可能な条件で評価し、CPU、CUDA、ハイブリッドの適用境界を示します。

## 対象範囲

DGX Spark GB10、Jetson Orin、OpenCV ArUco3 CPU 実装、開発する CUDA 実装を対象とします。

## 現状

CPU 経路と hybrid 経路の測定 harness、合成 corpus 生成器、CPU 基準 runner は作成済みです。完全 GPU 経路 (`CUDA-E2E`、`CUDA-Resident`) の測定と実画像データセットは未作成です。

両経路を測った結果は [benchmark 報告](benchmark-report.md) にあります。測定区間は検出のみであり、画像の読み込みと checksum は含みません。1 反復ごとに読み込むと、合成 corpus の 1280x720 PNG では測定区間の 58% から 85% が PNG の復号になり、検出時間の比較として成立しないためです。

分位点は nearest-rank 法で求めます。補間しないため、返る値は必ず実測値のいずれかになります。集計方法が実装依存になると環境をまたいだ比較が成立しないため、この方法を標準とします。

## 目標

### 比較経路

| ID | 経路 | 測定範囲 |
| --- | --- | --- |
| CPU | OpenCV ArUco3 | `cv::Mat` 入力から結果取得まで |
| CUDA-E2E | host 入力 CUDA | upload、検出、download、同期を含む |
| CUDA-Resident | device 入力 CUDA | GPU 常駐画像から device 結果まで |
| Hybrid | CUDA + CPU | 各段階と同期を個別計測 |

DGX Spark GB10 は `integrated` を 1 と報告し、host と device が同一物理 memory を共有します。Jetson Orin も同じ構成です。このため `CUDA-E2E` と `CUDA-Resident` の差は discrete GPU より小さくなることが見込まれます。上記の経路に加えて、入力 buffer の memory 種別を独立した測定軸として記録します。

| 記号 | memory 種別 | 備考 |
| --- | --- | --- |
| M-Pageable | pageable host memory | 最も一般的な呼び出し方 |
| M-Pinned | page-locked host memory | 明示的な copy あり |
| M-Managed | managed memory | 明示的な copy なし。同期と cache の費用は残る |
| M-Device | device 常駐 | 上流処理が GPU 上にある場合 |

得られた crossover point は統合 GPU 環境の結果であり、discrete GPU へ一般化できません。benchmark report へこの制約を明記します。

### 入力条件

- 解像度: 640x480、1280x720、1920x1080、3840x2160
- 画像形式: 8-bit grayscale
- マーカー数: 0、1、4、16、上限近傍
- マーカー辺長: 8、16、32、64、128 pixel 以上
- 条件: 回転、射影歪み、ぼけ、noise、照度差、部分遮蔽、画像境界
- Dictionary: 最初の対応範囲を決定後に固定する

### 正確性指標

- precision / recall
- false positive 数
- ID と rotation の一致率
- 四隅座標の RMSE と最大誤差
- マーカー辺長、角度、条件別の検出率
- CPU と CUDA の差異件数および差異画像

OpenCV CPU 結果は互換性の基準ですが、必ずしも ground truth ではありません。合成データでは生成時の ground truth、実画像では注釈結果を併用します。

### 性能指標

- `T_kernel`: CUDA event で測定する検出処理時間
- `T_end_to_end`: 入力準備から host 結果取得までの wall-clock
- latency: p50、p95、p99
- throughput: 連続処理時の frame/s
- peak device memory とフレームごとの allocation 数
- CPU 使用率、GPU 使用率、対象機で取得可能な消費電力

### 測定手順

1. hardware、OS、CUDA、driver、compiler、OpenCV、power mode、clock を記録する。CPU の core 構成、測定に使用した core、ASLR の状態も記録する。
1a. CPU を core 種別で固定する。DGX Spark GB10 は Cortex-X925 (性能) と Cortex-A725 (効率) の混成であり、固定しないと同じ条件で 1.64 倍の差が出る。`--cpu-list` で指定し、どの core を使ったかを結果へ残す。
2. 同じ入力と detector parameters を固定する。ArUco3 検出戦略の設定は、縮小率 `fxfy` の実効値も併せて記録する。`minMarkerLengthRatioOriginalImg` の既定値は 0.0 であり、この場合 `useAruco3Detection` を有効にしても縮小は発生しない。
3. 初期化と memory allocation を測定区間から分離する。
4. warm-up 後に十分な回数を測定する。
5. 外れ値を削除せず、集計方法と全分布を保存する。
5a. 同一条件を独立した process として複数回実行し、実行間ばらつきを報告する。1 回の実行内の分位点は process ごとの memory 配置による変動を捉えない。全解像度の CPU 経路では ASLR だけで p50 が 9% 動く。前後比較で変化を切り分けたい場合は `setarch -R` で ASLR を無効にし、その旨を記録する。
6. CPU が速い条件を含め、crossover point を求める。
7. CPU 基準の測定値が桁違いに外れていないかを確認する。OpenCV Issue #27118 の報告者は環境と設定が不明な参考値として 640x480 で約 50 ms、1920x1080 で約 150 ms を挙げている。これは合格基準ではなく、測定条件を疑うための sanity check として扱う。

### 合格条件

- 正確性テストが対象条件で安定して再現する。
- Compute Sanitizer で memory error と race が検出されない。
- DGX Spark と Jetson Orin の両方で同一 test corpus が通る。
- 性能結果に測定範囲と同期点が明記されている。
- CUDA の優位条件と非優位条件を説明できる。

## 成果物

- machine-readable な環境情報と測定結果
- 集計表とグラフ
- CPU / CUDA 差異画像
- 再現 command
- benchmark report

大容量の画像や動画は Git repository へ直接 commit せず、保存先と checksum を manifest へ記録します。

## 未確定事項

- warm-up 回数と測定反復数。
- 測定時に GPU clock を固定するか、既定のまま測るか。CPU の core 固定と ASLR の扱いは決定済み。使用する Jetson Orin model は AGX Orin Developer Kit、power mode は MAXN で確定した。
- CPU thread 数を固定するか、各環境の既定値を使用するか。
- 実画像データセットの入手・配布条件。
- 許容する四隅座標誤差と性能改善率の数値基準。

## 関連

- [実装計画](implementation-plan.md)
- [検出パイプライン設計](design/detector-pipeline.md)
- [ADR-0002: build 基盤と対象環境の baseline を固定する](adr/0002-toolchain-and-target-baseline.md)
- [OpenCV Issue #27118](https://github.com/opencv/opencv/issues/27118)
