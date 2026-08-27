# 実装計画

## 目的

[ロードマップ](roadmap.md) の Phase 定義を、着手可能な作業単位、成果物、完了条件、依存関係へ分解し、設計・実装・評価を同じ前提で進められる状態にします。

## 対象範囲

Phase 0 から Phase 4 までの作業単位、repository 構成、開発 container 環境、検証戦略、評価計画への追加提案、リスクを対象とします。Phase 5 の upstream 提案作業は判断条件のみを扱います。日程と担当は対象外です。

## 現状

- Phase 0 の全作業単位 WP-0.0 から WP-0.7 が完了しました。開発 container、CMake build 基盤、`ctest`、Compute Sanitizer 経路、format と静的解析、CPU 基準 runner、合成 corpus 生成器、Dictionary 変換 tool、benchmark harness が DGX Spark と Jetson AGX Orin の両方で動作します。
- core は build 基盤の疎通確認に必要な最小構成のみです。CUDA 検出器、adapter、benchmark harness、データセットはありません。
- 開発機は DGX Spark GB10 です。実測値と build baseline の決定案は [ADR-0002](adr/0002-toolchain-and-target-baseline.md) にまとめています。
- Jetson Orin 実機はこの環境から参照できていません。JetPack version と power mode は未確認です。
- 開発機に OpenCV、`clang-format`、`ninja` が install されていません。`compute-sanitizer` は使用できます。これらは host へ直接 install せず、開発 container 側で供給します。
- Docker 28.3.3、Docker Compose v2.39.1、NVIDIA Container Toolkit 1.18.0 が利用可能で、`nvidia` runtime が登録済みです。
- 開発 container 環境は `dgx-spark` profile を構築済みで、CUDA Toolkit を image へ固定する `pinned` mode と host から mount する `mounted` mode の両方で環境検査と smoke test が合格しています。設計は [Docker 環境設計](design/docker-environment.md) を参照してください。
- 互換対象である OpenCV 4.x の ArUco3 経路の観測仕様は [検出パイプライン設計](design/detector-pipeline.md) に記録済みです。

### 着手前の blocker

| ID | blocker | 解消方法 | 影響する作業単位 |
| --- | --- | --- | --- |
| B1 | OpenCV が開発機に無い | WP-0.0 の image へ 4.14.0 を commit 固定で build して含める。`dgx-spark` profile で解消済み | WP-0.2 以降のほぼ全て |
| B2 | Jetson Orin の環境が未確認 | 実機の JetPack、CUDA、power mode を確認し ADR-0002 の未確定事項を閉じる。解消済み | WP-0.7、WP-4.4 |
| B3 | `clang-format` が無い | WP-0.0 の `dgx-spark` profile へ含める。解消済み | WP-0.1 |
| B4 | test corpus と ground truth が無い | WP-0.4 で生成器を作る | Phase 1 以降の正確性評価 |

B1 と B3 を host への直接 install ではなく container で解消するのは、DGX Spark と Jetson Orin で同じ手順を使い、測定結果へ環境情報を機械可読形式で残すためです。

B2 は WP-0.7 で解消しました。実機は Jetson AGX Orin Developer Kit、L4T R35.4.1 (JetPack 5.1.2)、CUDA 11.4 です。CUDA Toolkit の最低 version を 11.4 とする決定は [ADR-0002](adr/0002-toolchain-and-target-baseline.md) に記録しました。

## 目標

### 全体構成

```mermaid
flowchart TD
    F0["Phase 0 基盤と基準"] --> F1["Phase 1 ハイブリッド最小成立版"]
    F1 --> F2["Phase 2 GPU 候補抽出"]
    F2 --> F3["Phase 3 GPU decode"]
    F3 --> F4["Phase 4 最適化と評価"]
    F4 --> F5["Phase 5 提案判断"]
    F0 -.-> G0{"G0 基準結果を保存できる"}
    F1 -.-> G1{"G1 差異を分類できる"}
    F2 -.-> G2{"G2 候補抽出が GPU 常駐"}
    F3 -.-> G3{"G3 検出が GPU 常駐"}
    F4 -.-> G4{"G4 評価成果物が再現可能"}
```

### repository 構成案

```
ArUco3-CUDA/
├── CMakeLists.txt             作成済み
├── CMakePresets.json          作成済み。profile ごとの architecture を切り替える
├── cmake/                     作成済み。build option、警告、Compute Sanitizer target
├── docker/                    開発 container。profile ごとの image と環境検査 script
├── include/aruco3cuda/        公開 header。OpenCV へ依存しない
│   └── opencv/                adapter の公開 header
├── src/
│   ├── core/                  作成済み。kernel と host 制御。段階ごとに file を分ける
│   ├── util/                  作成済み。CUDA にも OpenCV にも依存しない共通処理
│   ├── dictionary/            作成済み。packed table と照合。generated/ は生成物
│   ├── dictionary/            packed table と loader
│   └── adapter/opencv/        cv::Mat / cv::cuda::GpuMat 変換
├── tools/
│   ├── dictgen/               作成済み。OpenCV から packed codeword を生成する
│   ├── corpusgen/             作成済み。ground truth 付き合成画像生成器
│   └── report/                差異分類と集計 script
├── reference/                 作成済み。OpenCV CPU 基準 runner
├── bench/                     作成済み。測定 harness と集計 script
├── test/
│   ├── unit/                  型、設定検証、kernel 単体
│   ├── conformance/           Dictionary 互換
│   ├── differential/          CPU 基準との差分
│   └── robustness/            異常入力、境界値、overflow
├── data/manifest/             データセットの所在と checksum のみ
└── docs/
```

大容量の画像と動画は repository へ commit せず、`data/manifest/` に保存先と checksum を記録します。

### 作業単位

`規模` は相対的な目安であり、日程ではありません。日程は [ロードマップ](roadmap.md) の未確定事項のままです。

#### Phase 0: 基盤と基準

| ID | 内容 | 成果物 | 完了条件 | 依存 | 規模 |
| --- | --- | --- | --- | --- | --- |
| WP-0.0 | 開発 container。CUDA Toolkit の供給を pinned と mounted から選べる 2 profile、環境検査 script、環境情報記録 script、環境 smoke test | `docker/**`、[Docker 環境設計](design/docker-environment.md) | `dgx-spark` profile の image が build でき、`verify-environment.sh` と `smoke-test.sh` が合格する。`jetson-orin` profile は WP-0.7 で確認する | - | M |
| WP-0.1 | CMake project、C++17、`sm_87` と `sm_121`、warning、`clang-format`、静的解析、`ctest` 骨格 | `CMakeLists.txt`、`CMakePresets.json`、`cmake/**`、`.clang-format`、`.clang-tidy` | 空 library と smoke test が両 architecture 向けに build でき `ctest` が通る。達成済み | WP-0.0 | M |
| WP-0.2 | OpenCV 4.14.0 の commit 固定 build と provenance 記録 | `docker/scripts/build-opencv.sh`、image 内 provenance JSON | 再現可能に build でき、version と commit が環境情報 JSON へ出力される | WP-0.0 | S |
| WP-0.3 | CPU 基準 runner | `reference/**`、`aruco3cuda_reference_runner` | 同一入力に対し決定的な ID と四隅、および環境情報を保存できる。達成済み | WP-0.2 | M |
| WP-0.4 | 合成 corpus 生成器 | `tools/corpusgen`、manifest | seed 固定で再生成でき、四隅の ground truth を持つ corpus が得られる。達成済み | WP-0.2 | M |
| WP-0.5 | Dictionary 変換 tool と互換テスト | `tools/dictgen`、`src/dictionary`、生成 table | [Dictionary 方針](dictionaries.md) の検証 1 から 5 が自動テストで通る。達成済み | WP-0.2 | M |
| WP-0.6 | benchmark harness 骨格 | `bench/**`、`aggregate.py` | CPU 経路の p50・p95・p99 と環境情報を機械可読形式で保存できる。達成済み | WP-0.3、WP-0.0 | M |
| WP-0.7 | Jetson Orin profile の実機検証 | `docker/.env` の既定値確定、[ADR-0002](adr/0002-toolchain-and-target-baseline.md) の更新 | Jetson 実機で image が build でき、環境検査が合格し、CUDA Toolkit の最低 version が確定する。達成済み | B2、WP-0.0 | M |

G0 の完了条件: 同一入力に対する CPU 基準結果と環境情報を保存でき、CUDA 側の空実装が両 profile の container 内で build とテストを通ること。**達成済み**。

規約準拠の作業後に P0 の検証を全項目やり直し、結果が変わっていないことを確認しました。

| 検証項目 | 再検査の結果 |
| --- | --- |
| 環境検査 | 両機で全項目合格 |
| build | DGX Spark で `sm_87` と `sm_121`、Jetson で `sm_87` |
| `ctest` | 両機 155 件。Compute Sanitizer 込みで 159 件 |
| smoke test の四隅 | `(399.57, 259.57)` と `(559.43, 419.43)`。両機一致、当初の記録と一致 |
| `fxfy` と segmentation | 0.3333、427x240。当初の記録と一致 |
| corpus の checksum | `clean_1280x720_n1_s128` が `ac79179c8e8b`。当初の記録と一致 |
| Dictionary の生成物 | `dictgen --check` が byte 単位で一致 |
| CPU 経路の測定値 | Jetson は一致。DGX Spark の ArUco3 無効のみ変化。原因は測定条件の固定不足であり、下記に記録した |

corpus の manifest と benchmark の JSONL は `schema_version` を 2 へ上げました。manifest は単位を名前へ示す規約に合わせて key を改名したため破壊的変更です。JSONL は測定条件の項目を追加したためです。

WP-0.7 で判明し修正した、実機でしか現れない問題を記録します。

| 問題 | 症状 | 対応 |
| --- | --- | --- |
| 環境検査が `nvidia-smi` に依存していた | Jetson には `nvidia-smi` が無く、正常な環境で不合格になる | device 確認を CUDA runtime API の probe へ置き換えた |
| Tegra の device node に権限が無い | `libcuda.so.1` は見えるのに `cudaGetDeviceCount` が `NvRmMemInitNvmap failed with Permission denied` で失敗する | 非 root 実行の container へ video と debug の補助 GID を付与した |
| `nvpmodel` が container に無い | 評価計画が必須とする power mode が空のまま記録される | 状態 file と定義 file を mount し、取得元の差を script へ集約した |
| Docker が `/sys/firmware` を mask する | 基板名を取得できない。Orin は AGX と NX で性能が大きく異なる | device tree の model file を個別に mount した |
| `jetson-orin` の test preset が無い | `ctest --preset jetson-orin` が preset 一覧を表示して終わる | 全 configure preset に build と test preset を対応させた |

WP-0.1 の検証結果は次のとおりです。`libaruco3cuda.a` に `sm_87` と `sm_121` の両方の code が含まれることを `cuobjdump --list-elf` で確認しています。

| 確認項目 | 結果 |
| --- | --- |
| `sm_87` と `sm_121` の同時 build | 成功 |
| `ctest` | 12 件全て合格 |
| Compute Sanitizer 4 mode を含む `ctest` | 16 件全て合格 |
| `format-check` | 差分なし |
| `clang-tidy` | 指摘なし |

WP-0.3 の検証結果は次のとおりです。

| 確認項目 | 結果 |
| --- | --- |
| 合成マーカーの検出 | ID と四隅が描画位置と 1 pixel 以内で一致 |
| 決定性 | `--omit-timing` 指定時、同一入力から byte 単位で同じ JSON |
| ArUco3 の縮小率 | 1280x720、`S=32`、`tau_i=0.05` で `fxfy=0.3333`、segmentation 427x240 |
| ArUco3 の効果 | 同一画像・同一検出結果で 5.917 ms から 1.112 ms |
| `ctest` | 35 件全て合格。Compute Sanitizer を含めて 39 件 |

`rotation` は OpenCV の `detectMarkers` が独立した値として返さないため、出力へ含めていません。marker rotation は四隅の並び順として表現されるため、比較は四隅の順序で行います。

WP-0.4 と WP-0.5 の検証結果は次のとおりです。

| 確認項目 | 結果 |
| --- | --- |
| corpus の決定性 | 同じ seed から manifest と画像 checksum が一致 |
| ground truth の妥当性 | CPU 基準実装の検出結果と四隅が 1.5 pixel 以内で一致 |
| Dictionary 適合性 | 250 ID 全 4 回転の codeword が OpenCV の `bytesList` と一致 |
| Dictionary 訂正境界 | bit 反転に対する accept / reject が OpenCV の `identify()` と一致 |
| 最小 Hamming 距離 | 再計算値 12。公称値と一致 |
| 生成物の再現性 | `dictgen --check` が既存 file と byte 単位で一致 |
| `ctest` | 61 件全て合格。Compute Sanitizer を含めて 65 件 |

WP-0.6 と WP-0.7 の実測結果は次のとおりです。1280x720、マーカー 4 個、OpenCV thread 1、CPU 経路、遅延 200 回、独立した process として 5 回実行した結果です。CPU は性能 core へ固定しています。

| 機体 | 条件 | fxfy | p50 中央値 | p50 範囲 | p95 | p99 | throughput |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DGX Spark GB10 | ArUco3 有効 (`tau_i = 0.05`) | 0.3333 | 2.198 ms | 2.198 - 2.202 (0.2%) | 2.204 ms | 2.206 ms | 455 frame/s |
| DGX Spark GB10 | ArUco3 無効 | 1.0000 | 7.035 ms | 6.998 - 7.169 (2.4%) | 7.647 ms | 7.714 ms | 142 frame/s |
| Jetson AGX Orin | ArUco3 有効 (`tau_i = 0.05`) | 0.3333 | 4.387 ms | 4.382 - 4.389 (0.1%) | 4.423 ms | 4.492 ms | 228 frame/s |
| Jetson AGX Orin | ArUco3 無効 | 1.0000 | 13.432 ms | 13.392 - 13.476 (0.6%) | 13.695 ms | 14.080 ms | 74 frame/s |

検出結果は全条件で 4 個と一致します。ArUco3 検出戦略の効果は DGX Spark で 3.2 倍、Jetson AGX Orin で 3.1 倍です。同一条件での機体差は Jetson を基準として DGX Spark が約 0.5 倍の時間です。Jetson の power mode は MAXN、GPU 最大 clock は 1300 MHz です。

これらは CPU 基準側の値であり、CUDA 経路との比較は Phase 1 以降に行います。

#### 測定条件を固定しないと値が再現しない

規約準拠の作業後に P0 の測定値を再検査したところ、DGX Spark の ArUco3 無効の値が当初記録した 7.570 ms から動きました。原因を調べた結果、追加した検証処理ではなく測定条件の固定不足でした。追加した `validate_config` の費用は 1 回あたり 20.7 ns であり、7 ms の検出に対して 0.0003% です。

判明した要因は 2 つです。

**CPU core の種別**: DGX Spark GB10 は Cortex-X925 が 10 個 (cpu 5-9、15-19、最大約 4.0 GHz) と Cortex-A725 が 10 個 (cpu 0-4、10-14、最大約 2.86 GHz) の混成です。同じ条件でも効率 core へ割り当てられると 1.64 倍遅くなります。core を固定しない測定は、割り当て次第で二極化します。Jetson AGX Orin は Cortex-A78AE 12 個の均一構成であり、この影響を受けません。当初の Jetson の値が再測定でも一致したのはこのためです。

**address space の無作為化 (ASLR)**: 全解像度を扱う CPU 経路では、process ごとの memory 配置の違いだけで p50 が 9% 変動します。`setarch -R` で無効化すると 8 回の実行が完全に同じ値になることを確認しました。1 回の実行内の分位点はこの変動を捉えません。

対応として、benchmark harness へ `--cpu-list` を追加し、CPU の core 構成、実際の親和性、ASLR の状態を測定結果へ記録するようにしました。`aggregate.py` は同一条件の複数実行をまとめ、実行間ばらつきを表示します。

合成マーカーの検出四隅は DGX Spark と Jetson AGX Orin で一致しました。CPU 基準実装の結果が機体に依存しないことを示します。

ArUco3 が検出対象とする最小辺長は `tau_i` そのものではなく `S + L * tau_i` で決まります。`S` は `minSideLengthCanonicalImg`、`L` は画像の長辺です。1280x720、`S = 32`、`tau_i = 0.05` の場合は 96 pixel が下限であり、実測では 96 pixel は検出されず 128 pixel は検出されました。corpus の manifest は各マーカーの `side_ratio` を記録し、どの `tau_i` で検出対象になるかを後から判断できるようにしています。

#### Phase 1: ハイブリッド最小成立版

WP-1.5 の案 C ハイブリッド経路は、GPU で pyramid・segmentation・適応的二値化を行い、二値化画像と pyramid を host へ戻して、候補抽出から decode までを CPU で行います。合成 12 場面 x 設定 3 通り (ArUco3 有効、OpenCV 既定、OpenCV 既定 + subpixel 補正) の計 36 比較で、CPU 基準との差は未検出 0、過検出 0、四隅の最大差 0.0000 pixel でした。DGX Spark と Jetson AGX Orin の双方で同じ結果です。

四隅が一致するには、近接候補の grouping を OpenCV と同じにする必要がありました。適応的二値化は window を 3 通り試すため、同じマーカーから少しずつ違う候補が得られます。OpenCV は識別の前に候補を周長の降順へ並べ、近接するものを 1 グループにまとめ、グループ内で最大周長の候補を代表に選びます。当初はこの grouping を省き識別後に ID と位置で重複を落としていましたが、最小 window 由来の小さい候補が残るため四隅が内側へ寄り、原寸へ戻した時点で 7.9 pixel の差になりました。OpenCV と同じ grouping、包含関係の木、親候補の識別打ち切りまで実装した結果、差は 0 になりました。この作業は当初 WP-2.5 に置いていたものを前倒ししたものです。

あわせて、OpenCV が `useAruco3Detection` 有効時に補正方法の指定を `CORNER_REFINE_SUBPIX` へ上書きすることを実装へ反映し、ArUco3 無効時の subpixel 補正 (window をセル 1 辺から決める経路) も実装しました。

WP-1.1 では [公開 API 草案](design/public-api.md) の未確定事項を 3 件解決しました。公開 aggregate の field にも末尾 `_` を付けること、検証は `Status` を返し理由を任意の out 引数で受け取ること、画像の失敗に `kInvalidImage` を割り当てることです。

WP-1.4 の適応的二値化は OpenCV の `adaptiveThreshold` と完全に一致しました。3 通りの window と 3 種類の画像寸法、5 種類の定数のいずれでも不一致は 0 です。行方向と列方向へ分けて合計するため中間で丸めが入らず、2 次元の総和と同じ値になります。

あわせて WP-1.3 で受け入れた resize の 1 階調差の下流影響を実測しました。1280x720 を 427x240 へ縮小して二値化した場合、画素の白黒が入れ替わる割合は 0.039% から 0.054% です。無作為な 1 階調の付加を仮定した見積もり 0.45% の 10 分の 1 に収まりました。実際の差は構造を持ち、局所平均も同じ方向へ動くためです。

WP-1.3 では pyramid が OpenCV の `buildPyramid` と全 level で完全一致しました。segmentation は最大 1 階調の差が残ります。OpenCV の 8-bit `INTER_LINEAR` が `softdouble` と `ufixedpoint16` による bit exact 経路であり、kernel 内での再現にこの 2 つの数値型の移植が必要になるためです。差の影響と扱いは [検出パイプライン設計](design/detector-pipeline.md) に記録しました。

WP-1.2 の workspace は bump pointer 方式の arena です。段階ごとの buffer をここから切り出し、フレームの先頭で `reset()` を呼びます。`allocate()` は容量が足りなくても自動で拡張しません。自動拡張はフレームごとの確保を招き、規約が避けよと定める状態を静かに作るためです。容量は `ensure_capacity()` で初期化時に確保します。

完了条件は統計で確認します。100 フレーム分の `reset()` と切り出しを繰り返しても `allocation_count_` が 1 のまま、`reallocation_count_` と `exhausted_count_` が 0 のままであることをテストで固定しました。`MemorySpace` を引数に取るため、評価計画が求める device、pinned host、managed の 3 経路を同じ arena の実装で測れます。

| ID | 内容 | 成果物 | 完了条件 | 依存 | 規模 |
| --- | --- | --- | --- | --- | --- |
| WP-1.1 | 公開型、設定、`validate()` | `include/aruco3cuda/{types,config}.hpp`、`src/core/{types,config}.cpp` | 設定の矛盾と不正な画像 view を境界で拒否するテストが通る。達成済み | WP-0.1 | S |
| WP-1.2 | workspace 所有と再確保統計 | `include/aruco3cuda/workspace.hpp`、`src/core/workspace.cpp` | フレームごとの確保が発生しないことをテストで確認できる。達成済み | WP-1.1 | M |
| WP-1.3 | S1 pyramid と S2 segmentation の kernel | `src/core/preprocess.{hpp,cu}` | OpenCV の縮小結果との差が定めた許容内に収まる。達成済み | WP-1.2 | M |
| WP-1.4 | S3 適応的二値化 kernel | `src/core/threshold.{hpp,cu}` | 3 通りの window size で CPU 基準の二値化と一致率が許容内。完全一致を達成 | WP-1.3 | M |
| WP-1.5 | 案 C ハイブリッド経路 | `hybrid/hybrid_detector.{hpp,cpp}` | 合成画像の基本条件で ID と四隅を取得できる。36 比較で四隅の差 0.0000 pixel を達成 | WP-1.4、WP-0.3 | M |
| WP-1.6 | 差分レポート tool | `tools/report` | 差異を未検出・過検出・ID 不一致・rotation 不一致・四隅ずれへ分類できる | WP-1.5 | S |

G1 の完了条件: 合成画像の基本条件で ID と四隅を取得でき、CPU 基準結果との差異が分類できること。

#### Phase 2: GPU 候補抽出

| ID | 内容 | 成果物 | 完了条件 | 依存 | 規模 |
| --- | --- | --- | --- | --- | --- |
| WP-2.1 | S4 連結成分ラベリング | `src/core` | 既知 label 画像に対し CPU 実装と label 集合が一致する | WP-1.4 | L |
| WP-2.2 | label 統計の集計 | `src/core` | bbox、pixel 数、重心が CPU 集計と一致する | WP-2.1 | S |
| WP-2.3 | S5 極点探索による四隅推定 | `src/core` | 合成マーカーで四隅を許容誤差内に推定できる | WP-2.2 | L |
| WP-2.4 | S6 フィルタと compaction と overflow | `src/core` | 上限超過時に `kCandidateOverflow` を返し結果を打ち切ることをテストで確認できる | WP-2.3 | M |
| WP-2.5 | 近接候補の統合 | `src/core` | CPU 基準の grouping との差異を分類できる。CPU 版は WP-1.5 で実装済みのため、残りは GPU 側への移設 | WP-2.4 | S |
| WP-2.6 | 案 A と案 C の比較 spike | 比較レポート、ADR | 差異と速度の両面から主案を決定し ADR へ記録する | WP-2.5、WP-1.6 | M |

G2 の完了条件: 候補抽出まで GPU 常駐で完結し、CPU 基準との差異を分類でき、案の選択が ADR に記録されていること。

#### Phase 3: GPU decode

| ID | 内容 | 成果物 | 完了条件 | 依存 | 規模 |
| --- | --- | --- | --- | --- | --- |
| WP-3.1 | S7 射影変換とセル sampling | `src/core` | 既知 homography で正しいセル値を得られる | WP-2.6 | M |
| WP-3.2 | S8 前半 Otsu と border 検証 | `src/core` | `minOtsuStdDev` と `maxErroneousBitsInBorderRate` の境界で CPU と判定が一致する | WP-3.1 | M |
| WP-3.3 | S8 後半 Dictionary 照合 | `src/core` | 全 ID と 4 回転で CPU と同じ ID・rotation・距離を返す | WP-3.2、WP-0.5 | M |
| WP-3.4 | S9 重複整理 | `src/core` | 重複入力に対し CPU と同じ代表候補を選ぶか、差異を説明できる | WP-3.3 | M |
| WP-3.5 | S10 四隅の subpixel 補正と upsampling | `src/core` | 四隅 RMSE が CPU 基準に対する許容内に収まる | WP-3.4 | L |
| WP-3.6 | device 常駐出力 API | `DeviceDetections` | host 同期なしで検出結果を参照できることをテストで確認できる | WP-3.5 | S |

G3 の完了条件: ID と四隅の出力まで GPU 経路で完結し、`CUDA-Resident` 経路が測定可能であること。

#### Phase 4: 最適化と評価

| ID | 内容 | 成果物 | 完了条件 | 依存 | 規模 |
| --- | --- | --- | --- | --- | --- |
| WP-4.1 | kernel 起動数と stream の最適化 | `src/core` | 最適化前後で正確性が不変であり、変化を測定値で示せる | WP-3.6 | M |
| WP-4.2 | memory 経路 4 種の実装と測定 | `src/core`、`bench` | pageable、pinned、managed、device 常駐を別結果として記録できる | WP-3.6 | M |
| WP-4.3 | 機種別 tuning の設定化 | 設定 file | block size 等が source の固定値でなく設定から上書きできる | WP-4.1 | S |
| WP-4.4 | Jetson Orin での全評価 | 測定結果 | 同一 corpus が通り、環境情報と power mode が記録されている | B2、WP-4.3 | L |
| WP-4.5 | crossover point の報告 | benchmark report | CPU が有利な条件を含めて境界を示せる | WP-4.4 | M |
| WP-4.6 | Compute Sanitizer 全経路 | CI target | memcheck、racecheck、initcheck、synccheck で指摘が無い | WP-4.1 | M |

G4 の完了条件: [評価計画](evaluation-plan.md) の成果物が揃い、再現手順で同じ結果が得られること。

### 推奨する着手順

```mermaid
flowchart LR
    W00["WP-0.0 container"] --> W01["WP-0.1 build 基盤"]
    W00 --> W03["WP-0.3 CPU 基準"]
    W00 --> W04["WP-0.4 corpus"]
    W00 --> W05["WP-0.5 dictgen"]
    W00 --> W07["WP-0.7 Jetson 検証"]
    W01 --> W11["WP-1.1 型と設定"]
    W03 --> W16["WP-1.6 差分 tool"]
    W04 --> W16
    W11 --> W13["WP-1.3 pyramid"]
    W13 --> W14["WP-1.4 二値化"]
    W14 --> W15["WP-1.5 ハイブリッド"]
    W16 --> W15
    W15 --> W21["WP-2.1 ラベリング"]
```

critical path は WP-0.0 から WP-0.1、WP-1.1、WP-1.3、WP-1.4、WP-1.5 を経て WP-2.1 へ至る経路です。WP-0.0 が全ての起点になるため最優先で着手します。WP-0.3 から WP-0.5、および WP-0.7 は WP-0.0 の完了後に並行して進められます。WP-0.2 は WP-0.0 の image build 手順そのものへ統合されます。

### 検証戦略

| 層 | 対象 | 実行頻度 |
| --- | --- | --- |
| unit | 型、設定検証、host 側の境界処理、kernel 単体 | 全 commit |
| conformance | Dictionary の ID 数、markerSize、4 回転 codeword、訂正境界 | 全 commit |
| differential | CPU 基準結果との ID・rotation・四隅の比較 | 全 commit |
| robustness | 0 マーカー、上限超過、非連続 pitch、ROI、極小画像、null pointer、memory 空間の不一致 | 全 commit |
| cli | CLI の引数解析。正常系、異常系、境界値を実行 file の起動で検証 | 全 commit |
| doc | 公開 API の Doxygen 要件の充足を機械検査 | 全 commit |
| sanitizer | Compute Sanitizer の 4 mode | 日次または PR |
| coverage | C0 と C1 の測定と未達理由の確認 | Phase 完了時 |
| benchmark | カーネル時間と end-to-end 時間、latency 分布 | 変更時と Phase 完了時 |

`tools/check_doxygen.py` が公開ヘッダの宣言を列挙し、規約が求める 7 要素 (目的、引数、戻り値、所有権、同期動作、入力例、出力例) の欠落を検出します。`ctest` から実行するため、宣言を追加したときに記載漏れへ気付けます。

所有権と同期動作が class 全体で共通する場合は、class の Doxygen へ「全ての public member 関数に適用される」と明記することで member 側の記載に代えます。同一の記述を member ごとに複写すると、規約が避けよと定める「処理の言い換え」に近い冗長さを生み、可読性を損なうためです。検査もこの扱いに従います。

Compute Sanitizer の実行では、意図的に CUDA API を失敗させる test を suite 名で除外します。Compute Sanitizer は意図の有無に関わらず全ての API エラーを報告するため、除外しないと意図した失敗が指摘として現れ、本物の問題を埋もれさせます。

差分テストは CPU 基準結果を互換性の基準として扱い、ground truth は合成 corpus の生成時情報を使用します。両者が食い違う場合は、差異を分類したうえで CPU 基準側の挙動として記録します。

CUDA device code は host coverage と別に、入力分割と境界値で実行経路を確認します。対象は、overflow 経路、`minOtsuStdDev` 未満の候補、border 誤り上限、回転 4 種、pyramid の最上位と最下位 level です。

### カバレッジの現状と未達理由

`coverage` preset で C0 と C1 を測定します。`cmake --build build/coverage --target coverage-report` が `ctest` を実行してから `gcovr` で集計します。

| 指標 | 現状 | 目標 |
| --- | --- | --- |
| C0 (line) | 89.8% (2515 / 2802) | 100% |
| C1 (decision) | 81.8% (737 / 901) | 100% |
| function | 99.4% (155 / 156) | 100% |

規約は 100% を目標とし、未達の場合は対象外理由の記録を求めます。未到達 287 行の内訳は次のとおりです。

| 分類 | 内容 | 扱い |
| --- | --- | --- |
| 外部資源の獲得失敗 | `popen` の失敗、`ofstream` の書き込み失敗、`cudaMalloc` の失敗 | 対象外。故障注入の仕組みが無いと再現できない。仕組みの導入は Phase 4 で検討する |
| OpenCV 内部の失敗 | `cv::imwrite` の失敗、`getPerspectiveTransform` の異常入力 | 対象外。OpenCV 側の内部状態に依存し、外から誘発できない |
| CUDA API の失敗経路 | `cudaGetDeviceProperties` や `cudaMemcpy` の失敗 | 対象外。`check_cuda` の記録経路自体は `test_cuda_check.cpp` で検証済み |
| 到達不能な防御的分岐 | `percentile_nearest_rank` の順位下限、列挙に無い値の既定処理 | 対象外。仕様上到達しないが、将来の変更に対する防御として残す |
| CUDA device code | `fill_squares_kernel`、`invert_kernel` | gcov は device code の実行経路を計測できない。規約の定めどおり入力分割と境界値で確認する。自己診断の要素数を block size の倍数にしないことで、範囲外判定の分岐も実行される |
| 候補 grouping の代替経路 | `close_contours_` を使う再識別 | 対象外。代表候補の識別が失敗し、かつ同一 group に離れた候補が残る場面を合成で作れていない。実機 corpus の導入時に再評価する |
| 同一 ID の並び替え | 検出結果の整列で ID が等しい場合の比較 | 対象外。合成 corpus は 1 場面に同じ ID を置かない |

`main.cpp` の CLI 層は `test/cli/` の ctest から実行 file を起動して測定に含めています。未到達分は上記の外部資源の失敗経路が中心です。

CUDA 経路の実装が入る Phase 1 以降、この表を更新します。

### 評価計画へ反映した変更

[評価計画](evaluation-plan.md) へ次の 3 点を反映済みです。理由をここへ記録します。

#### 1. memory 経路を測定軸へ加える

DGX Spark GB10 は `integrated` を 1 と報告し、host と device が同一物理 memory を共有します。Jetson Orin も同様の構成です。このため `CUDA-E2E` と `CUDA-Resident` の差は、discrete GPU の場合より小さくなることが見込まれます。現在の 4 経路に加えて、pageable host、pinned host、managed、device 常駐の 4 通りを独立した測定軸として記録します。

あわせて、得られた crossover point は統合 GPU 環境の結果であり、discrete GPU へは一般化できないことを benchmark report へ明記します。OpenCV への提案時には、利用者の多くが discrete GPU を使う点が判断材料になります。

#### 2. ArUco3 の既定値に関する測定条件を明記する

OpenCV の `minMarkerLengthRatioOriginalImg` の既定値は 0.0 であり、この場合 `useAruco3Detection` を有効にしても縮小率は 1 になります。ArUco3 検出戦略の効果を測るには、この値を明示的に設定した条件で測定する必要があります。測定条件へ縮小率 `fxfy` の実効値を記録する項目を追加しました。

#### 3. 外部報告値を sanity check の基準に使う

OpenCV Issue #27118 の報告者は、CPU 実装の処理時間として 640x480 で約 50 ms、1920x1080 で約 150 ms を挙げています。これは環境と設定が不明な参考値ですが、こちらの CPU 基準測定が桁違いに離れた場合は測定条件を疑う材料になります。基準値としてではなく sanity check として記録します。

### リスクと対策

| ID | リスク | 影響 | 対策 |
| --- | --- | --- | --- |
| R1 | 案 A の四隅推定が CPU 基準と一致せず、差異が許容できない | Phase 2 の手戻り | 案 C を fallback として維持し、WP-2.6 で判断する |
| R2 | 統合 GPU では転送費用が小さく、CPU 有利の条件が広い | 成功判定 2 を満たせない | CPU が有利な条件も成果物として報告する。crossover point の提示自体を成果と位置付ける |
| R3 | 四隅精度が subpixel 補正の実装差で劣化する | 成功判定 1 を満たせない | WP-3.5 を最重要作業とし、pyramid level ごとの誤差を分解して記録する |
| R4 | Jetson Orin の環境差で build または測定が再現しない | 成功判定 3 を満たせない | B2 を Phase 0 の間に解消し、環境情報を測定結果へ必ず埋め込む |
| R5 | Dictionary data の由来を説明できない状態で取り込む | ライセンス上の問題 | WP-0.5 で hash と変換手順を [Code Provenance 記録](code-provenance.md) へ残す |
| R6 | 測定条件の差で有利な結果だけが残る | 評価の信頼性低下 | 外れ値を削除せず全分布を保存し、条件を機械可読形式で併記する |
| R8 | compiler が image と分離していると測定結果を後から再現できない | 成功判定 3 を満たせない | CUDA Toolkit を image へ固定する `pinned` mode を既定とし、version を環境情報へ記録する |
| R7 | patent clearance が未実施のまま商用検討が進む | 事業判断の遅延 | [知的財産・ライセンス方針](ip-and-licensing.md) の手順を Phase 4 完了までに開始する |

## 未確定事項

- 各作業単位の担当、開始日、完了日。
- 案 A の妥当性検証しきい値と、CPU 基準との候補差異の許容範囲。
- 四隅座標誤差と性能改善率の合格基準の数値。[評価計画](evaluation-plan.md) の未確定事項と同じ。
- CUDA Toolkit の最低 version。[ADR-0002](adr/0002-toolchain-and-target-baseline.md) の未確定事項と同じ。
- CI を実行する環境。開発機、Jetson 実機、外部 runner のどれを使うか。
- corpus に実画像を含める時期と、その入手・配布条件。

## 関連

- [ロードマップ](roadmap.md)
- [アーキテクチャ](architecture.md)
- [検出パイプライン設計](design/detector-pipeline.md)
- [公開 API 草案](design/public-api.md)
- [Docker 環境設計](design/docker-environment.md)
- [評価計画](evaluation-plan.md)
- [ADR-0002: build 基盤と対象環境の baseline を固定する](adr/0002-toolchain-and-target-baseline.md)
