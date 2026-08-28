# Benchmark 報告: CPU 基準と hybrid 経路

- 測定日: 2026-08-28
- 対象: Phase 2 完了時点の実装

## 目的

[評価計画](evaluation-plan.md) が定める crossover point を、現時点で実装がある 2 経路について求めます。CUDA 化に投資を続ける価値があるかを、Phase 3 (GPU decode) へ進む前に判断するための材料です。

## 対象範囲

`CPU` 経路 (OpenCV ArUco3) と `Hybrid` 経路 (GPU で前処理と二値化、CPU で候補抽出と decode) を比較します。`CUDA-E2E` と `CUDA-Resident` は未実装のため対象外です。

## 測定した区間

**検出のみです。画像の読み込みと checksum は含みません。**

この点は本報告以前の測定と異なります。従来 `measure_image` は 1 反復ごとに `cv::imread` と `sha256_file` を呼んでおり、測定区間の 58% から 85% が PNG の復号でした。

| 画像 | imread | sha256 | detectMarkers | 検出以外の割合 |
| --- | --- | --- | --- | --- |
| clean_1280x720_n1_s128.png | 1.834 ms | 0.031 ms | 1.168 ms | 61.5% |
| clean_1280x720_n4_s128.png | 1.941 ms | 0.036 ms | 1.403 ms | 58.5% |
| degraded_1280x720.png | 8.875 ms | 4.893 ms | 2.465 ms | 84.8% |

実時間処理では PNG を復号しません。検出時間を比べる測定として成立していなかったため、読み込みを初期化側へ移しました。JSONL の `schema_version` を 3 へ上げています。version 2 以前の結果は同じ key が違う区間を指すため、混ぜて集計できません。

## 測定条件

| 項目 | 値 |
| --- | --- |
| 反復 | 暖機 30 回、遅延 200 回、throughput 100 frame |
| OpenCV thread | 1 |
| ASLR | 有効。container の権限では `setarch -R` を使えない |
| 独立実行 | 3 回。実行間ばらつきを併記する |
| 分位点 | nearest-rank。補間しない |
| corpus | 合成 corpus の `full` preset から 9 場面 |
| Dictionary | DICT_ARUCO_MIP_36h12 |
| ArUco3 | 有効。`minSideLengthCanonicalImg` = 32、`minMarkerLengthRatioOriginalImg` = 0.05 |

### 固定する CPU core

**性能 core へ固定します。** 性能 core と効率 core が混在する機では、どちらで測るかで値が 2 倍変わります。

| 機体 | core 構成 | 測定に使う CPU |
| --- | --- | --- |
| DGX Spark GB10 | Cortex-X925 x10 (性能)、Cortex-A725 x10 (効率) | 5 (X925) |
| RTX 5070 Ti の機 | 4.80GHz x2、4.70GHz x6 (性能)、4.60GHz x12 (効率) | 2 (4.70GHz) |
| Jetson AGX Orin | Cortex-A78AE x12 (均一) | 0 |

DGX Spark と RTX の機はいずれも CPU 0 が異なる種別です。DGX Spark の CPU 0 は効率 core、RTX の機の CPU 0 は性能 core です。番号だけで揃えると別種の core を比べることになります。

本報告の初版は DGX Spark を CPU 0 (効率 core) で測っており、GPU 側の優位を過大に評価していました。差の大きさは [core 種別の影響](#core-種別の影響) に示します。

再現手順は [測定の再現](#測定の再現) にあります。生の結果は `docs/measurements/` にあります。

### memory 種別

| 種別 | 測定区間 |
| --- | --- |
| `M-Device` | 画像は既に device にある。転送は測定区間の外 |
| `M-Pageable` | host の画像を毎 frame 転送する。転送は測定区間の内 |

`M-Pinned` と `M-Managed` は未実装です (WP-4.2)。

## 対象機

| 機体 | GPU | CC | GPU 種別 | CPU |
| --- | --- | --- | --- | --- |
| DGX Spark GB10 | NVIDIA GB10 | 12.1 | 統合 | Cortex-X925 / A725 |
| RTX 5070 Ti の機 | GeForce RTX 5070 Ti | 12.0 | **単体** | Intel Core Ultra 7 265 |
| Jetson AGX Orin | Orin | 8.7 | 統合 | Cortex-A78AE |

## 結果

`p50` の中央値 (独立実行 3 回)。比は CPU を 1 とします。1 未満なら hybrid が速いことを示します。

### DGX Spark GB10 (統合 GPU、性能 core)

| 場面 | CPU | Hybrid M-Device | 内 GPU 段 | 内 CPU 段 | 比 | M-Pageable | 比 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 640x480 マーカー 4 | 0.554 | 0.512 | 0.309 | 0.205 | 0.93 | 0.622 | 1.12 |
| 1280x720 マーカー 1 | 0.603 | 0.466 | 0.394 | 0.071 | 0.77 | 0.515 | 0.86 |
| 1280x720 マーカー 4 | 0.700 | 0.640 | 0.461 | 0.176 | 0.91 | 0.719 | 1.03 |
| 1280x720 マーカー 16 | 1.169 | 1.145 | 0.463 | 0.671 | 0.98 | 1.310 | 1.12 |
| 1920x1080 マーカー 4 | 0.982 | 0.797 | 0.557 | 0.241 | 0.81 | 0.897 | 0.91 |
| 3840x2160 マーカー 4 | 1.722 | 1.186 | 0.947 | 0.245 | **0.69** | 1.652 | 0.96 |
| ぼけ | 0.584 | 0.447 | 0.393 | 0.053 | 0.77 | 0.502 | 0.86 |
| noise | 2.699 | 2.670 | 0.466 | 2.200 | 0.99 | 2.833 | 1.05 |
| 複合劣化 | 1.567 | 1.488 | 0.418 | 1.069 | 0.95 | 1.680 | 1.07 |

### GeForce RTX 5070 Ti (単体 GPU、性能 core)

| 場面 | CPU | Hybrid M-Device | 内 GPU 段 | 内 CPU 段 | 比 | M-Pageable | 比 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 640x480 マーカー 4 | 0.376 | 0.352 | 0.144 | 0.208 | 0.94 | 0.368 | 0.98 |
| 1280x720 マーカー 1 | 0.513 | 0.456 | 0.386 | 0.070 | 0.89 | 0.499 | 0.97 |
| 1280x720 マーカー 4 | 0.612 | 0.563 | 0.386 | 0.177 | 0.92 | 0.605 | 0.99 |
| 1280x720 マーカー 16 | 1.149 | 1.117 | 0.386 | 0.727 | 0.97 | 1.150 | 1.00 |
| 1920x1080 マーカー 4 | 0.899 | 0.648 | 0.406 | 0.242 | **0.72** | 0.745 | 0.83 |
| 3840x2160 マーカー 4 | 1.743 | 1.378 | 1.154 | 0.224 | 0.79 | 1.688 | 0.97 |
| ぼけ | 0.487 | 0.427 | 0.386 | 0.041 | 0.88 | 0.469 | 0.97 |
| noise | 2.648 | 2.629 | 0.435 | 2.192 | 0.99 | 2.690 | 1.02 |
| 複合劣化 | 1.431 | 1.362 | 0.390 | 0.972 | 0.95 | 1.401 | 0.98 |

### Jetson AGX Orin (統合 GPU、均一 core)

| 場面 | CPU | Hybrid M-Device | 内 GPU 段 | 内 CPU 段 | 比 | M-Pageable | 比 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 640x480 マーカー 4 | 1.269 | 2.406 | 1.812 | 0.589 | 1.90 | 2.131 | 1.68 |
| 1280x720 マーカー 4 | 1.685 | 3.371 | 2.839 | 0.536 | 2.00 | 3.159 | 1.88 |
| 1920x1080 マーカー 4 | 2.614 | 4.017 | 3.271 | 0.731 | 1.54 | 4.338 | 1.66 |
| 3840x2160 マーカー 4 | 5.749 | 7.865 | 6.931 | 0.906 | 1.37 | 8.628 | 1.50 |
| noise | 5.854 | 6.971 | 2.085 | 4.882 | 1.19 | 7.126 | 1.22 |

### 実行間ばらつき

CPU を固定した状態で、独立 process 3 回の `p50` の幅は 0.1% から 1.1% です。ASLR を無効化できていないにもかかわらず小さく収まりました。

## 読み取れること

### 1. hybrid の優位は限定的である

性能 core で測ると、DGX Spark と RTX 5070 Ti のいずれも比は 0.69 から 0.99 に収まります。**最良でも 1.45 倍**であり、多くの条件では 1.1 倍未満です。前処理と二値化だけを GPU へ移した段階としては妥当ですが、CUDA 化の投資を正当化するには不足します。

### 2. 効きが出るのは高解像度に限られる

両機とも 1920x1080 と 3840x2160 で比が 0.69 から 0.79 へ下がります。低解像度と劣化の強い場面では 0.9 以上です。GPU 段が解像度にあまり依らない一方、CPU 段は場面の内容に強く依存するためです。

### 3. GPU 段は解像度でほとんど変わらない

RTX 5070 Ti では 640x480 の 0.144 ms を除き、1280x720 から 1920x1080 まで 0.386 から 0.406 ms で一定です。画素数が 4.5 倍でも変わりません。kernel 起動の固定費が支配的であることを示します。3840x2160 でようやく 1.154 ms へ増えます。

DGX Spark も同様に 0.39 から 0.56 ms で一定です。**この固定費を下げることが、Phase 4 で最も効く最適化になります。**

### 4. Jetson Orin では全条件で CPU が速い

比は 1.19 から 2.00 です。GPU 段が他 2 機の 5 倍以上を要します (1280x720 で 2.839 ms 対 0.39 から 0.46 ms)。解像度を上げると比は 2.00 から 1.37 へ改善しますが、逆転はしません。

### 5. 単体 GPU でも転送費用は小さい

RTX 5070 Ti で `M-Device` と `M-Pageable` の差は 1280x720 で 0.042 ms、3840x2160 で 0.310 ms です。比にすると 0.92 が 0.99 へ、0.79 が 0.97 へ動きます。**転送を含めるとほぼ優位が消えます。**

意外なことに、統合 GPU である DGX Spark の方が転送の影響が大きく出ています (1280x720 で 0.079 ms、比 0.91 が 1.03 へ)。PCIe の帯域が効く単体 GPU より不利になる理由は特定できていません。統合 GPU では copy engine を使わず CPU が転送を行う経路になっている可能性がありますが、確認していません。

### 6. hybrid の GPU 段には Phase 3 で消える費用が含まれる

現在の hybrid は毎 frame、pyramid の全 level と二値化画像 3 枚を host へ戻します。decode が CPU にあるためであり、GPU decode が入れば不要になります。したがってこの測定は GPU 経路の下限であり、上限ではありません。

### core 種別の影響

DGX Spark を効率 core (Cortex-A725) と性能 core (Cortex-X925) の両方で測りました。

| 場面 | 効率 core CPU | 効率 core 比 | 性能 core CPU | 性能 core 比 |
| --- | --- | --- | --- | --- |
| 1280x720 マーカー 4 | 1.377 ms | 0.76 | 0.700 ms | 0.91 |
| 3840x2160 マーカー 4 | 3.816 ms | 0.46 | 1.722 ms | 0.69 |
| 640x480 マーカー 4 | 1.052 ms | 1.03 | 0.554 ms | 0.93 |

CPU 経路は core 種別で約 2 倍変わります。GPU 段も kernel 起動が CPU 側の処理であるため 0.69 ms から 0.46 ms へ下がります。結果として**効率 core で測ると GPU の優位が実際より大きく見えます**。3840x2160 では 2.17 倍と 1.45 倍で、結論の印象が変わる差です。

実運用で検出を効率 core へ固定することは考えにくいため、性能 core の値を本報告の主結果とします。

## 判断

**Phase 3 (GPU decode) へ進む根拠は弱まりました。ただし進める価値はあります。**

- 性能 core で測ると、前処理と二値化を GPU へ移した効果は最良で 1.45 倍、多くの条件で 1.1 倍未満です。転送を含めるとほぼ消えます。
- 一方、GPU 段の内訳を見ると kernel 起動の固定費が支配的です。1280x720 で GPU 段 0.386 ms のうち、画素数に比例する分は 640x480 との差から 0.24 ms 程度と見積もれます。固定費を削れば GPU 段は半分以下になりえます。
- CPU 段 (0.04 から 2.2 ms) は Phase 3 で GPU へ移る対象です。noise を含む場面では CPU 段が 2.2 ms を占めており、ここが GPU 化の主な余地です。
- Jetson Orin では GPU 段が支配的で、decode を移すと GPU 段はさらに増えます。**Orin での成立は kernel 起動の削減が前提**です。

Phase 3 と Phase 4 の順序を入れ替え、先に kernel 起動の固定費を削ることも選択肢になります。判断は別途 ADR へ記録します。

## 未確定事項

- kernel 起動の固定費をどこまで下げられるか。window 3 通りの融合と CUDA Graph が候補。
- 統合 GPU で転送費用が単体 GPU より大きく出る理由。
- `M-Pinned` と `M-Managed` の値。
- CUDA event による kernel 時間と wall-clock の分離 (WP-4.1)。現在の段階時間は wall-clock であり、host 同期を含む。
- Jetson の CUDA Toolkit は 11.4 (JetPack 5.x)、他 2 機は 13.0。toolkit version の影響は切り分けていない。
- 実写 corpus での測定。現在は合成 corpus のみ。
- 4K を超える解像度。Jetson の crossover がその先にあるかを確かめられていない。

## 測定の再現

```
# container 内で実行する。<cpu> は性能 core の番号。
#   DGX Spark GB10 -> 5   RTX 5070 Ti の機 -> 2   Jetson AGX Orin -> 0
./build/<preset>/tools/corpusgen/aruco3cuda_corpusgen --preset full --seed 20260827 \
  --output-dir /tmp/benchcorpus --manifest /tmp/benchcorpus/manifest.json

B=./build/<preset>/bench/aruco3cuda_bench
COMMON="--warmup 30 --latency-iterations 200 --throughput-frames 100 --cpu-list <cpu> --threads 1"
IMGS="--input /tmp/benchcorpus/clean_640x480_n4_s128.png \
      --input /tmp/benchcorpus/clean_1280x720_n4_s128.png \
      --input /tmp/benchcorpus/clean_1920x1080_n4_s256.png \
      --input /tmp/benchcorpus/clean_3840x2160_n4_s256.png \
      --input /tmp/benchcorpus/clean_1280x720_n1_s128.png \
      --input /tmp/benchcorpus/clean_1280x720_n16_s128.png \
      --input /tmp/benchcorpus/blur_1280x720.png \
      --input /tmp/benchcorpus/noise_1280x720.png \
      --input /tmp/benchcorpus/combined_1280x720.png"

for run in 1 2 3; do
  taskset -c <cpu> $B $IMGS $COMMON --route CPU    --memory-mode N/A        >> results.jsonl
  taskset -c <cpu> $B $IMGS $COMMON --route Hybrid --memory-mode M-Device   >> results.jsonl
  taskset -c <cpu> $B $IMGS $COMMON --route Hybrid --memory-mode M-Pageable >> results.jsonl
done
python3 bench/aggregate.py results.jsonl
```

実機への同期は `tools/sync-to-host.sh <user@host>` を使います。

## 関連

- [評価計画](evaluation-plan.md)
- [実装計画](implementation-plan.md)
- [ADR-0002: build 基盤と対象環境の baseline を固定する](adr/0002-toolchain-and-target-baseline.md)
- [ADR-0003: 四角形候補抽出は案 A を主案とする](adr/0003-candidate-extraction-approach.md)
- [測定 harness](../bench/benchmark_harness.md)
