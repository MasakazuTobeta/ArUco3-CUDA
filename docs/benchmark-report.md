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
| 起動の費用 | 別に測定。1 process 1 画像で 3 回 |
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

### 起動の費用

warm-up 後の分位点には現れない費用です。1 process 1 画像で測りました。1280x720 マーカー 4 枚、`M-Device`。

| 機体 | CPU 1 枚目まで | Hybrid 1 枚目まで | CUDA 文脈生成 | 追加費用 | 1 frame の利得 | 相殺に要する frame |
| --- | --- | --- | --- | --- | --- | --- |
| DGX Spark GB10 | 3.19 ms | 201.9 ms | 16.7 ms | 215.3 ms | 0.134 ms | **約 1600** |
| RTX 5070 Ti | 2.18 ms | 71.2 ms | 73.4 ms | 142.4 ms | 0.014 ms | **約 10200** |
| Jetson AGX Orin | 6.24 ms | 64.1 ms | 17.8 ms | 75.7 ms | 負 | **相殺しない** |

内訳は機体で異なります。DGX Spark は 1 枚目の検出そのものが 22.2 ms かかり、kernel の読み込みが大きく出ます。RTX 5070 Ti は文脈生成が 73.4 ms と大きい一方、1 枚目の検出は 1.45 ms に収まります。

30fps の動画とすると、DGX Spark で約 54 秒、RTX 5070 Ti で約 5 分 42 秒の連続処理で元が取れます。Jetson Orin は定常状態でも CPU が速いため、いくら回しても相殺しません。

**単発の検出や短い burst では、CPU 経路が桁違いに速くなります。** 1 枚だけ処理する用途では 2 ms 対 200 ms です。

### 実行間ばらつき

CPU を固定した状態で、独立 process 3 回の `p50` の幅は 0.1% から 1.1% です。ASLR を無効化できていないにもかかわらず小さく収まりました。

## 読み取れること

### 1. hybrid の優位は限定的である

性能 core で測ると、DGX Spark と RTX 5070 Ti のいずれも比は 0.69 から 0.99 に収まります。**最良でも 1.45 倍**であり、多くの条件では 1.1 倍未満です。前処理と二値化だけを GPU へ移した段階としては妥当ですが、CUDA 化の投資を正当化するには不足します。

### 2. 効きが出るのは高解像度に限られる

両機とも 1920x1080 と 3840x2160 で比が 0.69 から 0.79 へ下がります。低解像度と劣化の強い場面では 0.9 以上です。GPU 段が解像度にあまり依らない一方、CPU 段は場面の内容に強く依存するためです。

### 3. GPU 段の大半は host への転送であり、kernel 実行ではない

GPU 段を「kernel 実行」と「host への転送」へ分けて測りました。1280x720、`M-Device`。

| 機体 | kernel 実行のみ | host への転送 | GPU 段 合計 | 転送の割合 |
| --- | --- | --- | --- | --- |
| DGX Spark GB10 | 0.083 ms | 0.269 ms | 0.357 ms | 75.2% |
| RTX 5070 Ti | 0.033 ms | 0.353 ms | 0.386 ms | **91.6%** |
| Jetson AGX Orin | 0.928 ms | 1.258 ms | 2.184 ms | 57.6% |

転送量は 1499 KB を 8 回の同期転送で戻しています。内訳は pyramid 5 level (level 0 は原寸) と二値化画像 3 枚です。**decode が CPU にあるためだけに発生する費用であり、GPU decode が入れば不要になります。**

kernel 実行そのものは DGX Spark で 0.083 ms、RTX 5070 Ti で 0.033 ms です。前処理と二値化に GPU が要する時間は、CPU 経路全体 (0.61 から 0.70 ms) の 5% から 14% にすぎません。

Jetson Orin の kernel 実行は 0.928 ms で、他 2 機の 11 倍から 28 倍です。GPU の能力差は実在します。ただし転送を除いた 0.928 ms は、Orin の CPU 経路 1.685 ms の 55% です。

### 4. Jetson Orin では全条件で CPU が速い

比は 1.19 から 2.00 です。GPU 段が他 2 機の 5 倍以上を要します (1280x720 で 2.839 ms 対 0.39 から 0.46 ms)。解像度を上げると比は 2.00 から 1.37 へ改善しますが、逆転はしません。

### 5. 単体 GPU でも転送費用は小さい

RTX 5070 Ti で `M-Device` と `M-Pageable` の差は 1280x720 で 0.042 ms、3840x2160 で 0.310 ms です。比にすると 0.92 が 0.99 へ、0.79 が 0.97 へ動きます。**転送を含めるとほぼ優位が消えます。**

意外なことに、統合 GPU である DGX Spark の方が転送の影響が大きく出ています (1280x720 で 0.079 ms、比 0.91 が 1.03 へ)。PCIe の帯域が効く単体 GPU より不利になる理由は特定できていません。統合 GPU では copy engine を使わず CPU が転送を行う経路になっている可能性がありますが、確認していません。

### 6. 起動の費用は定常状態の差を大きく上回る

CUDA の文脈生成と kernel の読み込みで、1 枚目の結果が出るまでが CPU 経路の 30 倍から 60 倍になります。定常状態の利得は 1 frame あたり 0.014 ms から 0.134 ms しかないため、相殺には 1600 frame から 10200 frame を要します。

用途によって結論が変わります。連続する動画を扱うなら 1 分程度で元が取れますが、単発の検出や短い burst では CPU 経路が圧倒的に有利です。本報告の定常状態の表だけを見ると、この違いが見えません。

### 7. この測定は GPU 経路の下限であり、上限ではない

転送の内訳から、hybrid は完全 GPU 経路より構造的に不利です。この測定で「GPU が CPU に勝てない」と結論するのは誤りです。

### core 種別の影響

DGX Spark を効率 core (Cortex-A725) と性能 core (Cortex-X925) の両方で測りました。効率 core の生データは schema_version 3 で取得したものであり、起動の費用を含まないため `docs/measurements/` へは残していません。

| 場面 | 効率 core CPU | 効率 core 比 | 性能 core CPU | 性能 core 比 |
| --- | --- | --- | --- | --- |
| 1280x720 マーカー 4 | 1.377 ms | 0.76 | 0.700 ms | 0.91 |
| 3840x2160 マーカー 4 | 3.816 ms | 0.46 | 1.722 ms | 0.69 |
| 640x480 マーカー 4 | 1.052 ms | 1.03 | 0.554 ms | 0.93 |

CPU 経路は core 種別で約 2 倍変わります。GPU 段も kernel 起動が CPU 側の処理であるため 0.69 ms から 0.46 ms へ下がります。結果として**効率 core で測ると GPU の優位が実際より大きく見えます**。3840x2160 では 2.17 倍と 1.45 倍で、結論の印象が変わる差です。

実運用で検出を効率 core へ固定することは考えにくいため、性能 core の値を本報告の主結果とします。

## 判断

**Phase 3 (GPU decode) へ進むべきです。** 定常状態の優位は現時点で限定的ですが、その原因が hybrid という中間形態の構造にあることが測定で分かりました。

- 定常状態では、前処理と二値化を GPU へ移した効果は最良で 1.45 倍、多くの条件で 1.1 倍未満です。転送を含めるとほぼ消えます。
- 起動の費用が定常状態の差を大きく上回ります。相殺には 1600 frame から 10200 frame を要し、Jetson Orin では相殺しません。**連続する動画を扱う用途に限れば意味があり、単発の検出では CPU 経路が桁違いに有利です。**
- **一方、現在の hybrid が遅い主因は kernel ではなく host への転送です。** GPU 段の 58% から 92% を占めており、これは decode が CPU にあるためだけに発生します。Phase 3 で decode を GPU へ移せば消えます。
- kernel 実行そのものは DGX Spark で 0.083 ms、RTX 5070 Ti で 0.033 ms です。CPU 経路全体の 5% から 14% にすぎません。**前処理と二値化に関する限り、GPU は十分に速い**と言えます。
- CPU 段 (0.04 から 2.2 ms) も Phase 3 で GPU へ移る対象です。noise を含む場面では 2.2 ms を占めます。
- Jetson Orin の kernel 実行は 0.928 ms で他 2 機の 11 倍以上です。GPU の能力差は実在します。ただし CPU 経路 1.685 ms の 55% であり、転送が消えれば競合しうる水準です。

**Phase 3 (GPU decode) を先に進めるのが妥当です。** 現在の測定は、hybrid という中間形態が構造的に負担している転送費用に支配されており、GPU 経路の実力を示していません。Phase 4 の kernel 最適化は、転送を消した後の測定を見てから判断すべきです。

## 未確定事項

- 転送を消した場合に GPU 段がどこまで下がるか。kernel 実行だけなら DGX Spark 0.083 ms、RTX 5070 Ti 0.033 ms、Jetson Orin 0.928 ms である。Phase 3 の完了後に測る。
- 8 回の同期転送を非同期化して重ねられるか。現在は 1 回ずつ blocking する。
- 統合 GPU で転送費用が単体 GPU より大きく出る理由。
- `M-Pinned` と `M-Managed` の値。
- CUDA event による kernel 時間と wall-clock の分離 (WP-4.1)。現在の段階時間は wall-clock であり、host 同期を含む。
- Jetson の CUDA Toolkit は 11.4 (JetPack 5.x)、他 2 機は 13.0。toolkit version の影響は切り分けていない。
- 実写 corpus での測定。現在は合成 corpus のみ。
- 起動の費用を減らせるか。CUDA の文脈生成は減らせないが、kernel の読み込みは
  対象 architecture を 1 つに絞る、または cubin を事前に読み込むことで短縮
  できる可能性がある。DGX Spark の 22.2 ms は 3 つの architecture 向け binary を
  含む fatbin から選ぶ費用を含む。
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
- [host と device の間の memory 受け渡し](design/memory-transfer.md)
- [測定 harness](../bench/benchmark_harness.md)
