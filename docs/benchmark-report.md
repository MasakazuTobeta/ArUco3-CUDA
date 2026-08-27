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
| CPU 固定 | `taskset -c 0` かつ `--cpu-list 0` |
| OpenCV thread | 1 |
| ASLR | 有効。container の権限では `setarch -R` を使えない |
| 独立実行 | 3 回。実行間ばらつきを併記する |
| 分位点 | nearest-rank。補間しない |
| corpus | 合成 corpus の `full` preset から 9 場面 |
| Dictionary | DICT_ARUCO_MIP_36h12 |
| ArUco3 | 有効。`minSideLengthCanonicalImg` = 32、`minMarkerLengthRatioOriginalImg` = 0.05 |

再現手順は [測定の再現](#測定の再現) にあります。生の結果は `docs/measurements/` にあります。

### memory 種別

| 種別 | 測定区間 |
| --- | --- |
| `M-Device` | 画像は既に device にある。転送は測定区間の外 |
| `M-Pageable` | host の画像を毎 frame 転送する。転送は測定区間の内 |

`M-Pinned` と `M-Managed` は未実装です (WP-4.2)。

## 結果

### 解像度に対する変化

マーカー 4 枚、劣化なし。`p50` の中央値 (独立実行 3 回) です。比は CPU を 1 とします。

**DGX Spark GB10**

| 解像度 | 辺長 | fxfy | CPU | Hybrid M-Device | 内 GPU 段 | 内 CPU 段 | 比 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 640x480 | 128 px | 0.500 | 1.050 ms | 1.087 ms | 0.667 ms | 0.421 ms | 1.04 |
| 1280x720 | 128 px | 0.333 | 1.365 ms | 1.051 ms | 0.687 ms | 0.359 ms | **0.77** |
| 1920x1080 | 256 px | 0.250 | 2.027 ms | 1.287 ms | 0.796 ms | 0.489 ms | **0.63** |
| 3840x2160 | 256 px | 0.143 | 3.818 ms | 1.772 ms | 1.256 ms | 0.507 ms | **0.46** |

**Jetson AGX Orin**

| 解像度 | 辺長 | fxfy | CPU | Hybrid M-Device | 内 GPU 段 | 内 CPU 段 | 比 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 640x480 | 128 px | 0.500 | 1.268 ms | 2.406 ms | 1.812 ms | 0.589 ms | 1.90 |
| 1280x720 | 128 px | 0.333 | 1.685 ms | 3.368 ms | 2.839 ms | 0.536 ms | 2.00 |
| 1920x1080 | 256 px | 0.250 | 2.617 ms | 3.999 ms | 3.269 ms | 0.722 ms | 1.53 |
| 3840x2160 | 256 px | 0.143 | 5.755 ms | 7.813 ms | 6.896 ms | 0.909 ms | 1.36 |

### 場面の内容に対する変化 (1280x720)

**DGX Spark GB10**

| 場面 | 検出数 | CPU | Hybrid M-Device | 内 GPU 段 | 内 CPU 段 | 比 |
| --- | --- | --- | --- | --- | --- | --- |
| マーカー 1 枚 | 1 | 1.173 ms | 0.841 ms | 0.686 ms | 0.154 ms | 0.72 |
| マーカー 4 枚 | 4 | 1.365 ms | 1.051 ms | 0.687 ms | 0.359 ms | 0.77 |
| マーカー 16 枚 | 16 | 2.397 ms | 2.066 ms | 0.677 ms | 1.388 ms | 0.87 |
| ぼけ | 0 | 1.130 ms | 0.786 ms | 0.674 ms | 0.111 ms | 0.70 |
| noise | 0 | 4.721 ms | 4.387 ms | 0.680 ms | 3.701 ms | 0.93 |
| 複合劣化 | 0 | 2.848 ms | 2.496 ms | 0.682 ms | 1.812 ms | 0.88 |

### 実行間ばらつき

CPU を固定した状態で、独立 process 3 回の `p50` の幅は 0.1% から 1.1% です。ASLR を無効化できていないにもかかわらず小さく収まりました。

## 読み取れること

### 1. GB10 では 640x480 付近に crossover があり、解像度とともに hybrid が有利になる

640x480 では同等 (比 1.04)、1280x720 で 1.30 倍、1920x1080 で 1.59 倍、3840x2160 で 2.16 倍 hybrid が速くなります。

### 2. GB10 の GPU 段は解像度にほとんど依らない

画素数が 27 倍になっても GPU 段は 0.667 ms から 1.256 ms にしか増えません。kernel 起動の固定費が支配的であることを示します。1 frame あたりの kernel 起動は、pyramid と segmentation と二値化 3 window で 20 回前後です。この固定費を下げれば低解像度側の crossover も動きます。

### 3. Jetson Orin では全条件で CPU が速い

比は 1.36 から 2.00 です。GPU 段が GB10 の 4 倍を要しており (1280x720 で 2.839 ms 対 0.687 ms)、CPU 側の差は 1.2 倍しかありません。両機で GPU と CPU の力関係が大きく違います。

ただし解像度を上げると比は 2.00 から 1.36 へ改善します。傾向は crossover が 4K の先にあることを示しますが、外挿であり測定していません。

### 4. hybrid の GPU 段には Phase 3 で消える費用が含まれる

現在の hybrid は毎 frame、pyramid の全 level と二値化画像 3 枚を host へ戻します。decode が CPU にあるためであり、GPU decode が入れば不要になります。したがってこの測定は GPU 経路の下限であり、上限ではありません。

### 5. CPU 側は場面の内容で 4 倍変わる

DGX Spark の 1280x720 で、ぼけ 1.130 ms に対し noise 4.721 ms です。GPU 段は同じ場面で 0.674 ms と 0.680 ms であり、ほぼ変わりません。実時間処理では最悪値が要件を決めるため、この性質の違いは平均値の比より重要です。

## 判断

**Phase 3 (GPU decode) へ進む根拠はあります。ただし Jetson Orin での成立は現時点で示せていません。**

- GB10 では、前処理と二値化を GPU へ移しただけで既に 1280x720 以上で CPU を上回ります。decode を GPU へ移せば CPU 段 (0.36 ms から 3.70 ms) が減り、同時に host への転送も不要になります。改善の余地は両方にあります。
- Orin では GPU 段が支配的です。decode を GPU へ移すと GPU 段はさらに増える可能性があり、単純に有利にはなりません。kernel 起動の削減と kernel 自体の最適化 (Phase 4) が Orin での成立条件です。
- 「統合 GPU の結果であり discrete GPU へは一般化できない」という制約は両機に当てはまります。discrete GPU では転送費用が大きく、`M-Pageable` と `M-Device` の差が本測定 (5% から 20%) より大きくなります。

## 未確定事項

- Orin の crossover が 4K の先にあるか、そもそも存在しないか。4K を超える解像度は corpus に無い。
- kernel 起動の固定費をどこまで下げられるか。window 3 通りの融合と CUDA Graph が候補。
- `M-Pinned` と `M-Managed` の値。統合 GPU では `M-Device` との差が小さいと見込まれるが未測定。
- CUDA event による kernel 時間と wall-clock の分離 (WP-4.1)。現在の段階時間は wall-clock であり、host 同期を含む。
- Jetson の CUDA Toolkit は 11.4 (JetPack 5.x)、DGX Spark は 13.0。toolkit version の影響は切り分けていない。
- 実写 corpus での測定。現在は合成 corpus のみ。

## 測定の再現

```
# container 内で実行する
./build/<preset>/tools/corpusgen/aruco3cuda_corpusgen --preset full --seed 20260827 \
  --output-dir /tmp/benchcorpus --manifest /tmp/benchcorpus/manifest.json

B=./build/<preset>/bench/aruco3cuda_bench
COMMON="--warmup 30 --latency-iterations 200 --throughput-frames 100 --cpu-list 0 --threads 1"
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
  taskset -c 0 $B $IMGS $COMMON --route CPU    --memory-mode N/A        >> results.jsonl
  taskset -c 0 $B $IMGS $COMMON --route Hybrid --memory-mode M-Device   >> results.jsonl
  taskset -c 0 $B $IMGS $COMMON --route Hybrid --memory-mode M-Pageable >> results.jsonl
done
python3 bench/aggregate.py results.jsonl
```

## 関連

- [評価計画](evaluation-plan.md)
- [実装計画](implementation-plan.md)
- [ADR-0003: 四角形候補抽出は案 A を主案とする](adr/0003-candidate-extraction-approach.md)
- [測定 harness](../bench/benchmark_harness.md)
