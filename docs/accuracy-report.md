# 正確性評価の結果 (WP-4.4)

## 目的

合成 corpus の ground truth に対して CPU 基準・hybrid・CUDA の 3 経路を測り、
[評価計画](evaluation-plan.md) の正確性指標を条件別に示します。速度は
[benchmark 結果まとめ](benchmark-report.md) が扱います。

## 対象範囲

corpus preset `full` の 91 場面、真値 480 個のマーカーを対象とします。
DGX Spark GB10、Jetson AGX Orin、GeForce RTX 5070 Ti の 3 機で測りました。
Dictionary は `DICT_ARUCO_MIP_36h12` に固定しています。

実画像は対象外です。合成 corpus のみで測っています。

## 現状

### 全体

ArUco3 検出戦略を有効にした場合です。3 経路とも同じ 88 件を検出しました。

| 機体 | 経路 | precision | recall (下限以上) | rotation 一致 | 四隅 RMSE | 四隅 最大 |
| --- | --- | --- | --- | --- | --- | --- |
| DGX Spark GB10 | CPU | 100.00% | 94.44% | 85/85 | 0.5184 px | 3.6351 px |
| DGX Spark GB10 | Hybrid | 100.00% | 94.44% | 85/85 | 0.5184 px | 3.6351 px |
| DGX Spark GB10 | CUDA | 100.00% | 94.44% | 85/85 | **0.4806 px** | **1.0936 px** |
| Jetson AGX Orin | CPU | 100.00% | 94.44% | 85/85 | 0.5184 px | 3.6351 px |
| Jetson AGX Orin | Hybrid | 100.00% | 94.44% | 85/85 | 0.5184 px | 3.6351 px |
| Jetson AGX Orin | CUDA | 100.00% | 94.44% | 85/85 | **0.4806 px** | **1.0936 px** |
| RTX 5070 Ti | CPU | 100.00% | 94.44% | 85/85 | 0.5042 px | 3.6351 px |
| RTX 5070 Ti | Hybrid | 100.00% | 94.44% | 85/85 | 0.5042 px | 3.6351 px |
| RTX 5070 Ti | CUDA | 100.00% | 94.44% | 85/85 | **0.4653 px** | **1.0936 px** |

**false positive は 91 場面 x 3 経路 x 3 機のどこにも 1 件もありません。** ID を
誤った検出も 0 件です。rotation は検出した 85 件すべてで真値と一致しました。

RMSE の値が aarch64 の 2 機と x86_64 で違うのは、**corpus 画像そのものが
architecture 間で一致しないため**です。下の「corpus の再現性」を参照してください。

### recall を 3 区分で示す理由

ArUco3 は縮小後の 1 辺が `minSideLengthCanonicalImg` を下回るマーカーを原理上
検出しません。下限は `S + L * tau_i` であり、既定 (S=32、tau_i=0.05) では次の
ようになります。

| 解像度 | 検出できる 1 辺の下限 |
| --- | --- |
| 640x480 | 64 px |
| 1280x720 | 96 px |
| 1920x1080 | 128 px |
| 3840x2160 | 224 px |

corpus はこの下限を下回る大きさを意図的に含みます。全体の recall は 18.33% ですが、
**この数値は戦略上の下限に支配されており、実装の取りこぼしを表しません。**

| 区分 | 真値 | 検出 | recall |
| --- | --- | --- | --- |
| 全て | 480 | 88 | 18.33% |
| 下限以上 | 90 | 85 | **94.44%** |
| 下限未満 | 390 | 3 | 0.77% |

読むべきは 94.44% です。下限未満で 3 件検出できているのは、下限が境界であり
ちょうど下回る大きさでは検出できる場合があるためです。

### 条件別 (下限以上のマーカーのみ)

DGX Spark の値です。CPU と CUDA の recall はすべての条件で一致します。

| 条件 | 真値 | recall | CPU 四隅 RMSE | CUDA 四隅 RMSE |
| --- | --- | --- | --- | --- |
| clean | 58 | 100.00% | 0.4910 px | 0.4911 px |
| rotation (37 度) | 4 | 100.00% | 0.3778 px | 0.3778 px |
| perspective (0.6) | 4 | 100.00% | 0.4778 px | 0.4775 px |
| blur (sigma 2.0) | 4 | 100.00% | 0.7052 px | 0.7020 px |
| noise (sigma 12) | 4 | 100.00% | 0.4257 px | 0.4257 px |
| illumination (0.8) | 4 | 100.00% | 0.3778 px | 0.3778 px |
| occlusion (25%) | 4 | 75.00% | 1.1497 px | **0.4722 px** |
| border (はみ出し) | 4 | 75.00% | 0.3064 px | 0.3056 px |
| combined | 4 | **25.00%** | 0.4317 px | 0.4215 px |

**取りこぼしは 5 件で、複合劣化 3 件、遮蔽 1 件、境界はみ出し 1 件です。**
回転、射影歪み、ぼけ、noise、照度差は単独では 1 件も落としません。

### CPU 基準との差異

| 機体 | Hybrid vs CPU | CUDA vs CPU |
| --- | --- | --- |
| DGX Spark GB10 | 91/91 枚一致、最大差 0.000 px | 90/91 枚一致、最大差 3.804 px |
| Jetson AGX Orin | 91/91 枚一致、最大差 0.000 px | 90/91 枚一致、最大差 3.804 px |
| RTX 5070 Ti | 91/91 枚一致、最大差 0.000 px | 90/91 枚一致、最大差 3.804 px |

**hybrid 経路は 3 機すべてで CPU と完全に一致します。** hybrid は輪郭抽出から先を
CPU で行うため、これは想定どおりです。

CUDA 経路の唯一の差異は `occlusion_640x480` の ID 140 で、四隅が 3.804 px
違います。**この 1 件では CUDA の方が真値に近くなっています。** 真値に対する
誤差は CPU 3.6351 px、CUDA 1.0936 px です。遮蔽で輪郭が途切れた候補に対し、
2 つの実装が別の局所解へ収束したものです。

### 差異の出どころ

subpixel 補正を切って測ると、差異の性質が変わります。`--use-aruco3 0` は
ArUco3 を無効にし、OpenCV の既定に合わせて補正も切ります。

| 機体 | CUDA vs CPU | 内訳 |
| --- | --- | --- |
| DGX Spark GB10 | 82/91 枚一致、最大差 1.414 px | 四隅ずれ 18 件、過検出 3 件 |
| Jetson AGX Orin | 82/91 枚一致、最大差 1.414 px | 四隅ずれ 18 件、過検出 3 件 |
| RTX 5070 Ti | 82/91 枚一致、最大差 1.414 px | 四隅ずれ 18 件、過検出 3 件 |

**差はすべてちょうど 1.414 px、つまり sqrt(2) です。** これは整数座標の四隅が
斜めに 1 pixel ずれた場合の距離です。四隅の推定方法の違いがそのまま出ています。
CUDA は極点探索 ([WP-2.3](implementation-plan.md))、OpenCV は輪郭の多角形近似を
使います。18 件のうち 16 件は blur 場面です。

**subpixel 補正はこの 1 pixel の違いを吸収します。** 補正を入れると 18 件が
1 件まで減ります。残る 1 件は遮蔽で輪郭が途切れた候補です。

「過検出 3 件」は CPU が取りこぼしたマーカーを CUDA が検出したものです。3 件とも
複合劣化の場面にあり、precision は 100% のままです。補正なしの recall は
CPU 90.83% に対し CUDA 91.46% で、**CUDA の方が高くなります**。

補正なしでは、真値に対する四隅の誤差も CUDA の方が小さくなります。

| 機体 | CPU 四隅 RMSE | CUDA 四隅 RMSE |
| --- | --- | --- |
| DGX Spark GB10 | 0.8779 px | 0.8337 px |
| Jetson AGX Orin | 0.8779 px | 0.8337 px |
| RTX 5070 Ti | 0.8653 px | 0.8204 px |

ぼけ場面に限ると差が大きく、CPU 1.6281 px に対し CUDA 0.8068 px です。ぼけた
辺では多角形近似より極点探索の方が真値に近い四隅を返しています。ただし
**この比較の母数は 16 マーカーしかありません。** 傾向として扱い、結論には
しません。

### device memory

| 設定 | workspace 最大使用量 | 確保した容量 | 検出 91 回で増えた確保回数 |
| --- | --- | --- | --- |
| ArUco3 有効 | 17.51 MB | 22.69 MB | **0 回** |
| ArUco3 無効 | 414.51 MB | 414.51 MB | **0 回** |

3 機とも同じ値です。**検出中の `cudaMalloc` は 1 回もありません。** 確保は
初期化時に済みます。

**ArUco3 を無効にすると必要量が 23.7 倍になります。** 縮小しないため、
segmentation と二値化を原寸で行うためです。3840x2160 の場面が最大値を決めます。

### corpus の再現性

同じ seed から生成した corpus 画像が、**aarch64 と x86_64 で一致しません。**
91 場面のうち 54 場面が異なります。

| 場面 | 異なる画素 | 最大差 | 平均差 |
| --- | --- | --- | --- |
| clean_1280x720_n4_s128 | 736 / 921600 (0.0799%) | 4 階調 | 0.00194 |
| blur_1280x720 | 672 / 921600 (0.0729%) | 1 階調 | 0.00073 |
| rotation_640x480 | 362 / 307200 (0.1178%) | 4 階調 | 0.00265 |

corpus 生成器は OpenCV の `warpPerspective` と `GaussianBlur` を使います。
SIMD 経路が architecture ごとに違い、丸めが一致しないためと見ています (未検証)。

**この差は architecture をまたぐ数値の比較にだけ影響します。** 同じ機の中で
CPU と CUDA を比べる測定には影響しません。両者は同じ画像を見ています。

影響の大きさは四隅 RMSE で 0.5184 px と 0.5042 px の差、すなわち 2.7% です。
速度への影響は無視できます。異なる画素は 0.1% 未満です。

### 実行環境

| 項目 | DGX Spark GB10 | Jetson AGX Orin | RTX 5070 Ti |
| --- | --- | --- | --- |
| OS | Ubuntu 24.04.4 LTS | Ubuntu 20.04.6 LTS | Ubuntu 24.04.4 LTS |
| architecture | aarch64 | aarch64 | x86_64 |
| CPU | Cortex-X925 x10 + A725 x10 | Cortex-A78AE x12 | Core Ultra 7 265 |
| GPU | NVIDIA GB10 (統合) | Orin (統合) | RTX 5070 Ti (単体) |
| compute capability | 12.1 | 8.7 | 12.0 |
| CUDA Toolkit | 13.0 | 11.4 | 13.0 |
| driver | 580.95.05 | (取得できない) | 610.43.02 |
| power mode | (指定なし) | MAXN (0) | (指定なし) |
| GPU 最大 clock | 3003 MHz | 1300 MHz | 3090 MHz |
| OpenCV | 4.14.0 (`0654a42e`) | 4.14.0 (`0654a42e`) | 4.14.0 (`0654a42e`) |

power mode と clock は benchmark の測定結果 (`docs/measurements/2026-08-29-<機体>-sweep.jsonl`)
の `environment` 行に記録しています。正確性評価は clock に依存しないため、
評価器の出力へは含めていません。

## 測定の再現

```
# container 内で実行する。<preset> は機体ごとの build preset。
B=./build/<preset>/tools/evaluate/aruco3cuda_evaluate

# ArUco3 有効
$B --preset full --corpus-dir /tmp/c --output accuracy.json

# ArUco3 無効 (差異の出どころを切り分ける)
$B --preset full --corpus-dir /tmp/c --use-aruco3 0 --output accuracy-noaruco3.json
```

corpus はこの tool が seed から生成するため、事前の生成は要りません。生成に
使う seed と preset は `aruco3cuda_corpusgen` と同じであり、同じ画像になります。

結果は `docs/measurements/2026-08-29-<機体>-accuracy{,-noaruco3}.{json,txt}` に
あります。

## 目標

- 実画像データセットの注釈結果を真値として同じ指標を出す
- 差異のあった場面の可視化画像を保存する
- corpus 生成を architecture 非依存にするか、architecture ごとに固定した
  corpus を配布するかを決める

## 未確定事項

- corpus 画像が architecture 間で一致しない原因。OpenCV の SIMD 経路を疑って
  いるが、どの関数かは切り分けていない。
- 複合劣化での取りこぼし 3 件が、どの段階で落ちているか。二値化、輪郭、
  Dictionary 照合のいずれかは特定していない。
- ぼけ場面で極点探索の方が真値に近い理由。母数が 16 マーカーしかなく、
  傾向以上のことは言えない。
- 1280x720 の 96 px マーカーが「下限以上」に入らない。下限がちょうど 96 px で
  あり、真値の実効辺長が丸めでわずかに下回るためである。境界の扱いを
  決めていない。

## 関連

- [評価計画](evaluation-plan.md)
- [benchmark 結果まとめ](benchmark-report.md)
- [実装計画](implementation-plan.md)
- [正確性評価 CLI](../tools/evaluate/main.md)
- [正確性評価の指標](../tools/evaluate/accuracy.md)
