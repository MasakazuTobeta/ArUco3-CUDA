# host と device の間の memory 受け渡し

## 目的

対象 3 機で host と device がどう memory を共有するかを実測し、どの方式を選ぶべきかを記録します。「統合 GPU だから転送は不要」という直感が成り立つ条件と成り立たない条件を、device の性質と実測値で示します。

## 対象範囲

kernel が書いた結果を host が読むまでの経路を対象とします。host から device への入力転送も同じ性質に従いますが、測定は結果の取り出しで行っています。

## なぜ統合 GPU でも転送が発生するのか

統合 GPU では host と device が同一の物理 memory を使います。それでも `cudaMalloc` で確保した領域と host の通常 memory は**別の割り当て**であり、`cudaMemcpy` は実際に複製します。同じ DRAM の中で複製するだけですが、費用は発生します。

転送を省くには、host と device の双方から同じ領域を参照できる確保方法を使う必要があります。

| 方式 | 確保 | 性質 |
| --- | --- | --- |
| A. device memory + 明示的な複製 | `cudaMalloc` + `cudaMemcpy` | 常に複製が発生する |
| B. managed memory | `cudaMallocManaged` | 同一の pointer を双方から参照する。移送の有無は device の性質による |
| C. mapped host memory | `cudaHostAlloc(cudaHostAllocMapped)` | host memory を device から直接参照する |

## 対象機の性質

`cudaGetDeviceProperties` の実測値です。

| 機体 | `integrated` | `canMapHostMemory` | `managedMemory` | `concurrentManagedAccess` | `pageableMemoryAccess` |
| --- | --- | --- | --- | --- | --- |
| DGX Spark GB10 | 1 | 1 | 1 | **1** | **1** |
| Jetson AGX Orin | 1 | 1 | 1 | **0** | **0** |
| GeForce RTX 5070 Ti | 0 | 1 | 1 | 1 | 1 |

**`integrated` だけでは判断できません。** Jetson Orin は統合 GPU でありながら `concurrentManagedAccess` が 0 です。この場合 managed memory は device 側へ attach され、host が触ると移送が起きます。同じ物理 memory を共有していても、driver の管理単位が分かれているためです。

DGX Spark GB10 は `pageableMemoryAccess` が 1 であり、通常の host memory へ device から直接 access できます。

## 実測

1.46 MB を kernel が書き、host が**全 byte を読み終える**までの時間です。中央値、暖機 30 回のあと 170 回。

| 機体 | A. 複製 | B. managed | C. mapped |
| --- | --- | --- | --- |
| DGX Spark GB10 | 0.170 ms | **0.110 ms** | 0.128 ms |
| Jetson AGX Orin | 2.430 ms | 2.156 ms | **2.065 ms** |
| GeForce RTX 5070 Ti | 0.402 ms | **4.831 ms** | 0.396 ms |

この値は host 側が 1.46 MB を走査する時間を含みます。どの方式でも host は結果を読むため、その分は共通の下限です。方式の差は「読めるようにするまで」の部分に現れます。

複製の段階だけを分けて測ると、DGX Spark で次のようになります。

| 段階 | 時間 |
| --- | --- |
| kernel + 同期のみ (B と C はここで host が読める) | 0.048 ms |
| + `cudaMemcpy` (A) | 0.120 ms |

**複製そのものが 0.071 ms** であり、統合 GPU では B か C でこれを丸ごと省けます。

## 機体ごとの選択

| 機体 | 推奨 | 理由 |
| --- | --- | --- |
| DGX Spark GB10 | B (managed) | 移送が起きず複製も不要。`pageableMemoryAccess` が 1 のため通常 memory も使える |
| Jetson AGX Orin | C (mapped) | `concurrentManagedAccess` が 0 のため managed は移送を伴う。zero copy なら避けられる |
| GeForce RTX 5070 Ti | A または C | 単体 GPU のため無償の共有は無い。**B は 12 倍遅くなる** |

RTX 5070 Ti で B が極端に遅いのは、host が読むたびに page が device から host へ移送されるためです。単体 GPU で managed memory を「便利だから」と選ぶと、明示的な複製より大幅に遅くなります。

## 現在の実装との差

案 C のハイブリッド経路は方式 A を使い、毎 frame 1499 KB を 8 回の同期転送で戻しています (pyramid 5 level と二値化画像 3 枚)。DGX Spark での実測は 0.269 ms でした。

単発の複製として測った 0.071 ms との差は、**8 回それぞれの呼び出し費用** (1 回あたり約 25 µs) です。改善の余地は 2 つあります。

1. 統合 GPU で B か C を使い、複製自体を省く
2. 8 回の blocking 呼び出しを stream へ載せ、非同期にまとめる

## 測定上の落とし穴

**managed memory の測定では、確保した領域を全て読む必要があります。**

最初の測定で 1 byte だけ読んだところ、RTX 5070 Ti の managed が複製より 21 倍速いという結果になりました。managed memory は必要な page だけを遅延移送するため、1 byte の読み出しでは 1 page しか移送されていませんでした。全 byte を読む形へ直すと、逆に 12 倍遅いことが分かりました。

同じ理由で、`cudaMemPrefetchAsync` を使わない managed memory の測定は、host 側の access pattern に強く依存します。比較する場合は、実際の利用と同じだけ読むこと。

## 未確定事項

- 入力 (host から device) についても同じ比較を行うか。現在は結果の取り出しのみ測定している。
- `cudaMemPrefetchAsync` を併用した場合の managed memory の挙動。単体 GPU で改善するかは未測定。
- Jetson で mapped host memory を使った場合、kernel 側の access が遅くなる度合い。zero copy は kernel から見ると host memory への access になる。
- 統合 GPU で `pageableMemoryAccess` が 1 の場合、`cv::Mat` の buffer をそのまま device へ渡せるか。DGX Spark では原理的に可能だが未検証。

## 統合 GPU では page cache が「device の空き」を食う

DGX Spark GB10 で `ctest -j 8` が断続的に落ちるようになりました。15 回中 5 回、毎回違う test が `out of memory` で失敗します。原因は**実装ではなく環境**でした。

```
GPU:  空き 3513.6 MiB / 全体 122572.2 MiB
host: MemTotal 119 GB / MemFree 5 GB / MemAvailable 107 GB / Cached 100 GB
```

統合 GPU では device memory が host memory と同じものです。`cudaMemGetInfo` が返す「空き」は **`MemFree` に相当し、`MemAvailable` ではありません**。page cache は回収可能ですが、CUDA の確保はそれを待たずに失敗します。

長い作業 session では build 出力、corpus、docker layer で page cache が 100 GB まで育ちます。そうなると 1 process でも 3.5 GB しか確保できず、`ctest -j 8` のように process を並べると簡単に枯渇します。

回収すれば戻ります。

```
sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
```

実測では 3.5 GB から **108.9 GB** へ戻り、`ctest -j 8` を 10 回続けて全て通りました。

**測定の前には page cache を落とすこと。** 落とさずに測ると、確保の失敗だけでなく、host 側の memory 帯域の奪い合いで測定値そのものが揺れます。

## 関連

- [検出パイプライン設計](detector-pipeline.md)
- [評価計画](../evaluation-plan.md)
- [Benchmark 報告](../benchmark-report.md)
- [アーキテクチャ](../architecture.md)
