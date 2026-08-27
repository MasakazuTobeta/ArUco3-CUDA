# ADR-0002: build 基盤と対象環境の baseline を固定する

- Status: Accepted
- Date: 2026-08-27

## 目的

実装着手前に、CUDA architecture、toolchain の最低 version、CPU 基準に使用する OpenCV version を固定し、[実装計画](../implementation-plan.md) の作業単位が同じ前提で進むようにします。

## 対象範囲

CMake、CUDA Toolkit、host compiler、対象 CUDA architecture、CPU 基準実装の OpenCV version を対象とします。依存 library の追加方針は対象外です。

## 背景

### 開発機の実測値

2026-08-27 に開発機で測定した値です。

| 項目 | 値 |
| --- | --- |
| GPU | NVIDIA GB10 |
| Compute Capability | 12.1 |
| SM 数 | 48 |
| L2 cache | 24576 KB |
| `integrated` | 1 |
| `managedMemory` / `concurrentManagedAccess` | 1 / 1 |
| CUDA Toolkit | 13.0 (V13.0.88) |
| Driver | 580.95.05 |
| host compiler | gcc 13.3.0 |
| CMake | 3.28.3 |
| OS / arch | Ubuntu 24.04.3 LTS / aarch64 |
| CPU 論理 core 数 | 20 |
| system memory | 119 GB |

`nvcc --list-gpu-arch` は `compute_75` から `compute_121` までを列挙し、`compute_87` と `compute_121` の両方を含みます。

### 文書との差異

[README](../../README.md) と [アーキテクチャ](../architecture.md) は DGX Spark GB10 を Compute Capability 12.0、`sm_120` と記載しています。実機は 12.1 を報告するため、記載を修正する必要があります。

### 未 install の前提条件

開発機には OpenCV、`clang-format`、`ninja` が install されていません。`compute-sanitizer` は `/usr/local/cuda/bin` に存在します。

### OpenCV の release 状況

OpenCV は 4.x 系の最新が 4.14.0、別系統として 5.0.0 が release されています。[ADR-0001](0001-independent-implementation.md) の未確定事項である「4.x と 5.x のどちらを最初の対象にするか」は未決のままです。

## 決定

1. 対象 CUDA architecture を `sm_87` と `sm_121` とする。`CMAKE_CUDA_ARCHITECTURES` へ `87;121` を既定で指定し、必要に応じて上書きできるようにする。
2. `sm_120` を既定 target にしない。GB10 は 12.1 を報告するため、`sm_120` 向け binary は PTX からの JIT へ依存する。
3. C++ 標準を C++17、CUDA 言語標準も C++17 とする。
4. CMake の最低 version を 3.24 とする。`CUDA_ARCHITECTURES` の扱いが安定する最初の version であり、開発機と JetPack 6 系のどちらも満たす。
5. CPU 基準実装は OpenCV 4.14.0 に固定する。tag と build option を script へ記録し、測定結果へ埋め込む。
6. 最初の対象を OpenCV 4.x とし、5.x への追随は Phase 5 の判断へ委ねる。
7. 生成する成果物は host 側で例外を使用してよいが、core の公開 API は `Status` を返す。詳細は [公開 API 草案](../design/public-api.md) に従う。
8. toolchain の固定は host への直接 install ではなく開発 container で行う。CUDA Toolkit のみ host から bind mount する。詳細は [Docker 環境設計](../design/docker-environment.md) に従う。

## 理由

- 実機が報告する Compute Capability に合わせることで、JIT 待ちと機種差の切り分けが不要になる。
- `sm_87` と `sm_121` の 2 target に絞ることで、[アーキテクチャ](../architecture.md) の「共通アルゴリズム、機種固有最適化は分離」という方針を build 構成で表現できる。
- OpenCV 4.14.0 は `useAruco3Detection` を含む 4.x 系の最新であり、[Dictionary 方針](../dictionaries.md) が正本とする `predefined_dictionaries.hpp` を含む。
- CUDA Toolkit の最低 version を開発機の 13.0 に固定すると Jetson 側の JetPack に依存するため、ここでは決めない。

## 影響

### 利点

- 実装着手時に build 設定の議論が発生しない。
- 測定結果に environment の差が混入した場合の切り分けが容易になる。

### 欠点

- Jetson Orin 側の CUDA Toolkit version が確認できるまで、最低 version を確定できない。
- OpenCV 5.x を先に対象とする選択肢を一旦閉じる。

## 未確定事項

- CUDA Toolkit の最低 version。Jetson Orin 実機の JetPack version を確認してから決める。開発機に存在する別 project の container image は JetPack 5.1.2 と CUDA 11.4 を示しており、この値が Jetson 側の実際の環境である可能性がある。CUDA 11.4 が対象になる場合、DGX Spark の CUDA 13.0 との差が大きく、使用できる CUDA 機能と CUB / Thrust の version が制約される。実機確認は実装計画の WP-0.7 で行う。
- Jetson Orin 向けを cross-compile とするか実機 build とするか。
- `clang-format` の style 設定を OpenCV 準拠にするか独自にするか。

## 関連

- [ADR-0001: 独立リポジトリで CUDA 実装を先行する](0001-independent-implementation.md)
- [実装計画](../implementation-plan.md)
- [Docker 環境設計](../design/docker-environment.md)
- [アーキテクチャ](../architecture.md)
