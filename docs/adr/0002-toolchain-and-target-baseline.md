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

### Jetson Orin 実機の実測値

2026-08-27 に実機で測定した値です。

| 項目 | 値 |
| --- | --- |
| 基板 | Jetson AGX Orin Developer Kit |
| L4T | R35.4.1 (JetPack 5.1.2) |
| OS / glibc | Ubuntu 20.04.6 LTS / 2.31 |
| CUDA Toolkit | 11.4 (V11.4.315) |
| nvcc の最大 architecture | `compute_87` |
| host compiler | gcc 9.4.0 |
| power mode | MAXN (0) |
| GPU 最大 clock | 1300 MHz |
| CPU 論理 core 数 | 12 |
| system memory | 61 GB |
| `nvidia-smi` | 存在しない |

CUDA 11.4 の nvcc は `compute_87` までしか対応しません。`CMAKE_CUDA_ARCHITECTURES` の既定値 `87;121` は Jetson では使用できないため、Jetson では `jetson-orin` preset を使用します。

### 未 install の前提条件

開発機には OpenCV、`clang-format`、`ninja` が install されていません。`compute-sanitizer` は `/usr/local/cuda/bin` に存在します。

### OpenCV の release 状況

OpenCV は 4.x 系の最新が 4.14.0、別系統として 5.0.0 が release されています。[ADR-0001](0001-independent-implementation.md) の未確定事項である「4.x と 5.x のどちらを最初の対象にするか」は未決のままです。

## 決定

1. 対象 CUDA architecture を `sm_87`、`sm_120`、`sm_121` とする。`CMAKE_CUDA_ARCHITECTURES` へ `87;120;121` を既定で指定し、必要に応じて上書きできるようにする。2026-08-28 に `sm_120` を追加した。経緯は [2026-08-28 の更新](#2026-08-28-の更新-対象-architecture-へ-sm_120-を追加) を参照。
2. 対象機が報告する Compute Capability のみを target とする。JIT へ依存させない。
3. C++ 標準を C++17、CUDA 言語標準も C++17 とする。
4. CMake の最低 version を 3.24 とする。`CUDA_ARCHITECTURES` の扱いが安定する最初の version であり、開発機と JetPack 6 系のどちらも満たす。
5. CPU 基準実装は OpenCV 4.14.0 に固定する。tag と build option を script へ記録し、測定結果へ埋め込む。
6. 最初の対象を OpenCV 4.x とし、5.x への追随は後の判断へ委ねる。
7. 生成する成果物は host 側で例外を使用してよいが、core の公開 API は `Status` を返す。詳細は [公開 API 草案](../design/public-api.md) に従う。
8. toolchain の固定は host への直接 install ではなく開発 container で行う。CUDA Toolkit も image へ固定し、必要な package のみを install する。host から bind mount する mode は手元の試行用に残すが、測定を伴う実行には使用しない。詳細は [Docker 環境設計](../design/docker-environment.md) に従う。
9. CUDA Toolkit の最低 version を 11.4 とする。対象とする Jetson AGX Orin が JetPack 5.1.2 (L4T R35.4.1) であり、同梱の CUDA が 11.4 であることを実機で確認した。共通経路はこの version で compile できる範囲に留める。
10. Jetson の pinned mode の base image を `nvcr.io/nvidia/l4t-cuda:11.4.19-devel` とする。CUDA のみを含み `l4t-jetpack` より小さい。

## 理由

- 実機が報告する Compute Capability に合わせることで、JIT 待ちと機種差の切り分けが不要になる。
- 対象機が実際に報告する Compute Capability だけを target とすることで、[アーキテクチャ](../architecture.md) の「共通アルゴリズム、機種固有最適化は分離」という方針を build 構成で表現できる。
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

- Jetson Orin を JetPack 6.x へ更新する時期。更新すれば CUDA 12.6 となり、DGX Spark の CUDA 13.0 との差が縮まる。
- Jetson Orin 向けを cross-compile とするか実機 build とするか。
- `clang-format` の style 設定を OpenCV 準拠にするか独自にするか。暫定として、OpenCV へのコントリビュートを想定し Google style を基礎に indent 4、桁数 100 とした `.clang-format` を置いている。決定後に本 ADR を更新する。

## 2026-08-28 の更新: 対象 architecture へ sm_120 を追加

### 背景

評価対象へ 3 機目を加えました。x86_64 の workstation に載せた ZOTAC GAMING GeForce RTX 5070 Ti 16GB (GB203) です。

既存 2 機はいずれも統合 GPU であり、host と device が同一物理 memory を共有します。[評価計画](../evaluation-plan.md) と [benchmark 報告](../benchmark-report.md) は「統合 GPU の結果であり discrete GPU へ一般化できない」ことを制約として明記しており、この制約を実測で埋められる機体がありませんでした。転送費用が効くため、memory 種別 (`M-Pageable`、`M-Pinned`、`M-Managed`、`M-Device`) の差もこの機で初めて意味を持ちます。

| 項目 | 値 |
| --- | --- |
| GPU | ZOTAC GAMING GeForce RTX 5070 Ti 16GB GDDR7 (GB203) |
| device memory | 16303 MiB |
| Compute Capability | 12.0 (実機で確認済み) |
| CPU | Intel Core Ultra 7 265 (20 core) |
| system memory | 62 GB |
| OS / arch | Ubuntu 22.04.5 LTS / x86_64 |
| CUDA Toolkit (host) | 13.2 |
| NVIDIA driver | 610.43.02 (`nvidia-driver-610-open`) |

### 決定の変更

対象 CUDA architecture を `sm_87` と `sm_121` の 2 つから、`sm_87`、`sm_120`、`sm_121` の 3 つへ広げます。`portability` preset の既定を `87;120;121` とし、`rtx-blackwell` preset を追加します。

当初 `sm_120` を「既定 target にしない」と決めていました。理由は GB10 が 12.1 を報告するため `sm_120` 向け binary が JIT へ依存する、というものでした。この理由は GB10 に対しては今も正しく、GB203 という 12.0 を報告する実機が加わったことで前提が変わりました。決定の向きは変えていません。「対象機が報告する Compute Capability のみを target とする」という原則をそのまま適用した結果です。

### container の構成

`rtx-blackwell` profile の base image と CUDA package は `dgx-spark` と同一にします (ubuntu:24.04、CUDA 13.0)。container 側を揃えておけば、両者の測定差は hardware の差だけになります。host の CUDA が 13.2 である点は pinned mode では影響しません。

### Compute Capability の確認

本 ADR の初版で DGX Spark GB10 を 12.0 と記載し、実機が 12.1 を報告して訂正した経緯があります。同じ取り違えを避けるため、製品仕様からの推定ではなく実測値で確定させました。

```
$ nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv
name, compute_cap, memory.total [MiB], driver_version
NVIDIA GeForce RTX 5070 Ti, 12.0, 16303 MiB, 610.43.02
```

推定どおり 12.0 でした。`sm_120` を target とします。

### Secure Boot への対応

対象機は Secure Boot が有効です。DKMS で kernel module を build すると署名が無く、再起動後に読み込まれません。Canonical が署名した prebuilt module (`linux-modules-nvidia-610-open-generic-hwe-22.04`) を使うことで、MOK 登録を伴わずに導入できます。

```
$ modinfo nvidia | grep signer
signer:         Canonical Ltd. Kernel Module Signing
```

この package は稼働中の kernel ではなく HWE の最新 kernel 向けに module を入れるため、導入後に再起動して新しい kernel で起動する必要があります。

## 関連

- [ADR-0001: 独立リポジトリで CUDA 実装を先行する](0001-independent-implementation.md)
- [ADR-0003: 四角形候補抽出は案 A を主案とする](0003-candidate-extraction-approach.md)
- [実装計画](../implementation-plan.md)
- [評価計画](../evaluation-plan.md)
- [Benchmark 報告](../benchmark-report.md)
- [Docker 環境設計](../design/docker-environment.md)
- [アーキテクチャ](../architecture.md)
