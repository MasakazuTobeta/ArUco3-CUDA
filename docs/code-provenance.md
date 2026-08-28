# Code Provenance 記録

## 目的

[知的財産・ライセンス方針](ip-and-licensing.md) の `Code provenance 記録` に従い、実装および設計の根拠として参照した外部資料を追跡可能な形で記録します。

## 対象範囲

論文、公開仕様、permissive license の source code、定義済み Dictionary data、CPU 基準実装として実行した software を対象とします。

## 現状

Phase 3 の途中まで実装済みです。GPU 側は前処理、二値化、候補抽出、射影変換とセル sampling、Otsu と border 検証、Dictionary 照合までが記録の対象に含まれます。残りの段 (重複整理、四隅の subpixel 補正) は案 C のハイブリッド経路として CPU で実装しており、これも対象です。

## 記録

### PR-000: ArUco3 互換仕様の調査

| 項目 | 内容 |
| --- | --- |
| Implementation | [検出パイプライン設計](design/detector-pipeline.md)、[公開 API 草案](design/public-api.md) |
| Basis | OpenCV 公開 header および source の観測仕様、ArUco3 論文 |
| Source version | `opencv/opencv` branch `4.x`、branch head `6dc8e409035489769b4fe7edf3cd63f55bd23ec0`（2026-08-27 取得） |
| License | Apache-2.0 |
| Reused expression | なし。parameter 名、既定値、段階の順序、縮小率と pyramid 段数の算出式を仕様として記述した。source code の control flow、関数分割、コメントは複製していない |
| Patent review | 未実施 |

参照した file と取得時 hash は次のとおりです。

| File | SHA-256 |
| --- | --- |
| `modules/objdetect/include/opencv2/objdetect/aruco_detector.hpp` | `9e2d5bae344e1bb8dc7636430cc63310f3a9fc4e4c6013bfdd95cbade295b54d` |
| `modules/objdetect/src/aruco/aruco_detector.cpp` | `329ac3f0fd90939a23e1cbf21096352e2229a01998bd08d70ca50e17b99f11ed` |

公式 ArUco の GPLv3 source code は参照していません。

### PR-001: 開発 container への OpenCV 組み込み

| 項目 | 内容 |
| --- | --- |
| Implementation | `docker/scripts/build-opencv.sh`、[Docker 環境設計](design/docker-environment.md) |
| Basis | CPU 基準実装として実行するための build。source code の改変なし |
| Source version | `opencv/opencv` tag `4.14.0`、commit `0654a42e19215ef25b1d367d822f3c630447e7c7` |
| License | Apache-2.0 |
| Reused expression | なし。改変せず build して image へ install するのみ |
| Patent review | 未実施 |

build option は `BUILD_LIST=core,imgproc,imgcodecs,calib3d,objdetect`、`WITH_CUDA=OFF` です。取得元 commit と build option は image 内の `/opt/opencv/share/aruco3cuda/opencv-provenance.json` へ記録し、`record-environment.sh` が出力する環境情報 JSON へ埋め込みます。

script は clone した commit が指定値と一致することを確認し、不一致の場合は build を中止します。

### PR-002: 開発 container への CUDA Toolkit 組み込み

| 項目 | 内容 |
| --- | --- |
| Implementation | `docker/scripts/install-cuda-toolkit.sh` |
| Basis | NVIDIA が配布する CUDA Toolkit の apt package。改変なし |
| Source version | `cuda-nvcc-13-0` 13.0.88-1 ほか。実際の version は image 内 `/opt/aruco3cuda/cuda-provenance.json` に記録 |
| License | NVIDIA CUDA Toolkit EULA |
| Reused expression | なし。再配布せず、image build 時に NVIDIA の repository から取得する |
| Patent review | 未実施 |

image を第三者へ配布する場合は、CUDA Toolkit の再配布条件を EULA で確認する必要があります。現時点では各機で build する運用を前提とします。

### PR-003: Dictionary packed table の生成

| 項目 | 内容 |
| --- | --- |
| Implementation | `tools/dictgen`、`src/dictionary/generated/dict_aruco_mip_36h12.cpp` |
| Basis | OpenCV の `getPredefinedDictionary()` と `Dictionary::getBitsFromByteList()` の出力 |
| Source version | OpenCV 4.14.0、commit `0654a42e19215ef25b1d367d822f3c630447e7c7` |
| License | Apache-2.0 |
| Reused expression | codeword data を packed 表現へ変換して格納した。source code の複製はない |
| Patent review | 未実施 |

生成物には取得元 OpenCV version と再生成手順を header comment として記録しています。公式 ArUco の GPLv3 配布物からは抽出していません。生成物が OpenCV と整合することは `aruco3cuda_dictgen --check` と `test/reference/test_dictionary_conformance.cpp` で継続的に検証します。

### PR-004: 案 C ハイブリッド経路の CPU 側段階

| 項目 | 内容 |
| --- | --- |
| Implementation | `hybrid/hybrid_detector.cpp` |
| Basis | OpenCV `objdetect` の ArUco 検出器。互換性の検証に必要な範囲で振る舞いを再実装した |
| Source version | OpenCV 4.14.0、commit `0654a42e19215ef25b1d367d822f3c630447e7c7` |
| License | Apache-2.0 |
| Reused expression | 振る舞いの再実装。判定式、閾値の適用順序、候補の並び替え規則は OpenCV と同一にした。識別子、コメント、関数分割は本 project の規約に従って独自に書いた |
| Patent review | 未実施 |

対応関係は次のとおりです。左が本 project、右が OpenCV の関数です。

| 本 project | OpenCV |
| --- | --- |
| `find_quad_candidates` | `_findMarkerContours` |
| `reorder_corners` | `_reorderCandidatesCorners` |
| `quad_perimeter`、`average_quad_distance` | `MarkerCandidateTree` の周長計算、`getAverageDistance` |
| `average_module_size` | `getAverageModuleSize` |
| `quad_inside_quad` | `checkMarker1InMarker2` |
| `filter_too_close_candidates` | `ArucoDetectorImpl::filterTooCloseCandidates` |
| `find_optimal_level` | `_findOptPyrImageForCanonicalImg` |
| `extract_cell_pixel_ratio` | `_extractCellPixelRatio` |
| `count_border_errors` | `_getBorderErrors` |
| `run_cpu_stages` の depth 走査 | `ArucoDetectorImpl::identifyCandidates` |
| `run_cpu_stages` の四隅復元 | `findCornerInPyrImage`、`performCornerSubpixRefinement` |

出力の同一性を要求する以上、判定式と適用順序は一致させる以外にありません。一方で、同一にしたのは観測可能な振る舞いであり、source code の表現は複製していません。参照した file と取得時 hash は PR-000 と同じです。

OpenCV は `detectInvertedMarker` と複数 Dictionary の同時検出にも対応しますが、本 project は現時点でどちらも実装していません。`filter_too_close_candidates` は `detectInvertedMarker` が偽の場合の分岐のみを持ちます。

公式 ArUco の GPLv3 source code は参照していません。

### PR-005: GPU decode 段 (S7 から S8)

| 項目 | 内容 |
| --- | --- |
| Implementation | `src/core/cell_sample.{hpp,cu}`、`src/core/cell_decode.{hpp,cu}`、`src/core/dictionary_match.{hpp,cu}`、`src/dictionary/dictionary.cpp` の `build_cell_masks` と `identify_marker` |
| Basis | OpenCV `objdetect` の ArUco 検出器と `Dictionary::identify`、および `imgproc` の `warpPerspective`、`getPerspectiveTransform`、`threshold` の Otsu 経路、`core` の `meanStdDev` |
| Source version | OpenCV 4.14.0、commit `0654a42e19215ef25b1d367d822f3c630447e7c7`（2026-08-28 取得） |
| License | Apache-2.0 |
| Reused expression | 振る舞いの再実装。判定式、演算の順序、丸めの規則、打ち切りの位置は OpenCV と同一にした。識別子、コメント、関数分割、data 構造は本 project の規約に従って独自に書いた |
| Patent review | 未実施 |

対応関係は次のとおりです。

| 本 project | OpenCV |
| --- | --- |
| `cell_sample.cu` の `solve_lu8` | `getPerspectiveTransform` が使う `LUImpl` |
| `cell_sample.cu` の `warp_canonical_kernel` | `warpPerspective` の `INTER_NEAREST` 経路 |
| `cell_decode.cu` の `otsu_threshold` | `getThreshVal_Otsu_8u` |
| `cell_decode.cu` の内側の和と二乗和 | `cv::meanStdDev` |
| `cell_decode.cu` の外周走査 | `_getBorderErrors` |
| `build_cell_masks` | `CellBitMasks` の構築部 |
| `identify_marker`、`dictionary_match.cu` の `distance_to_id` | `Dictionary::identify`、`CellBitMasks::hammingDistanceToId` |

出力の同一性を要求する以上、判定式と演算の順序は一致させる以外にありません。一方で、同一にしたのは観測可能な振る舞いであり、source code の表現は複製していません。とくに OpenCV が `hal::and8u` と `hal::normHamming` を byte 列に対して呼ぶところを、本 project は 64 bit の packed 表現に対する `__popcll` 1 回で書いています。

参照した file と取得時 hash は次のとおりです。

| File | SHA-256 |
| --- | --- |
| `modules/objdetect/src/aruco/aruco_dictionary.cpp` | `9fd90d079e62239683300625ad001f437278b75fce6523368ffa3e5a64198ac8` |
| `modules/objdetect/include/opencv2/objdetect/aruco_dictionary.hpp` | `6b14b87c13dd3629d3fcfd82e63829e41f01f7cc60c166216e0ae99851f6f42e` |
| `modules/objdetect/include/opencv2/objdetect/aruco_detector.hpp` | `9e2d5bae344e1bb8dc7636430cc63310f3a9fc4e4c6013bfdd95cbade295b54d` |
| `modules/imgproc/src/imgwarp.cpp` | `d953cdd11db1bf7b562ba7cfb8b36f2aba3198f15714e5deb40840b01ae77912` |
| `modules/imgproc/src/thresh.cpp` | `ec40ffd08c842c946213cdd46c2b8036e43770445a4c4b797017c96fd3f91117` |
| `modules/core/src/mean.dispatch.cpp` | `8e52e41c7bf7f1942a5c366683d2742040f6d82db770f253b7d61cb18205133a` |
| `modules/core/src/matrix_decomp.cpp` | `887deb3905e1f4fbcab426be4346b4ada863051579f19e3011721704dd2ac596` |

公式 ArUco の GPLv3 source code は参照していません。

## 目標

- 実装 PR ごとに 1 行以上の記録を追加し、根拠を後から説明できる状態を保つ。
- 定義済み Dictionary を取り込む際は、取得元 file、commit、変換手順、変換前後の hash を同じ表へ記録する。
- 第三者 code を改変利用する場合は、必要な copyright と notice を `NOTICE` へ反映する。

## 未確定事項

- 記録の粒度を PR 単位とするか module 単位とするか。
- `NOTICE` の作成時期。現時点で第三者 code は含めていない。

## 関連

- [知的財産・ライセンス方針](ip-and-licensing.md)
- [Dictionary 方針](dictionaries.md)
