# Code Provenance 記録

## 目的

[知的財産・ライセンス方針](ip-and-licensing.md) の `Code provenance 記録` に従い、実装および設計の根拠として参照した外部資料を追跡可能な形で記録します。

## 対象範囲

論文、公開仕様、permissive license の source code、定義済み Dictionary data、CPU 基準実装として実行した software を対象とします。

## 現状

CUDA 実装は未着手です。現時点の記録は、設計のための API 調査のみです。

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
