# Dictionary 方針

## 目的

ArUco3-CUDA が対応する marker Dictionary の互換性、生成方法、データ由来、検証方法を定義します。

## 対象範囲

- OpenCV 4.x の定義済み ArUco Dictionary
- `DICT_ARUCO_MIP_36h12`
- marker image の生成
- custom Dictionary の生成
- CUDA 向け lookup table への変換

## 現状

- CUDA detector と Dictionary loader は未実装です。
- 現在の OpenCV 4.x は `DICT_ARUCO_MIP_36h12` を定義済み Dictionary として収録しています。
- `DICT_ARUCO_MIP_36h12` は 6x6 bits、250 codes、最小 Hamming 距離 12 です。OpenCV 4.14.0 で実測したところ、`markerSize = 6`、`bytesList` の行数 250、`maxCorrectionBits = 5` でした。
- OpenCV の `getPredefinedDictionary()` は収録済み codeword を取得します。
- `Dictionary::generateImageMarker()` は取得した Dictionary の指定 ID を画像化します。
- `extendDictionary()` は custom Dictionary を生成できますが、MILP で生成された既定の `DICT_ARUCO_MIP_36h12` と同じ codeword 集合を再現する API ではありません。

## 用語の区別

| 用語 | 意味 |
| --- | --- |
| Dictionary 取得 | OpenCV に収録済みの `bytesList`、`markerSize`、`maxCorrectionBits` を読み出すこと |
| marker image 生成 | Dictionary と ID から、黒枠を含む印刷・評価用画像を描画すること |
| custom Dictionary 生成 | 指定された marker size と code 数から、新しい codeword 集合を探索すること |
| MIP Dictionary 再生成 | 論文の MILP 問題を解き、既定 MIP Dictionary と同等または同一の codeword 集合を得ること |

## 採用方針

初期実装では、OpenCV 4.x の定義済み Dictionary を互換性の正本とします。最初の必須対象は `DICT_ARUCO_MIP_36h12` とし、他の `DICT_4X4_*`、`DICT_5X5_*`、`DICT_6X6_*`、`DICT_7X7_*` は同じ loader と lookup 形式で段階的に追加します。

定義済み Dictionary の取得元は、tag または commit を固定した OpenCV 4.x の `modules/objdetect/src/aruco/predefined_dictionaries.hpp` とします。公式 ArUco GPLv3 配布物から codeword、table、marker image を抽出しません。

CUDA 側では、正本の codeword から build 時に次を生成します。

- 各 ID の canonical bits
- 4 回転分の packed codeword
- Hamming distance 計算用の配置
- `markerSize` と `maxCorrectionBits` の metadata
- 取得元 OpenCV version、commit、入力 hash、生成物 hash

生成物を source tree へ格納する場合は、Apache-2.0 の attribution と生成手順を同じ変更に含めます。

## 検証

Dictionary ごとに、次の自動 test を必須とします。

1. ID 数、marker size、最大訂正 bit 数が OpenCV 基準と一致する。
2. 全 ID、全 4 回転の packed codeword が OpenCV の `bytesList` と一致する。
3. 全 ID の marker image を decode し、元の ID と回転を得られる。
4. 1 bit から訂正上限付近までの反転について、OpenCV と accept / reject が一致する。
5. Dictionary 内の最小 Hamming 距離を再計算し、公称値と一致する。
6. CPU と CUDA の lookup が、同じ入力に対して同じ ID、回転、距離を返す。

## 目標

- `DICT_ARUCO_MIP_36h12` を含む対象 Dictionary を OpenCV 4.x と byte 単位で互換にする。
- Dictionary data と生成 code の由来を Apache-2.0 の範囲で説明できるようにする。
- CUDA の memory 配置を変更しても、正本との一致を自動 test で保証する。
- 将来 custom Dictionary を追加しても、定義済み Dictionary と混同しない API にする。

## 未確定事項

- 最初の release で `DICT_ARUCO_MIP_36h12` 以外に必須とする Dictionary。
- codeword を repository に commit するか、build 時に OpenCV から生成するか。
- OpenCV への runtime dependency を許容するか、生成時 dependency のみにするか。
- CUDA constant memory と global memory の切替条件。
- MILP solver による独自 Dictionary 生成を将来 scope に含めるか。

## 関連

- [知的財産・ライセンス方針](ip-and-licensing.md)
- [実装計画](implementation-plan.md)
- [Docker 環境設計](design/docker-environment.md)
- [OpenCV Dictionary API](https://docs.opencv.org/4.x/d5/d0b/classcv_1_1aruco_1_1Dictionary.html)
- [OpenCV predefined dictionaries](https://github.com/opencv/opencv/blob/4.x/modules/objdetect/src/aruco/predefined_dictionaries.hpp)
- [Generation of fiducial marker dictionaries using Mixed Integer Linear Programming](https://doi.org/10.1016/j.patcog.2015.09.023)
