# reference_runner

## 目的

OpenCV の ArUco 検出を固定した設定で実行し、結果を機械可読形式で保存します。CUDA 実装との差分比較および crossover point の測定に使用する CPU 基準結果を作ります。

## 対象範囲

定義済み Dictionary を用いた単一画像の検出、検出設定の写像、実効縮小率の算出、結果の JSON 出力を対象とします。姿勢推定、board 検出、corner refinement 方式の切替は対象外です。

## 現状

- 定義済み Dictionary 18 種に対応します。
- 8-bit grayscale として読み込んだ静止画 1 枚以上を処理します。
- 出力は `schema_version` 1 の JSON です。

## 実装上の判断

### 検出結果を並べ替える

OpenCV の `detectMarkers` が返す順序は候補の抽出順に依存します。差分比較では扱いにくいため、ID、最初の corner の x、y の順で安定に並べ替えて出力します。

### rotation を出力しない

ArUco の marker rotation は、`detectMarkers` の戻り値に独立した項目として現れません。rotation は四隅の並び順として表現されます。値を推測して出力すると基準結果として誤りの元になるため、四隅の並びをそのまま記録し、比較は並び順で行います。

### 実効縮小率を記録する

ArUco3 検出戦略の `fxfy` を、[検出パイプライン設計](../docs/design/detector-pipeline.md) に記録した式で算出し、segmentation 画像の寸法とあわせて出力します。`minMarkerLengthRatioOriginalImg` の既定値は 0.0 であり、この場合 `useAruco3Detection` を有効にしても縮小は発生しません。測定条件としてこの値が残らないと、ArUco3 の効果を測ったのかどうかを後から判別できません。

### 実行時間を省略できる

`--omit-timing` を指定すると `detect_ms` を出力しません。実行時間は毎回変動するため、golden file との byte 単位比較にはこの指定を使用します。時間の測定そのものは benchmark harness の責務です。

### thread 数を既定で 1 に固定する

再現性を優先し、`cv::setNumThreads(1)` を既定とします。`--threads 0` を指定すると OpenCV の既定に従います。いずれの場合も実際の thread 数を出力へ記録します。

### 入力の checksum を記録する

結果 JSON と入力画像の対応を後から確認できるよう、入力の SHA-256 を記録します。実装は `aruco3cuda::util::sha256_file` です。

## 目標

- 合成 corpus と実画像に対して、環境情報とあわせた基準結果を一括生成できるようにする。
- CUDA 実装の出力と同じ schema を共有し、差分 tool が両方を同じ経路で読めるようにする。
- corner refinement 方式を選択できるようにし、比較条件を広げる。

## 関連

- [検出パイプライン設計](../docs/design/detector-pipeline.md)
- [評価計画](../docs/evaluation-plan.md)
- [実装計画](../docs/implementation-plan.md)
- [Code Provenance 記録](../docs/code-provenance.md)
