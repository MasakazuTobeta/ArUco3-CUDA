# corpus_generator

## 目的

四隅の ground truth を持つ合成マーカー画像を、seed を固定すれば再生成できる形で作ります。CPU 基準結果は互換性の基準であって ground truth ではないため、正確性評価には生成時に既知である真値が必要です。

## 対象範囲

合成画像の生成、劣化条件の適用、ground truth の算出、manifest の出力を対象とします。実画像の収集と注釈は対象外です。

## 現状

- preset は `smoke`、`basic`、`full` の 3 種です。
- 劣化条件は回転、射影歪み、ぼけ、noise、照度勾配、遮蔽、画像境界にかかる配置です。
- 出力は PNG と `schema_version` 1 の manifest JSON です。

## 実装上の判断

### ground truth を射影変換の入力側から決める

マーカー画像を canonical 解像度で描き、目的の四角形へ射影変換して配置します。ground truth は変換先の四隅そのものであり、生成後の画像から推定しません。回転や射影歪みを加えても真値が厳密に定まります。

四隅は pixel の境界規約で表します。pixel 中心を整数座標とすると、`size` pixel の画像の外周は `-0.5` から `size - 0.5` です。OpenCV の検出結果もこの規約に一致します。

### scene ごとに独立した乱数列を使う

乱数種は `seed` と scene の通し番号から導出します。単一の乱数列を全 scene で共有すると、scene を 1 つ追加しただけで以降すべての scene の内容が変わり、corpus の差分が追えなくなります。

noise は `cv::randn` ではなく自前の分布で生成します。OpenCV の大域 RNG 状態に依存すると、他の処理の呼び出し順序で結果が変わるためです。

### 劣化は配置の後に画像全体へ適用する

ぼけ、照度勾配、noise の順で適用します。順序を固定しないと同じ条件から同じ画像が得られません。

### side_ratio を記録する

ArUco3 は辺長が小さいマーカーを縮小段階で除外します。除外の境界は `tau_i` そのものではなく、次の式で決まります。

```
side_px >= S + L * tau_i
```

`S` は `minSideLengthCanonicalImg`、`L` は画像の長辺、`tau_i` は `minMarkerLengthRatioOriginalImg` です。縮小率 `fxfy = S / (S + L * tau_i)` を掛けた後の辺長が `S` 以上である必要があることから導かれます。辺長比で言えば `side_ratio >= S / L + tau_i` です。

`minimum_detectable_side_px()` がこの下限を返します。manifest には各マーカーの `side_ratio` を記録し、どの `tau_i` で検出対象になるかを後から判断できるようにします。

境界値ちょうどでは再標本化の影響で検出できないことがあります。1280x720、`S = 32`、`tau_i = 0.05` の場合の下限は 96 pixel ですが、実測では 96 pixel は検出されず 128 pixel は検出されました。`smoke` preset は余裕のある寸法を使います。

### 画像は repository へ commit しない

生成物は大容量になるため commit せず、manifest に保存先と checksum を記録します。

## 目標

- 実画像を manifest へ同じ schema で登録できるようにする。
- マーカー ID の重複を避け、ID による対応付けを一意にする。
- `full` preset の規模と生成時間を評価計画の測定条件へ合わせる。

## 関連

- [検出パイプライン設計](../../docs/design/detector-pipeline.md)
- [評価計画](../../docs/evaluation-plan.md)
- [実装計画](../../docs/implementation-plan.md)
- [reference_runner](../../reference/reference_runner.md)
