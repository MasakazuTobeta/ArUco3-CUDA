# 評価計画

## 目的

CUDA 実装の正確性、性能、memory 使用量、移植性を再現可能な条件で評価し、CPU、CUDA、ハイブリッドの適用境界を示します。

## 対象範囲

DGX Spark GB10、Jetson Orin、OpenCV ArUco3 CPU 実装、開発する CUDA 実装を対象とします。

## 現状

評価用 code、データセット、基準値は未作成です。

## 目標

### 比較経路

| ID | 経路 | 測定範囲 |
| --- | --- | --- |
| CPU | OpenCV ArUco3 | `cv::Mat` 入力から結果取得まで |
| CUDA-E2E | host 入力 CUDA | upload、検出、download、同期を含む |
| CUDA-Resident | device 入力 CUDA | GPU 常駐画像から device 結果まで |
| Hybrid | CUDA + CPU | 各段階と同期を個別計測 |

### 入力条件

- 解像度: 640x480、1280x720、1920x1080、3840x2160
- 画像形式: 8-bit grayscale
- マーカー数: 0、1、4、16、上限近傍
- マーカー辺長: 8、16、32、64、128 pixel 以上
- 条件: 回転、射影歪み、ぼけ、noise、照度差、部分遮蔽、画像境界
- Dictionary: 最初の対応範囲を決定後に固定する

### 正確性指標

- precision / recall
- false positive 数
- ID と rotation の一致率
- 四隅座標の RMSE と最大誤差
- マーカー辺長、角度、条件別の検出率
- CPU と CUDA の差異件数および差異画像

OpenCV CPU 結果は互換性の基準ですが、必ずしも ground truth ではありません。合成データでは生成時の ground truth、実画像では注釈結果を併用します。

### 性能指標

- `T_kernel`: CUDA event で測定する検出処理時間
- `T_end_to_end`: 入力準備から host 結果取得までの wall-clock
- latency: p50、p95、p99
- throughput: 連続処理時の frame/s
- peak device memory とフレームごとの allocation 数
- CPU 使用率、GPU 使用率、対象機で取得可能な消費電力

### 測定手順

1. hardware、OS、CUDA、driver、compiler、OpenCV、power mode、clock を記録する。
2. 同じ入力と detector parameters を固定する。
3. 初期化と memory allocation を測定区間から分離する。
4. warm-up 後に十分な回数を測定する。
5. 外れ値を削除せず、集計方法と全分布を保存する。
6. CPU が速い条件を含め、crossover point を求める。

### 合格条件

- 正確性テストが対象条件で安定して再現する。
- Compute Sanitizer で memory error と race が検出されない。
- DGX Spark と Jetson Orin の両方で同一 test corpus が通る。
- 性能結果に測定範囲と同期点が明記されている。
- CUDA の優位条件と非優位条件を説明できる。

## 成果物

- machine-readable な環境情報と測定結果
- 集計表とグラフ
- CPU / CUDA 差異画像
- 再現 command
- benchmark report

大容量の画像や動画は Git repository へ直接 commit せず、保存先と checksum を manifest へ記録します。

## 未確定事項

- warm-up 回数と測定反復数。
- 使用する Jetson Orin model と power mode。
- CPU thread 数を固定するか、各環境の既定値を使用するか。
- 実画像データセットの入手・配布条件。
- 許容する四隅座標誤差と性能改善率の数値基準。
