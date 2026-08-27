# プロジェクト概要

## 目的

OpenCV の ArUco3 検出戦略と比較可能な CUDA 実装を独立リポジトリで構築し、DGX Spark と Jetson Orin における正確性と性能を検証します。結果に有効性と再現性がある場合は、OpenCV へのコントリビュートを検討します。

## 対象範囲

初期対象は、単一の 8-bit grayscale 画像から ArUco マーカーの ID と四隅座標を検出する処理です。

### 対象

- ArUco3 の画像ピラミッドを用いた高速検出戦略
- 定義済み ArUco Dictionary
- CUDA Stream による非同期実行
- DGX Spark GB10 と Jetson Orin
- OpenCV CPU 実装との正確性・性能比較
- GPU 常駐入力、転送込み入力、ハイブリッド処理の比較
- DGX Spark と Jetson Orin で共通に使用する開発 container 環境

### 初期対象外

- 姿勢推定
- ChArUco board と board refinement
- AprilTag 専用 quad detector
- ROS2 node
- Jetson Nano、Xavier、Thor
- Python API

## 現状

- リポジトリと初期文書を準備した段階です。
- CUDA 検出器、build system、テスト、データセットは未実装です。
- 開発 container 環境の設計と script は作成済みです。詳細は [Docker 環境設計](design/docker-environment.md) を参照してください。
- OpenCV の ArUco 検出には、公式の CUDA API がありません。
- CPU、CUDA、ハイブリッドのどれが最適かは入力条件に依存するため、測定前には決定しません。

## 目標

### 正確性

- OpenCV CPU 実装と同じ Dictionary および検出設定を使用できる。
- ID と marker rotation が一致する。
- 四隅座標の誤差を定量化できる。
- CPU 実装との差異を再現可能な画像と metadata で保存できる。

### 性能

- 640x480、1280x720、1920x1080、3840x2160 で crossover point を示す。
- カーネル時間と end-to-end 時間を分離する。
- 単一フレーム遅延と複数フレームのスループットを示す。
- DGX Spark と Jetson Orin で共通アルゴリズムが動作する。

### 保守性

- CUDA 固有処理と OpenCV adapter を分離する。
- 設定、作業領域、stream の所有権を明示する。
- 機種固有最適化を共通経路から分離する。
- 自動テスト、静的解析、Compute Sanitizer を継続実行できる構成にする。

## 成功判定

次を満たした場合に、OpenCV への提案準備へ進みます。

1. CPU 基準実装との検出差異が説明可能である。
2. 少なくとも 1 つの実用条件で、転送または同期を含む CUDA 経路が CPU より有意に高速である。
3. DGX Spark と Jetson Orin の両方で再現可能な build と測定手順がある。
4. API、テスト、ライセンスが OpenCV への移植を妨げない。

## 未確定事項

- ArUco3 に関する patent clearance の範囲と実施時期。
- 最初に対応する Dictionary の範囲。
- corner refinement を CUDA 化するか、CPU へ委譲するか。
- public API で OpenCV の型に依存するか、独自の軽量型を正本とするか。
- CUDA Toolkit と CMake の最低 version。
- 精度比較に使用する公開データセットと独自撮影データの構成。

## 関連

- [実装計画](implementation-plan.md)
- [検出パイプライン設計](design/detector-pipeline.md)
- [Docker 環境設計](design/docker-environment.md)
