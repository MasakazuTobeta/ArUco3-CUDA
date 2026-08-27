# ADR-0001: 独立リポジトリで CUDA 実装を先行する

- Status: Accepted
- Date: 2026-08-27

## 目的

OpenCV 本体へ直接大規模な CUDA 実装を提出する前に、独立リポジトリで技術的有効性、API、正確性、性能を検証する方針を記録します。

## 対象範囲

ArUco3 検出戦略の CUDA 実装、DGX Spark と Jetson Orin での評価、OpenCV への将来のコントリビュート判断を対象とします。

## 背景

- OpenCV の `cv::aruco::ArucoDetector::detectMarkers` には CUDA 実装がない。
- OpenCV Issue #27118 には CUDA 実装の要望があるが、API、module、実装範囲は合意されていない。
- ArUco3 は輪郭、候補整理、可変長出力など GPU に不向きな処理を含む。
- ユニファイドメモリ環境でも同期、cache、kernel launch の費用があり、CUDA が常に高速とは限らない。

## 決定

`ArUco3-CUDA` を独立した検証・参照実装として開発します。

1. OpenCV の review 制約から独立して algorithm と API を反復する。
2. CPU、CUDA、ハイブリッドを同じ入力で評価する。
3. DGX Spark GB10 と Jetson Orin の両方を対象とする。
4. OpenCV 互換性をテストしつつ、core の OpenCV 型依存を最小化する。
5. 有効性を確認してから Issue #27118 へ結果を提示し、upstream API を合意する。
6. 公式 ArUco の GPLv3 code はコピーまたは翻案せず、論文、公開仕様、permissive license の参照実装から独立して実装する。
7. 本リポジトリと contribution の license は Apache License 2.0 とする。

## 理由

- 大規模 PR の前に性能上の成立性を確認できる。
- OpenCV review 中の interface の手戻りを減らせる。
- CPU が有利な条件を含め、適切な backend 選択を設計できる。
- Jetson 固有実装ではなく、Ampere と Blackwell で移植性を示せる。

## 影響

### 利点

- 実験、測定、API 変更を迅速に行える。
- 再現可能な benchmark と test corpus を upstream 提案に添付できる。
- OpenCV に採用されない場合も単独 library として利用できる。

### 欠点

- 後で OpenCV の module 構造と API へ移植する作業が発生する。
- 独立実装と upstream 実装が分岐する可能性がある。
- 初期段階では OpenCV package から直接利用できない。

## OpenCV への移行条件

- 正確性差異が説明可能である。
- 転送または同期を含む実用条件で性能上の利点がある。
- DGX Spark と Jetson Orin の build・test・benchmark が再現可能である。
- permissive license とコード来歴が明確である。
- 商用利用を想定する場合に必要な patent clearance が完了している。
- maintainer と module、branch、API を合意している。

## 未確定事項

- upstream 先が `opencv` 本体か `opencv_contrib` か。
- `cv::cuda::ArucoDetector` の具体的な API。
- OpenCV 4.x と 5.x のどちらを最初の対象にするか。

## 関連

- [プロジェクト概要](../project-overview.md)
- [アーキテクチャ](../architecture.md)
- [評価計画](../evaluation-plan.md)
- [知的財産・ライセンス方針](../ip-and-licensing.md)
- [OpenCV Issue #27118](https://github.com/opencv/opencv/issues/27118)
