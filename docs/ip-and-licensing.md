# 知的財産・ライセンス方針

## 目的

ArUco3 の CUDA 実装において、公式 ArUco、OpenCV、論文、外部実装の code と知的財産を適切に取り扱い、将来の商用利用と OpenCV へのコントリビュートを妨げないための方針を定義します。

> [!CAUTION]
> この文書は開発上のリスク管理方針であり、法律上の助言ではありません。商用公開、製品組込み、OpenCV への大規模な寄稿前には、対象国を定めて知的財産の専門家へ確認してください。

## 対象範囲

- 公式 ArUco library の GPLv3 code
- ArUco3 論文に記載されたアルゴリズム
- OpenCV の ArUco 実装
- ArUco Dictionary と評価データ
- 特許、商標、論文および図表

## 現状

- Universidad de Córdoba の公式 ArUco ページは、配布 software を GPLv3 と表示しています。
- 同ページには personal、research、educational purposes と commercial license への問い合わせも記載されています。GPLv3 自体は商用利用を一律に禁止する license ではないため、公式配布物へ追加条件があるかは、利用前に配布物内の license 本文または権利者へ確認が必要です。
- OpenCV は Apache-2.0 で配布され、OpenCV の ArUco3 対応は GPL 非互換 code に基づかないことを contribution checklist で確認して取り込まれています。
- 本リポジトリには、公式 ArUco または OpenCV の source code はまだ含まれていません。
- 本リポジトリの license は Apache License 2.0 とします。contribution も明示的な例外がない限り同 license で受け入れます。

## 著作権とアルゴリズムの区別

一般に著作権が保護するのは、source code、object code、文書、図表等の具体的な表現です。アイデア、処理手順、operation method、数学的概念、アルゴリズムそのものは、通常は著作権による独占対象と区別されます。

したがって、論文で公開されたアルゴリズムを理解し、GPL code の表現をコピーまたは翻案せず、独自の構造と code で実装した場合、その実装が GPL 派生物になるとは通常考えにくいです。ただし、次は避けます。

- GPL source code の逐語的または実質的なコピー。
- 変数名だけを変えた移植。
- 関数分割、制御構造、コメント、データ表現を高い類似性で維持した翻案。
- GPL code を build または link して配布する構成。
- GPL code から抽出した著作物性のある table、画像、test data の再配布。

## 特許

特許は著作権および open-source license とは別です。アルゴリズムを独立実装しても、対象国で有効な patent claim に該当すれば問題になり得ます。

初期調査では、2018 年の ArUco3 論文そのものを権利化した patent は確認できていません。ただし、この結果は不存在または非侵害を保証しません。商用公開前には、少なくとも次を実施します。

1. 発明者名、大学名、論文名、優先日前後の patent family を検索する。
2. 日本、米国、欧州および販売予定国の有効 claim を確認する。
3. marker Dictionary の生成方法と検出方法を分けて確認する。
4. 専門家による freedom-to-operate review の要否を判断する。

## 実装方針

### 使用してよい情報源

- ArUco3 論文に記載されたアルゴリズム、数式、評価条件。
- OpenCV の公開 API と観測可能な入出力。
- Apache-2.0 の OpenCV source code。ただし、利用または改変箇所と notice を記録する。
- permissive license の第三者実装。license と code provenance を記録する。
- 一般的な画像処理アルゴリズムと CUDA programming technique。

### 使用しない情報源

- 公式 ArUco GPLv3 code の実装詳細を移植目的で参照すること。
- license が不明な gist、forum attachment、生成 code。
- commercial license の内容が不明な proprietary code。

### 互換性評価

公式 ArUco または OpenCV を executable として実行し、同じ入力に対する出力を比較することは、source code のコピーとは分離して扱います。評価時は version、license、実行 command、設定を記録します。

## Code provenance 記録

実装単位または PR ごとに、次を記録します。

| 項目 | 内容 |
| --- | --- |
| Implementation | 対象 module / file |
| Basis | 論文、仕様、独自設計、参照実装 |
| Source version | DOI、URL、repository commit |
| License | Apache-2.0、MIT、BSD 等 |
| Reused expression | コピーまたは改変した code / table の有無 |
| Patent review | 未実施、簡易検索済み、専門家確認済み |

## Apache License 2.0 の運用

- repository root の `LICENSE` を正本とする。
- 新規 source file には `SPDX-License-Identifier: Apache-2.0` を記載する。
- Apache-2.0 code を取り込む場合は、元の copyright、patent、trademark、attribution notice を保持する。
- upstream に `NOTICE` がある第三者 code を配布する場合は、必要な attribution を本 repository の `NOTICE` へ追加する。
- contribution は Apache License 2.0 Section 5 に従う。

## 目標

- repository 全体を Apache License 2.0 で公開可能にする。
- 公式 ArUco GPLv3 code を含めず、GPL 派生物ではないことを code provenance で説明できるようにする。
- OpenCV へ寄稿する code が Apache-2.0 contribution checklist を満たすようにする。
- 商用利用前に必要な patent clearance を完了する。

## 未確定事項

- 公式 ArUco ページの commercial-use 表記と GPLv3 本文の関係。
- ArUco Dictionary の個別データに著作権その他の権利が成立する範囲。
- patent clearance の対象国と実施担当。

## 参考資料

- [Universidad de Córdoba: ArUco](https://www.uco.es/investiga/grupos/ava/portfolio/aruco/)
- [WIPO: Copyright](https://www.wipo.int/en/web/copyright)
- [U.S. Copyright Office: Computer Programs](https://www.copyright.gov/register/tx-programs.html)
- [OpenCV Issue: ArUco is now GPLv3](https://github.com/opencv/opencv_contrib/issues/2242)
- [OpenCV Issue #27118](https://github.com/opencv/opencv/issues/27118)
