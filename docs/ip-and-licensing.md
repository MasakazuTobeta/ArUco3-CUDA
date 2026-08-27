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
- 現在の OpenCV 4.x は従来方式だけではなく、`DetectorParameters::useAruco3Detection` によって 2018 年論文の高速検出戦略を有効化できます。
- OpenCV 4.x は `DICT_ARUCO_MIP_36h12` を 6x6、250 code、最小 Hamming 距離 12 の定義済み Dictionary として収録しています。
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

次の方針を本 project の必須条件とします。

1. 公式 ArUco GPLv3 source code を閲覧、コピー、翻案または移植して実装しない。
2. ArUco3 検出戦略は、2018 年論文のアルゴリズム、数式、公開評価条件を設計根拠とする。
3. 互換性の確認と不足仕様の特定には、Apache-2.0 の OpenCV 4.x API、source code、test、観測可能な入出力を使用する。
4. OpenCV code または定義済み Dictionary を利用する場合は、対象 version、commit、file、license、変更内容を code provenance に記録し、必要な copyright と notice を保持する。
5. Dictionary は公式 ArUco GPLv3 配布物から抽出しない。OpenCV 4.x の Apache-2.0 配布物を正本とするか、公開アルゴリズムから独自生成する。
6. 論文の本文、図、表、評価画像を repository へ転載しない。必要な内容は引用の範囲を超えない独自の文章、数式参照、実験条件として記述する。
7. CUDA kernel の分割、memory layout、work queue、同期、Dictionary 照合、corner refinement は本 project の独自設計として記録する。
8. CPU 基準との比較には OpenCV 4.x を使用し、公式 ArUco GPLv3 binary を build または link した成果物を配布しない。
9. `ArUco` の名称は互換対象と技術方式を示す目的でのみ使用し、公式実装または Universidad de Córdoba との提携・承認を示唆しない。
10. 商用製品へ組み込む前に、販売予定国を対象として有効な patent claim の freedom-to-operate review を行う。

## Dictionary の取扱い

`DICT_ARUCO_MIP_36h12` に関する「生成」は、次の処理を区別します。

| 処理 | OpenCV 4.x で可能か | 本 project の扱い |
| --- | --- | --- |
| 既定の `DICT_ARUCO_MIP_36h12` を取得する | 可能。`getPredefinedDictionary()` を使用する | CPU 基準および互換データの正本とする |
| 指定 ID の marker image を生成する | 可能。`Dictionary::generateImageMarker()` を使用する | test fixture の生成に使用できる |
| 新しい custom Dictionary を生成する | 可能。`extendDictionary()` を使用できる | 既定 MIP Dictionary とは別物として扱う |
| MILP を解いて既定 MIP Dictionary と同一集合を再生成する | 論文から原理的には実装可能だが、OpenCV API は同一集合の再現を保証しない | 初期 scope 外。solver、制約、seed、tie-break を固定しても byte 単位の一致を別途検証する |

定義済み Dictionary を CUDA の constant memory 等へ変換する場合は、OpenCV の `bytesList` と全 4 回転について byte 単位で比較する test を必須とします。詳細は [Dictionary 方針](dictionaries.md) を参照してください。

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
- OpenCV から取得した定義済み Dictionary の attribution を `NOTICE` と source header のどちらへ記載するか。
- patent clearance の対象国と実施担当。

## 参考資料

- [Universidad de Córdoba: ArUco](https://www.uco.es/investiga/grupos/ava/portfolio/aruco/)
- [WIPO: Copyright](https://www.wipo.int/en/web/copyright)
- [U.S. Copyright Office: Computer Programs](https://www.copyright.gov/register/tx-programs.html)
- [OpenCV Issue: ArUco is now GPLv3](https://github.com/opencv/opencv_contrib/issues/2242)
- [OpenCV Issue #27118](https://github.com/opencv/opencv/issues/27118)
- [OpenCV: DetectorParameters](https://docs.opencv.org/4.x/d1/dcd/structcv_1_1aruco_1_1DetectorParameters.html)
- [OpenCV: Dictionary](https://docs.opencv.org/4.x/d5/d0b/classcv_1_1aruco_1_1Dictionary.html)
- [OpenCV PR #3151: ArUco3 speedup](https://github.com/opencv/opencv_contrib/pull/3151)
- [OpenCV: predefined_dictionaries.hpp](https://github.com/opencv/opencv/blob/4.x/modules/objdetect/src/aruco/predefined_dictionaries.hpp)
