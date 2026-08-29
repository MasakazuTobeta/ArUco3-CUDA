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

- Universidad de Córdoba の公式 ArUco ページの表記を 2026-08-29 に確認しました。原文は次のとおりです。

  > This software is licensed under GPLv3 license for personal, research and educational purposes. For a commercial license please contact rmsalinas@uco.es

- GPLv3 自体は商用利用を一律に禁止する license ではありません。上の表記が GPLv3 への追加条件を意図しているのか、単に商用向けの別 license を案内しているだけなのかは、この表記だけでは決まりません。**この点は未解決のままです。**
- ただし**本 project はこの解釈に依存しません。** 公式 ArUco 配布物を取得も参照も利用もしていないためです。実装根拠は 2018 年論文と OpenCV 4.x に限定しており、Dictionary も OpenCV 由来です。上の表記が仮に厳しい側の意味であっても、本 project の成果物には及びません。
- OpenCV は Apache-2.0 で配布され、OpenCV の ArUco3 対応は GPL 非互換 code に基づかないことを contribution checklist で確認して取り込まれています。
- ただし **OpenCV 4.x は file ごとに license header が違います。** project 全体の LICENSE は Apache-2.0 ですが、`imgproc` の古い file は 3 条項 BSD の header を残し、Intel Corporation ほかを著作権者として挙げています。本 project が振る舞いを写した `cornersubpix.cpp`、`samplers.cpp`、`thresh.cpp`、`imgwarp.cpp`、`geometry.cpp` はいずれも 3 条項 BSD です。file ごとの内訳は [Code Provenance 記録](code-provenance.md) の PR-005 にあります。
- 現在の OpenCV 4.x は従来方式だけではなく、`DetectorParameters::useAruco3Detection` によって 2018 年論文の高速検出戦略を有効化できます。
- OpenCV 4.x は `DICT_ARUCO_MIP_36h12` を 6x6、250 code、最小 Hamming 距離 12 の定義済み Dictionary として収録しています。
- 本リポジトリに、公式 ArUco または OpenCV の source code を貼り付けた file はありません。ただし `test/reference/test_corner_refine.cpp` の `oracle` namespace は、`cv::cornerSubPix` と `cv::getRectSubPix` の演算と評価順序を逐語で写しています。GPU 実装が写し間違えていないかを機の違いから切り離して測るためです。写し元は 3 条項 BSD であり、**保守的な扱いとして著作権表示と license 本文を repository root の `NOTICE` へ置きました。** 写しが法的に二次的著作物にあたるかの判断は専門家の確認を要します。
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

商用公開前には、少なくとも次を実施します。

1. 発明者名、大学名、論文名、優先日前後の patent family を検索する。
2. 日本、米国、欧州および販売予定国の有効 claim を確認する。
3. marker Dictionary の生成方法と検出方法を分けて確認する。
4. 専門家による freedom-to-operate review の要否を判断する。

2026-08-29 に手順 1 と 3 を実施し、手順 2 を米国の登録 claim について一部実施しました。記録は [特許 Clearance 下調べ記録](patent-clearance.md) にあります。要点は次のとおりです。

- **ArUco3 論文の著者 5 名と Universidad de Córdoba を発明者・出願人とする marker 検出・生成の特許は 1 件も見つかりませんでした。** 論文著者名は他社特許の引用文献としてのみ現れます。ただしこれは不存在の証明ではありません。表記ゆれ (`Muñoz` の発音区別符号) で検索結果が変わることも実測しています。
- **ArUco3 の中核 (最小マーカー寸法からの縮小率決定、縮小画像での候補探索、pyramid による四隅復元) を claim する特許は見つかりませんでした。**
- **Dictionary の生成方法 (MILP による符号集合設計) を claim する特許も見つかりませんでした。** 本 project はそもそも Dictionary を生成せず、OpenCV 収録済み Dictionary の変換物を保持するだけです。
- 技術分野が最も近いのは **NVIDIA の GPU fiducial marker 検出 family** (`US11113819B2` ほか) です。独立 claim は候補 corner 点の検出と閾値画素距離・辺の角度による絞り込みを要件とし、本 project にはそのいずれも存在しません。これは事実の差異であり、非侵害の評価ではありません。
- **日本にも登録特許が 2 件あります。** `JP7429542B2` (NVIDIA、登録 2024-02-08) と `JP7459051B2` (Magic Leap、登録 2024-04-01) で、いずれも存続中の表示です。請求項 1 の全文を確認しました。NVIDIA の日本請求項 1 は米国と同じ 3 要件 (色空間変換、候補コーナ点の検出、しきいピクセル距離) を持ち、**本 project にはいずれも存在しません**。Magic Leap の日本請求項 1 は**頭部装着装置**を前提部とし、本 project は library です。ただし**日米で限定が異なります** (米国の Harris 系定数 `k1` は日本の請求項に無い)。

### 対象国の決定

**2026-08-29 に販売予定国を日本と米国に決定しました。** これにより手順 2 の範囲が定まりました。

| 国 | 扱い | 理由 |
| --- | --- | --- |
| 日本 | **対象** | 販売予定国 |
| 米国 | **対象** | 販売予定国 |
| 欧州 | **対象外** | 販売予定に含めないため。調べた結果として無かったのではなく、範囲から外した |
| その他 | **対象外** | 同上 |

手順 2 の条文は「日本、米国、欧州および販売予定国」と書いていますが、欧州を販売予定に含めない決定に従い、欧州は範囲から外します。**販売予定国を欧州へ広げる場合は手順 2 をやり直す必要があります。** この対応関係を残すため、条文はそのままにして本節で範囲を記録します。

実施担当は未定のままです。

手順 2 の進捗と、日本・米国それぞれの到達点は [特許 Clearance 下調べ記録](patent-clearance.md) にあります。手順 4 の判断はまだ行っていません。

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

## OSS として公開してよいか

商用製品として実施できるか (freedom-to-operate) とは別の問いです。**source を Apache-2.0 で公開すること**に限って整理します。

### 著作権と license: 障害は見当たりません

| 確認事項 | 状態 | 根拠 |
| --- | --- | --- |
| 公式 ArUco GPLv3 code の混入 | **無し** | 参照も取得もしていない。実装根拠は 2018 年論文と OpenCV 4.x。[Code Provenance 記録](code-provenance.md) に file と hash 単位で記録 |
| 第三者 code の同梱 (vendoring) | **無し** | `third_party` / `vendor` 等の dir は存在しない。依存はすべて `find_package` で環境から解決する。**本 repository は第三者の code を 1 byte も配布しない** |
| 依存の license | **すべて permissive** | OpenCV (Apache-2.0 と 3 条項 BSD の混在)、GoogleTest (3 条項 BSD)、CUDA Toolkit (NVIDIA EULA。**再配布せず**、利用者の環境で build する) |
| copyleft 依存 | **無し** | 上記のとおり |
| Dictionary data の出所 | **OpenCV** | Apache-2.0 の `getPredefinedDictionary()` 出力を変換。GPLv3 配布物からは抽出していない。OpenCV との byte 一致を test で継続検証 |
| 写した code の attribution | **記載済み** | 3 条項 BSD の 5 file について著作権表示と license 本文を `NOTICE` へ保持 |
| SPDX 表記 | **全 source file にあり** | 2026-08-29 に `tools/probe-memory-transfer.cu` の欠落を補って完了 |

**この範囲では、Apache-2.0 での source 公開を妨げるものは見つかりませんでした。**

### 特許: OSS にしても消えません

**「OSS だから特許は関係ない」は成り立ちません。** 日本の特許法 2 条 3 項は program の譲渡・提供を実施に含み、米国でも頒布は §271 の対象です。無償であることは実施か否かを左右しません。

一方で、**source library の公開は製品の販売より対比が明確です。** [特許 Clearance 下調べ記録](patent-clearance.md) が挙げた差異のうち、製品形態に依存するもの (NVIDIA の色空間変換が製品側に存在しうる点、Magic Leap の頭部装着装置の前提部) は、**library の source を配布するだけなら成立しにくくなります。** 配布物が library そのものだからです。

ただし次は未検討のままです。**OSS 公開でも消えません。**

- 均等論とクレーム解釈。
- 従属請求項。
- 間接侵害。日本の特許法 101 条は「発明による課題の解決に不可欠なもの」の提供を侵害とみなす類型を持ち、**library の提供はこの類型で議論されうる位置にあります。**

### 商標: 調査しました。第 9 類に一般の computer program は指定されていません

2026-08-29 に TMview (EUIPO/TMDN の全世界横断 database) と Toreru 商標検索を headless browser で開き、`aruco` の全世界 209 件を取得しました。

**米国には `ARUCO` の商標がありません。** 米国の登録・出願は 6 件ありますが、いずれも `JARUCO`、`DARUCOT`、`HARUCO`、`DRARUCO`、`SUBARUCONNECT` という別の標章です。標章そのものが `ARUCO` と一致するものは 0 件でした。

**日本には `ARUCO` / `aruco` と同一の商標が 8 件あります。** 本 project に関係しうるのは第 9 類を持つ次の 2 件です。

| 登録番号 | 標章 | 権利者 | 区分 | 存続期間満了日 |
| --- | --- | --- | --- | --- |
| 第5509717号 | ＡＲＵＣＯ | 株式会社学研ホールディングス | 09、16 | 2032-07-27 |
| 第5711433号 | ａｒｕｃｏ | 株式会社学研ホールディングス | 09、16、39、41、43 | 2034-10-17 |

いずれも旅行 guide「地球の歩き方」系列の商標で、株式会社ダイヤモンド・ビッグ社から学研ホールディングスへ移転済みです (第5711433号は 2020-12-10 に移転申請、2021-01-22 に移転登録)。2024-09-27 に更新申請済みで、**現に維持されています。**

**第 9 類の指定商品は次のとおりで、一般の computer program を含みません。**

- 第5509717号 第 9 類: 映写フィルム、スライドフィルム、スライドフィルム用マウント
- 第5711433号 第 9 類: 家庭用テレビゲーム機用プログラム、携帯用液晶画面ゲーム機用のプログラムを記憶させた電子回路及び CD-ROM、レコード、インターネットを利用して受信し及び保存することができる音楽ファイル、同じく画像ファイル、録画済みビデオディスク及びビデオテープ、映写フィルム、スライドフィルム、スライドフィルム用マウント、ダウンロード可能な電子書籍、電子出版物

**「電子計算機用プログラム」は、どちらにも指定されていません。** 指定されている program は家庭用テレビゲーム機用と携帯ゲーム機用に限られます。

そのほかの日本の同一商標は、ロート製薬 (第6805531号、第 35 類、健康促進を目的とする景品交換用ポイントの発行・管理)、西垣靴下 (第6109871号、第 25 類、靴下)、学研ホールディングス (第5326799号、第 39・41・43 類) です。学研ホールディングスは 2026-04-17 に第 18・24・25 類で 3 件を新規出願し、審査中です。

**Universidad de Córdoba および ArUco の著者を権利者とする商標は、全世界 209 件のいずれにもありませんでした。**

### 学研ホールディングスの登録についての判断

**2026-08-29 に、学研ホールディングスの登録は本 project の妨げにならないと判断しました。** 第 9 類の指定商品が家庭用・携帯用ゲーム機向け program と出版物・映像音楽 file に限られ、一般の computer program を含まないためです。**この判断は本 project の判断であり、専門家の意見ではありません。**

### 残る論点は考案者側の未登録商標です

登録の面では妨げが無いため、残るのは**未登録の商標としての保護** (不正競争防止法 2 条 1 項 1 号・2 号) です。**問題になりうるのは学研ホールディングスではなく、`ArUco` の考案者側です。** 計算機視覚の分野で `ArUco` は marker 方式の名称として広く通用しており、登録が無くても周知表示として扱われる余地があります。

これに対する材料は次のとおりです。

- **Universidad de Córdoba と著者 5 名は、全世界 209 件のいずれにも商標を持ちません。** 登録による権利主張の基盤がありません。
- **OpenCV 自身が `aruco` を module 名として長く使っています。** Apache-2.0 の主要 library が同名の module を公開し続けている事実は、この分野で名称が技術方式の記述として通用していることを示します。
- 本 project は方針 9 のとおり、名称を互換対象と技術方式の明示にのみ用い、提携・承認を示唆しません。README と `NOTICE` にその旨を記載しています。

**それでも不正競争防止法上の評価は行っていません。** 上記は材料であって判断ではありません。

### 商標調査の限界

- 209 件は TMview 収録分です。TMview に参加していない官庁の登録は含みません。
- 本節の情報源は TMview と Toreru であり、**J-PlatPat での一次確認はしていません。** 調査時、J-PlatPat は保守停止と SPA の描画不能により操作できませんでした。
- 指定商品の類否を類似群コードで対比してはいません。

### 判断

**著作権・license の面では公開できる状態です。商標も、名称の使用を妨げる登録は見つからず、見つかった登録 (学研ホールディングス) は本 project の妨げにならないと判断しました。**

残るのは次の 2 つです。

1. **特許のうち OSS 公開でも消えない部分。** 均等論とクレーム解釈、従属請求項、間接侵害の 3 点で、いずれも [特許 Clearance 下調べ記録](patent-clearance.md) に残件として挙げてあります。
2. **考案者側の未登録商標。** 登録は存在しませんが、不正競争防止法上の評価は行っていません。

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
- 3 条項 BSD の header を持つ file から振る舞いを写した場合も、著作権表示・条件・免責を `NOTICE` へ保持する。Apache-2.0 だけを前提にすると、この保持義務を落とす。
- 著作権者の名称を、本 project の宣伝や推奨の示唆に使わない。3 条項 BSD の第 3 条による。
- contribution は Apache License 2.0 Section 5 に従う。

## 目標

- repository 全体を Apache License 2.0 で公開可能にする。
- 公式 ArUco GPLv3 code を含めず、GPL 派生物ではないことを code provenance で説明できるようにする。
- OpenCV へ寄稿する code が Apache-2.0 contribution checklist を満たすようにする。
- 商用利用前に必要な patent clearance を完了する。

## 未確定事項

- 公式 ArUco ページの commercial-use 表記と GPLv3 本文の関係。表記そのものは 2026-08-29 に確認済み。解釈は未解決だが、本 project は公式配布物を使っていないため依存しない。
- OpenCV から取得した定義済み Dictionary の attribution を `NOTICE` と source header のどちらへ記載するか。現時点では `NOTICE` へ記載している。
- patent clearance の実施担当。対象国は 2026-08-29 に日本と米国へ決定しました (上の「対象国の決定」)。担当は未定です。
- `ArUco` について、**考案者側の未登録商標としての保護** (不正競争防止法) の評価。登録は全世界に存在しないことを 2026-08-29 に確認済みです。学研ホールディングスの登録は同日、本 project の妨げにならないと判断しました。

## 関連

- [特許 Clearance 下調べ記録](patent-clearance.md)
- [Code Provenance 記録](code-provenance.md)
- [Dictionary 方針](dictionaries.md)

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
