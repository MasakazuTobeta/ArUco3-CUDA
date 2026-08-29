# 特許 Clearance 下調べ記録

> [!CAUTION]
> この文書は開発上のリスク管理を目的とした**事実の収集記録**であり、法律上の助言ではありません。侵害の有無、権利の有効性、実施の可否についての判断は一切含みません。判断は弁理士または弁護士が行います。

## この文書の位置づけ

[知的財産・ライセンス方針](ip-and-licensing.md) の `特許` 節は、商用公開前に実施する事項として次の 4 手順を定めています。本文書は **2026-08-29 に実施した web 検索と公報取得の記録**です。

本文書は 2 段階で作成されました。

1. **第 1 次調査 (2026-08-29、販売予定国が未定の状態)。** 米国の登録 claim を中心に 8 件を検証しました。
2. **第 2 次調査 (2026-08-29、販売予定国の確定後)。** **販売予定国が日本と米国に確定した**ことを受け、日本の調査を新たに実施し、米国の残件の解消を試みました。

4 手順のうちどこまで到達したかは次のとおりです。

| 手順 | 方針文書の記載 | 本文書での到達点 |
| --- | --- | --- |
| 1 | 発明者名、大学名、論文名、優先日前後の patent family を検索する | **実施した。** 発明者 5 名、Universidad de Córdoba、論文題名と特徴語、優先日帯 2013-2020 を軸に、米国と日本の双方で検索した。ただし利用できた database は限られる (後述) |
| 2 | 日本、米国、欧州および販売予定国の有効 claim を確認する | **対象国が日本と米国に確定したため、この 2 国のみを範囲とする。米国は相当程度まで到達した。日本は請求項の確認がほとんどできていない。** 詳細は下表 |
| 3 | marker Dictionary の生成方法と検出方法を分けて確認する | **実施した。** 生成側と検出側を別の軸として棚卸しと検索を行い、本 project が Dictionary を生成していない事実を code 上で確認した |
| 4 | 専門家による freedom-to-operate review の要否を判断する | **判断していない。** 判断材料の提示までにとどめる。依頼事項の候補は末尾に列挙する |

### 手順 2 の到達点 (国別)

| 国 | 到達点 |
| --- | --- |
| **米国** | **登録 claim の逐語確認は相当程度まで到達した。** 独立 claim を USPTO 公報原本 (image-ppubs の PDF を頁画像化して直読) で確認したものが 5 件 (US9558560B2、US11887312B2、US11605223B2、US12322114B2、US20250292411A1)、二次 source (FreePatentsOnline / Google Patents) の表示で読んだものが 9 件ある。法的状態は Google Patents の Legal Events 表と Unified Patents API という**二次情報**で確認した。**USPTO 自身の一次記録 (維持年金 storefront、Patent Center、Assignment Search) には 1 件も到達できていない** |
| **日本** | **本 project に最も近い 2 件の請求項 1 は確認した。他はほとんど未読。** JP7429542B2 (NVIDIA) と JP7459051B2 (Magic Leap) の請求項 1 は、調査の途中で patents.google.com が復旧したため全文を取得した (下の「日本の請求項 1 の全文」)。ほかに請求項を読めたのは 4 件 (JP2023539810A の全 19 項、JP2024507819A の請求項 1-2、特許第7496546号 の請求項 1、特許第3842515号 の請求項 1 抜粋)。**従属請求項はいずれも未読。JP6493163B2 (オムロン) の請求項も未読。** 法的状態は Google Patents の表示に依拠しており、**J-PlatPat での一次確認は 1 件も行っていない** |
| 欧州・スペイン・中国 | **対象外とした。販売予定国に含まれないという判断による範囲外化であり、「調べたが権利が無かった」ではありません。** 第 1 次調査で記録した ES2894549A1 / ES2894549B2 (Seabery)、EP4195182A4、CN111435438B、CN116075875A は、いずれも権利の存否・claim・法的状態を確認していないまま範囲外にしています。**販売予定国を欧州・中国へ広げる場合、これらは未調査の状態から再開する必要があります** |

```mermaid
flowchart LR
    S1["手順 1<br/>発明者・大学・論文・family"] --> S2["手順 2<br/>有効 claim の確認<br/>(対象国: 日本・米国)"]
    S2 --> S3["手順 3<br/>Dictionary 生成 と 検出 の分離"]
    S3 --> S4["手順 4<br/>FTO review の要否判断"]
    S1 -.->|実施| D1["本文書に記録"]
    S2 -.->|米国: 相当程度| D1
    S2 -.->|日本: 近接 2 件の請求項 1 を確認<br/>他は書誌のみ| D1
    S2 -.->|欧州・中国: 対象外<br/>(判断による範囲外化)| D3["調査せず"]
    S3 -.->|実施| D1
    S4 -.->|未実施| D2["専門家へ引き継ぐ"]
```

## 目的

ArUco3-CUDA の技術特徴を棚卸しし、それぞれについて「**日本または米国**で関係しうる第三者特許を web 検索の範囲で発見できたか」を、再現可能な形 (検索語、database、実施日、取得 URL) で記録します。否定的結果 (探したが見つからなかった) も同じ重みで記録します。

この記録の用途は次の 2 つです。

- 専門家へ freedom-to-operate review を依頼する場合の入力資料とする。
- 依頼するかどうかを判断するための材料を、事実と不確実性に分けて提示する。

## 対象範囲

### 含むもの

- 本 repository の検出 pipeline S0-S11 の技術特徴 (`docs/design/detector-pipeline.md`、`src/core/*`)。
- marker Dictionary の**生成方法**と**検出・識別方法** (`tools/dictgen/`、`src/dictionary/`)。
- GPU 実装に固有の構成 (workspace 管理、CUDA Graph、並列 reduction、決定性の確保)。
- 上記に関係しうる**日本および米国**の特許出願・登録特許のうち、**実際に page を取得して番号と内容を確認できたもの**。

### 含まないもの

- 侵害の有無、claim の有効性、実施の可否についての評価。
- 網羅的な調査。本記録は 2026-08-29 に web 検索で辿れた範囲にとどまります。
- 商標、意匠、著作権および license の論点。著作権・license は [知的財産・ライセンス方針](ip-and-licensing.md) と [Code Provenance 記録](code-provenance.md) が扱います。
- **欧州・スペイン・中国の権利。販売予定国が日本と米国に確定したことによる範囲外化であり、調査した結果として権利が無かったという意味ではありません。**

## 現状

### 実施条件

| 項目 | 内容 |
| --- | --- |
| 実施日 | 2026-08-29 (第 1 次調査・第 2 次調査とも) |
| 販売予定国 | **日本と米国に確定** |
| 実施方法 | web 検索および公報 page の取得。**対話型 database の操作は行っていない。これが日本側の到達点を決定づけた** |
| 対象 repository | `/home/tobeta/ArUco3-CUDA`。第 1 次調査の grep は commit `65f1f41`、第 2 次調査の grep は commit `912a17c` に対して実行し直した |
| 検証の原則 | 番号は、実際に取得した page に記載されていたものだけを記録する。取得していない番号は記録しない |
| 本 repository に関する事実 | commit `912a17c` に対する grep で、corner 検出器 (`harris` / `goodFeatures` / `cornerHarris` / `FAST` / `shi-tomasi`)、角度計算 (`atan2` / `acos` / `asin` / `angle`)、色空間変換 (`cvtColor` / `COLOR_` / `rgb` / `yuv` / `hsv`)、`extendDictionary` 呼び出し、display 系 (`imshow` / `display` / `GL_` / `vulkan`)、frame 間追跡 (`previous.?frame` / `prev_frame` / `track`)、template matching (`template.?match` / `matchTemplate`)、edge 検出 (`canny` / `edgel` / `line.?segment` / `hough`)、`running.?buffer` / `label.?connection` がいずれも 0 件であることを確認した |

### 使用できた database と使用できなかった database

**この表は本記録の限界そのものです。**「見つからなかった」という記述は、すべてここに挙げた到達可能な経路の範囲での話です。

#### 米国

| Database | 状態 | 用途と備考 |
| --- | --- | --- |
| USPTO Patent Public Search 画像配信 (image-ppubs.uspto.gov) | 使用可 | 公報 PDF を安定して取得できた。ただし **PDF は text layer を持たない画像のみ**で、`pdftotext` では 21 byte しか抽出できない。`pdftoppm` で頁画像化して目視で読んだ (環境に OCR は無い)。**claim 全文の一次確認はこの経路で行った** |
| FreePatentsOnline (freepatentsonline.com) | 使用可 | claim 全文の取得と claim 限定検索 (`ACLM/`) に使用。ただし **claim 数の書誌値が原本と食い違う実例を確認した** (US9558560B2 を 25 claim と表示、原本は 16 claim) |
| Google Patents (patents.google.com) | **第 1 次では HTTP 503 が継続、第 2 次では一部の担当で HTTP 200 に回復** | US 公報の Legal Events (維持年金・譲渡・審査経過) の確認に使用。**回復は担当・時間帯によりばらつき、日本公報 (`/patent/JP.../ja`) は第 2 次でも複数の担当で 503 だった。安定した経路ではない** |
| Unified Patents API (api.unifiedpatents.com/patents/US-\<番号\>-\<kind\>) | 使用可 | 認証なしで JSON を返す。出願番号・登録日・満了予測・現権利者・publication_status の**独立した第 2 情報源**として使用。維持年金の納付イベント自体は持たない |
| USPTO Maintenance Fee Storefront (fees.uspto.gov) | **使用不可 (HTTP 403)** | SPA と JS bundle は取れ、内部 API path (`/mntfee-services/v1/maintenancefee/details` ほか) も特定できたが、CloudFront が `Request blocked.` を返す。**維持年金の一次記録は取得できていない** |
| USPTO Patent Center (patentcenter.uspto.gov) | **使用不可** | 全 path が同一の SPA 外殻 (17430 byte) を返し、backend API path を特定できなかった。**包袋・審査記録に到達できていない** |
| USPTO Open Data Portal API (api.uspto.gov) | **使用不可 (HTTP 401)** | API key が必要 |
| USPTO Patent Assignment Search (assignment.uspto.gov) | **使用不可 (DNS NXDOMAIN)** | **譲渡記録の一次照会ができていない。** REEL/FRAME 番号は Google Patents の転載で読んだだけ |
| USPTO bulk data / PatentsView / PEDS | **使用不可 (DNS NXDOMAIN または redirect)** | 維持年金イベントの全件 file、審査経過 API のいずれにも到達できず |
| PTAB (developer.uspto.gov) | **使用不可** | IPR / PGR の一次記録を確認していない。Unified Patents の `num_challenged=0` を見ただけ |
| qrcode.com (DENSO WAVE 特許 page) | 使用可 | 隣接分野の番号一覧として使用。**存続・失効の記載は一切なく、権利状態の記録としては使えない** |

#### 日本

**日本については、到達できなかった経路の方が圧倒的に多いことがこの記録の中心的な事実です。**

| Database | 状態 | 用途と備考 |
| --- | --- | --- |
| FreePatentsOnline の JP コレクション (`jp=on`) | **使用可。日本公報へ到達できた主要な経路** | 書誌 (和文・英文題名、出願番号、出願日、公報発行日、出願人、IPC、引用文献、代理人) と**英訳された要約**まで取得できる。**請求項は収録されていない。JP 公報の PDF も配信していない** (`JP<番号>.pdf` は `PDF for this patent/application is not available.` を返す)。索引は英訳要約が対象で、日本語のまま検索する手段は無い。並列に 3 本以上要求すると `ECONNRESET` で切断される |
| J-GLOBAL (jglobal.jst.go.jp) の detail ページ | **部分的に使用可** | `https://jglobal.jst.go.jp/detail?JGLOBAL_ID=<ID>` は取得できる。**日本語の請求項を読めた数少ない経路** (特許第7496546号 の請求項 1 全文、特許第3842515号 の請求項 1 抜粋)。ただし**検索 URL の形式を特定できず** (`/search/patent?q=` は HTTP 404)、WebSearch で detail ページを見つけて個別取得するしかない |
| lens.org の番号照会 (`/lens/patent/JP_<番号>_<kind>`) | **番号を知っていれば部分的に使用可** | JP2023539810A の請求項 1-19 をこの経路で読めた。ただし他の番号 (JP_7429542_B2) では本文が完全に空で返り、**番号によって取得可否が変わる**。検索 URL (`/lens/search/patent/list?q=`) は JS 描画で結果を得られない |
| patentfield.com の `/patents/<出願番号>` | **部分的に使用可 (bot 判定あり)** | JP2024507819A の請求項 1-2 と経過情報 (拒絶理由通知・補正の日付) をこの経路で読めた。ただし他の試行では「アクセスを確認しています…」の bot 判定画面のみ。**未ログイン検索の回数制限があり、正しいカタカナ表記での再検索ができなかった** |
| EPO Open Data / SPARQL (data.epo.org/linked-data) | **部分的に使用可** | JP 公報の書誌と family (公報種別 A / B の有無) の照会に使用。endpoint は `/linked-data/query` が正しく `/linked-data/sparql` は HTTP 406。**出願人名・発明者名の述語がデータセットに無いため、名前による検索には使えない**。`publication/JP/7429542/B2` は書誌を返さなかった |
| r.jina.ai 経由の Google Patents | **部分的に使用可** | JP7429542B2 の Google Patents 表示 (Status: Active、満了予測 2040-01-14) をこの経路で得た。**ただし page によって取得できる範囲にばらつきがあり、Claims 節は返らない** |
| patent.nweon.com | 使用可 | AR/VR 分野の公報要約 |
| **J-PlatPat (j-platpat.inpit.go.jp)** | **使用不可。本記録の最大の欠落** | host は生存し HTTP 200 を返すが、本文は `j-platpat` と `Loading...` のみの JavaScript SPA。公報固定アドレスの形式 (`/c1801/PU/JP-<登録番号>/15/ja` が特許公報、`/11/` が公開公報、`/10/` が出願) は 2 つの独立した解説記事から確認できたが、**どの URL でも本文が描画されない**。旧形式の直接 access URL (`/serv/wsc05?type=list&no=`) は HTTP 400。JS bundle (`main-DHU77CDT.js`、12MB) を解析して内部 REST API と JSON schema、検索項目コード (発明者 `INVENTOR_DEVICE_CREATOR_AUTHOR`、出願人 `APPN_RIGHT_HOLDER_HOLDER`) まで特定したが、`/app/comsrch/wsp0101` への POST は WAF (`Server: PWS`) により HTTP 302 で `/reject_sorry.html` へ飛ばされ、`/web/patnumber/wsp0102` は payload によらず常に `{"RSLT_CD":1,"MSG_ID":"C90001-E","SEARCH_HIT_CNT":0}` を返した。**「J-PlatPat で確認した」と言える結果は本記録に 1 件もありません** |
| Google Patents の JP 公報 (`/patent/JP.../ja`) | **使用不可 (HTTP 503)** | 複数の担当が 2 回以上試行してすべて 503。mirror の patents.glgoo.top も HTTP 403 |
| Espacenet (worldwide / at / es / de / nl) | **使用不可 (HTTP 403 または DNS 不可)** | 国別 mirror も同様。**INPADOC の法的状態と family の独立確認ができていない** |
| WIPO PATENTSCOPE (patentscope / patentscope2) | **使用不可 (HTTP 403、または JS 描画で本文なし)** | |
| ipforce.jp | **ほぼ使用不可** | 出願人一覧 page (`/applicant-148832`、出願人名「エヌビディア　コーポレーション」) だけは取得でき、日本での NVIDIA 登録が多数あることは確認できた。**個別公報 page は延べ 9 回以上の試行がすべて HTTP 500 / システムエラー / 60 秒 timeout**。`www.` 付きは TLS 証明書の altnames 不一致、`en.` は DNS 不可。**日本語の請求項を持つ数少ない無料 site なので、時間を置いた再試行の価値は残る** |
| patentimages.storage.googleapis.com | **使用不可** | 日本の特許公報 PDF (JP7493117B2.pdf、JP7117418B2.pdf を実例として確認) を配信していることは判ったが、path が hash 4 階層のため**番号から URL を構成できない**。JP7429542B2.pdf は検索でも surface しなかった。直接 access は HTTP 403 |
| ekouhou.net | **使用不可 (消滅)** | **特許 database としては消滅しており、ドメインが別内容 (WordPress の blog / オンラインポーカー比較) に転用されている。** 第 1 次調査で候補に挙げていた経路が使えなくなった |
| chizai-watch.com / patentfield の公報 page / tokkyo.ai | **使用不可 (bot 判定または URL 形式不明)** | |
| j-tokkyo.com (HTTP 403) / patentjp.com (TLS エラー) / tokkyoj.com (無関係な site へ転用) / patentguru.com (HTTP 468) / DEPATISnet (redirect loop) / Unified Patents portal (JS shell) / tokkyo.shinketsu.jpo.go.jp・shinketsu.jpo.go.jp・jstore.jst.go.jp (DNS 不可) | **使用不可** | いずれも日本の請求項を得る経路にならなかった |
| patent-i.com | **使用不可 (JP)** | 米国出願人 report は取得できたが、JP 側の NVIDIA 出願人 ID を特定できず (ID 0000105 は別法人のもの) |
| 開放特許情報データベース (plidb.inpit.go.jp) | **実質使用不可** | HTTP 200 だが文字エンコーディングが壊れ和文題名が判読不能。収録は開放特許のみで網羅性が無い |
| WebSearch (一般 web 検索) | **日本公報の発見手段としては機能しなかった** | 「デンソーウェーブ 特許 二次元マーカー 検出 AGV」等の日本語 query では、index が米国中心のため日本公報 page も番号も返らない。返るのは企業の製品紹介と特許庁の一般解説のみ。唯一 ipforce.jp の 2024 年 NVIDIA 登録一覧に 7429542 が含まれるという snippet が得られた。なお第 2 次調査では session の web 検索予算 (200 件) を使い切った担当があり、裏取りができなかった |

### 実施した検索の軸と検索語

再現のため、実施した検索を軸ごとに記録します。検索語は代表的なものを挙げます。

#### 米国 (第 1 次調査)

| 軸 | 主な検索語 | 主な database |
| --- | --- | --- |
| 発明者・出願人 | `inventor=Munoz-Salinas` / `Garrido-Jurado` / `Marin-Jimenez` / `Medina-Carnicer` / `Romero-Ramirez`、`IN/"muñoz salinas"` (発音区別符号あり・なしの両形)、`AN/"universidad de cordoba"` | Google Patents, FreePatentsOnline |
| 論文題名・要旨の特徴語 | `"squared fiducial marker"`、`"inter-marker distance"`、`"marker dictionary"`、`SPEC/"mixed integer linear programming" AND SPEC/"fiducial marker"`、`OREF/"Speeded up detection of squared fiducial markers"` | Google Patents, FreePatentsOnline |
| 縮小戦略 (ArUco3 の中核) | `ACLM/"fiducial marker" AND (ACLM/"reduced resolution" OR ACLM/"downsampled" OR ACLM/"downscaled")`、`coarse-to-fine corner detection pyramid propagate` | Google Patents, FreePatentsOnline |
| 候補抽出・二値化 | `ACLM/"marker" AND ACLM/"adaptive threshold" AND ACLM/"quadrilateral"`、`ACLM/"marker" AND ACLM/"connected component"`、`connected component labeling GPU union-find` | FreePatentsOnline, Google Patents |
| Dictionary 生成 | `fiducial marker dictionary generation mixed integer linear programming`、`generating marker code set "minimum Hamming distance" rotations` | Google Patents, USPTO 画像配信 |
| Dictionary 照合 | `ACLM/"fiducial marker" AND ACLM/"Hamming distance"`、`ACLM/"fiducial marker" AND (ACLM/"codeword" OR ACLM/"dictionary")` | FreePatentsOnline |
| GPU 実装 | `GPU accelerated fiducial marker detection`、`stream compaction prefix sum GPU`、`Otsu thresholding GPU parallel histogram`、`packed atomicMax argmax` | Google Patents, FreePatentsOnline |
| family 展開 (第 2 次) | `SPEC/"corner detection" AND AN/"Magic Leap"`、`SPEC/"connected component labeling" AND SPEC/"graphics processor"`、`TTL/"integral image"`、`AN/"Millennium Three"`、`TTL/"connected component"` | FreePatentsOnline |

#### 日本 (第 2 次調査)

すべて FreePatentsOnline の JP コレクション (`jp=on`) に対する expert 検索です。**同 collection は英訳された要約と題名しか索引していないため、日本語の技術語をそのまま検索する手段はありませんでした。** 指示された和語 (「二次元マーカー 検出」「AR マーカー 認識」「基準マーカー 姿勢推定」「正方形マーカー 識別」「マーカー 辞書 ハミング距離」「マーカー 四隅 サブピクセル」「適応的二値化 マーカー」) はすべて英語へ置き換えて実施しています。

| 軸 | 検索式 | 件数 |
| --- | --- | --- |
| 出願人 (NVIDIA) | `AN/"nvidia"` | 692 |
| 出願人 (NVIDIA) × marker | `AN/"nvidia" AND SPEC/"fiducial marker"` | 4 |
| 出願人 (NVIDIA) × marker | `AN/"nvidia" AND TTL/"marker"` / `AN/"NVIDIA" AND ABST/"marker"` | 2 / 2 |
| marker 一般 | `TTL/"fiducial"` / `ABST/"fiducial marker"` / `TTL/"fiducial marker"` | 69 / 124 / 10 |
| marker 一般 | `ABST/"two-dimensional marker"` / `ABST/"AR marker"` | 20 / 167 |
| marker × 形状 | `ABST/"square marker" OR ABST/"rectangular marker"` / `ABST/"marker" AND ABST/"quadrangle"` / `ABST/marker AND ABST/quadrilateral` | 9 / 7 / 9 |
| marker × 四隅 | `ABST/"four corners" AND ABST/"marker"` / `ABST/"marker" AND ABST/"corner"` | 60 / 158 |
| marker × 照合 | `ABST/"marker" AND ABST/"Hamming"` | **0** |
| marker × 辞書 | `ABST/"marker" AND ABST/"dictionary"` | 62 (全て電子辞書・かな漢字変換) |
| marker × subpixel | `ABST/"marker" AND ABST/"subpixel"` | 2 (同一 family) |
| ArUco 言及 | `SPEC/"AprilTag"` | 110 |
| GPU × marker | `SPEC/"fiducial marker" AND SPEC/"graphics processing unit"` / `ABST/"marker" AND (ABST/"GPU" OR ABST/"graphics processing unit")` | 69 / 1 |
| 縮小戦略 | `ABST/"reduced image" AND ABST/marker` / `ABST/"low resolution" AND ABST/"high resolution" AND ABST/marker` / `ABST/marker AND ABST/"reduced resolution"` | 7 / 6 / **0** |
| 縮小戦略 | `ABST/"reduction ratio" AND ABST/detection AND ABST/size` / `ABST/coarse AND ABST/fine AND ABST/search AND ABST/resolution` / `ABST/"image pyramid"` | 71 / 7 / 36 |
| 縮小戦略 | `ABST/corner AND ABST/"first resolution" AND ABST/"second resolution"` / `ABST/subpixel AND ABST/corner` / `ABST/"reduced image" AND ABST/candidate` | 6 / 10 / 44 |
| 二値化 | `ABST/"binarization" AND ABST/"parallel"` / `ABST/"binarization" AND (ABST/"GPU" OR ABST/"graphics processing")` / `ABST/"adaptive binarization"` | 187 / **0** / 22 |
| 二値化 | `ABST/"adaptive threshold" AND ABST/"parallel"` / `ABST/"plurality of thresholds" AND ABST/"binariz"` | 1 / **0** |
| Otsu | `ABST/"Otsu"` / `ABST/"discriminant analysis" AND ABST/"threshold"` | 37 (全て日本語の「乙」noise) / 22 |
| ラベリング | `ABST/"connected component"` / `ABST/"connected component" AND ABST/"label"` / `ABST/"label" AND ABST/"equivalence"` | 178 / 6 / 9 |
| ラベリング | `ABST/"labeling" AND (ABST/"SIMD" OR ABST/"GPU" OR ABST/"graphics processing")` / `ABST/"labeling" AND ABST/"parallel processing"` | 11 / 3 |
| scan | `ABST/"prefix sum" OR ABST/"parallel prefix"` / `ABST/"stream compaction" OR ABST/"scan operation" OR ABST/"cumulative sum"` | 19 / 651 (語が広すぎ) |
| CUDA Graph | `ABST/"graph code"` / `AN/"NVIDIA" AND ABST/"kernel"` / `ABST/"kernel" AND ABST/"launch"` | 17 / 3 / 7 |
| ArUco 著者 | `IN/"munoz salinas"` / `IN/"garrido jurado"` / `IN/"marin jimenez"` / `IN/"medina carnicer"` / `IN/"romero ramirez"` | **すべて 0** (陽性対照 `IN/"suzuki"` は 553,039 件) |
| UCO・Seabery | `AN/"universidad de cordoba"` / `AN/"seabery"` | **0** / 0 (陽性対照 `AN/"universidad nacional de cordoba"` は 2 件) |
| 日本企業 | `AN/"DENSO WAVE" AND ABST/"marker"` / `ABST/"marker" AND (AN/"OMRON" OR AN/"MITSUBISHI ELECTRIC" OR AN/"DENSO") AND ABST/"image"` | 77 / 228 |
| 日本企業 | `ABST/"marker" AND (AN/"FANUC" OR AN/"YASKAWA" OR AN/"KEYENCE")` / `ABST/"marker" AND AN/"MURATA MACHINERY"` | 94 / 32 |
| 日本企業 | `ABST/"marker" AND ABST/"recognition" AND (AN/"SEIKO EPSON" OR AN/"SONY" OR AN/"TOSHIBA" OR AN/"PANASONIC" OR AN/"RICOH" OR AN/"FUJITSU")` | 110 |
| 日本企業 (語の置換) | `AN/"CANON" AND ABST/"index" AND ABST/"position and orientation"` | 24 (Canon は marker を「指標 = index」と表記するため) |
| US family の JP 展開 | `AN/"Intel" AND SPEC/"connected component"` / `AN/"Texas Instruments" AND SPEC/"integral image"` / `AN/"Millennium Three"` | 12 (無関係) / 1 / **0** |
| 日本語 (patentfield 全文) | `ムニョス サリナス` / `ガリード フラド` / `ArUco` | 2 (無関係) / **0** / 4650 |

**FreePatentsOnline の JP コレクションに対する `ACLM/` (請求項) field 検索は信頼できないと判断しました。** `ACLM/"fiducial marker"` は 170 件と応答しましたが、個別 page に Claims section が存在しないことと矛盾し、`AN/` と組み合わせると 3 回とも `ECONNRESET` になりました。**したがって、第 1 次調査で米国について行った `ACLM/` 検索と同等のことを、日本については一切実施できていません。**

### 発見した特許 — 米国 (番号を独立に検証したもの)

第 1 次調査で 8 件を検証し、第 2 次調査で claim の逐語確認と法的状態の確認を進めました。**実在しなかった番号は 1 件もありませんでした。**

| 番号 | 種別 | 題名 | 出願人 | 優先日 | 状態 (確認できた範囲) | 確認 URL |
| --- | --- | --- | --- | --- | --- | --- |
| US11113819B2 | 登録 | Graphical fiducial marker identification suitable for augmented reality, virtual reality, and robotics | NVIDIA | 2019-01-15 | 出願 16/248,135。2021-09-07 登録。**2025-02-20 に 4 年目維持年金の納付イベント (MAFP, M1551, large entity) が記録されている。** 譲渡は 2019-03-01 の発明者 → NVIDIA (REEL/FRAME 048475/0643) のみ。失効・年金不納のイベントは無い。Google Patents / Unified Patents とも Active、満了予測 2039-05-03。**8 年目年金の期限は USPTO 記録で未確認** | [Google Patents](https://patents.google.com/patent/US11113819B2/en) / [claim 全文](https://www.freepatentsonline.com/11113819.html) |
| US12322114B2 | 登録 | Graphical fiducial marker identification | NVIDIA | 2019-01-15 (継続) | 出願 17/393,615。2025-06-03 登録。**維持年金の納付イベントは無いが、登録が新しく 4 年目年金の期間が到来していないためであり、不納付ではない (期間未到来であること自体は USPTO 記録で未確認)。** 譲渡は 2021-08-04 (REEL/FRAME 057077/0790)。満了予測 2039-12-12 | [USPTO 公報 PDF](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12322114) / [Google Patents](https://patents.google.com/patent/US12322114B2/en) |
| US20250292411A1 | **出願 (公開公報)** | Graphical fiducial marker identification | NVIDIA | 2019-01-15 (継続の連鎖) | **出願番号 19/225,463。2025-06-02 出願、2025-09-18 公開。US12322114B2 の継続であることが公報原本の Related U.S. Application Data 欄で確認できた。** Legal Events は 2 件のみ (2025-06-12 審査待ち、2025-07-01 譲渡) で、**2026-08-29 時点で拒絶理由通知・許可通知・放棄のいずれのイベントも無く、審査未着手のまま係属中と読める** | [USPTO 公報 PDF](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/20250292411) / [Google Patents](https://patents.google.com/patent/US20250292411A1/en) |
| US11430212B2 | 登録 | Methods and apparatuses for corner detection | Magic Leap | 2018-07-24 (仮出願 62/702,477) | 出願 16/520,119。2022-08-30 登録。**2026-01-22 に 4 年目維持年金の納付イベント (MAFP, M1551, large entity) が記録されている。** 譲渡・担保設定の記録が 4 件あり、うち 3 件は JPMorgan Chase / Citibank 宛の security interest (担保設定であって所有権移転ではない)。**担保権が現在も有効かは未確認。** 満了予測 2039-09-03 | [USPTO 公報 PDF](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11430212) / [Google Patents](https://patents.google.com/patent/US11430212B2/en) |
| US9558560B2 | 登録 | Connected component labeling in graphics processors | Intel | 2014-03-14 | 出願 14/210,585。2017-01-31 登録。**維持年金の納付イベントが 2 回ある (2020-06-25 に 4 年目 M1551、2024-03-27 に 8 年目 M1552)。本記録で 2 回の納付が確認できた唯一の件。** 12 年目は未納付だが期限は未到来 (期限自体は未確認)。譲渡は 2014-04-07 の Intel 宛のみ。満了予測 2034-09-20 | [USPTO 公報 PDF](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/9558560) / [claim 全文](https://www.freepatentsonline.com/9558560.html) |
| US11100649B2 | 登録 | Fiducial marker patterns, their automatic detection in images, and applications thereof | Millennium Three Technologies (発明者 Mark Fiala) | 2014-05-21 | 出願 16/787,546。2021-08-24 登録。**2025-02-24 に 4 年目維持年金の納付イベント (MAFP, M3551, micro entity) が記録されている。** entity status は 2020-02 に undiscounted → micro へ変更。譲渡は 2020-02-11 の Fiala → Millennium Three (REEL/FRAME 051894/0627) のみ。満了 2035-05-21。**公開公報は US20200211198A1 で、第 1 次調査が別候補として列挙していた番号は同一出願だった** | [Google Patents](https://patents.google.com/patent/US11100649B2/en) / [claim 全文](https://www.freepatentsonline.com/11100649.html) |
| US20230316669A1 | **出願 (公開公報)** | Augmented Reality or Virtual Reality System with Active Localisation of Tools | Seabery North America, Inc. | 2020-08-10 (ES P202030858) | 出願 18/020,172。2023-10-05 公開。**2026-01-09 に最終拒絶が発送され、最終後応答と advisory action を経て 2026-03-26 に再び審査待ちへ戻っている。許可通知も放棄も記録されておらず、2026-08-29 時点で係属中。** 譲渡は 2023-03-27 (REEL/FRAME 063109/0600)、譲渡人に **GARRIDO JURADO, SERGIO** が明記されている | [USPTO 公報 PDF](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/20230316669) / [Google Patents](https://patents.google.com/patent/US20230316669A1/en) |
| ES2894549A1 / ES2894549B2 | 出願公開 / 登録 | Sistema de realidad aumentada o realidad virtual con localización activa de herramientas | Seabery Soluciones S.L. | 2020-08-10 | **販売予定国に含まれないため対象外とした。** 第 1 次調査で「Google Patents 表示は Active、満了予測 2040-08-10、OEPM での年金納付状況は未確認」と記録したまま、第 2 次調査では一切追跡していない | [ES2894549B2](https://patents.google.com/patent/ES2894549B2/en) |

第 2 次調査で claim と法的状態を確認した米国の追加 4 件です。

| 番号 | 種別 | 題名 | 出願人 | 状態 (確認できた範囲) | 確認 URL |
| --- | --- | --- | --- | --- | --- |
| US11887312B2 | 登録 | (Millennium Three の AR system) | Millennium Three Technologies | 出願 17/404,514 (US11100649B2 の継続であることを公報原本で確認)。2024-01-30 登録、154(b) 調整 198 日。**維持年金イベントは無いが登録が新しく期間未到来。放棄・失効のイベントは無い**。claim 総数 23、独立 claim は 1 と 7 | [USPTO 公報 PDF](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11887312) |
| US11605223B2 | 登録 | Methods and apparatuses for corner detection | Magic Leap | 出願 17/814,494 (US11430212B2 の継続)。2023-03-14 登録。**維持年金イベントは無い。** Citibank 宛 security interest が 2 件。claim 総数 20、独立 claim は 1 / 19 / 20 | [USPTO 公報 PDF](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11605223) |
| US11682099B2 | 登録 | Hardware accelerator for integral image computation | Texas Instruments | US11132569B2 の継続 (出願 17/462,493)。2021-08-31 出願、2023-06-20 登録。Google Patents 表示は Active、満了 2039-09-05。**第 2 次調査で新たに page 上で番号を確認した** | [FreePatentsOnline](https://www.freepatentsonline.com/11682099.html) |
| US5726435 | 登録 | Optically readable two-dimensional code and method and apparatus using the same | Nippondenso / 豊田中央研究所 (現権利者 Denso Wave) | **Google Patents の Status 欄は "Expired - Lifetime"、満了 2015-03-14。本記録で失効が確認できた唯一の件。** qrcode.com の一覧にも掲載されているが、同 page には存続・失効の記載が一切無い | [Google Patents](https://patents.google.com/patent/US5726435A/en) |

### 発見した特許 — 米国 (第 2 次調査で claim を読んだが、二次 source によるもの)

次は claim 本文を FreePatentsOnline または Google Patents の表示で読んだもので、**USPTO 一次公報での逐語照合は行っていません。**

| 番号 | 題名 | 出願人 | 状態 (Google Patents 表示) | 確認 URL |
| --- | --- | --- | --- | --- |
| US7769236B2 | Marker and method for detecting said marker | 当初 National Research Council of Canada、**現権利者 Millennium Three Technologies (2014-10 譲渡)** | Active、満了 2029-05-03 | https://www.freepatentsonline.com/7769236.html |
| US10366307B2 | Coarse-to-fine search method, image processing device and recording medium | OMRON | Active、満了 2036-12-20。**日本出願 2015-218761 を優先権主張** | https://www.freepatentsonline.com/10366307.html |
| US10504231B2 | Fiducial marker patterns, their automatic detection in images, and applications thereof | Millennium Three Technologies | Active、満了 2035-05-21 | https://www.freepatentsonline.com/10504231.html |
| US20200211198A1 | 同上 (公開公報) | Millennium Three Technologies | **US11100649B2 として登録済み。同一出願の公開公報であり、独立した候補ではなかった** | https://www.freepatentsonline.com/y2020/0211198.html |
| US9042652B2 | Techniques for connected component labeling | Intel | Active、満了 2032-12-25 | https://www.freepatentsonline.com/9042652.html |
| US11132569B2 | Hardware accelerator for integral image computation | Texas Instruments | Active、満了 2039-11-27 | https://www.freepatentsonline.com/11132569.html |
| US7725518B1 | Work-efficient parallel prefix sum algorithm for graphics processing units | NVIDIA | Active、満了 2028-07-22。**2010 年登録であり 11.5 年年金の納付期限を通過している。lapse / reinstatement の行は表示されなかったが、表示が無いことは納付の証明ではない** | https://www.freepatentsonline.com/7725518.html |
| US20220366688A1 | Methods and apparatuses for corner detection (公開公報) | Magic Leap | **US11605223B2 として登録済み。同一出願の公開公報** | https://www.freepatentsonline.com/y2022/0366688.html |
| JP2938338 / JP2867904 / JP3716527 / JP3726395 / JP3843595 / JP3996520 / US5691527 / US7032823 | DENSO WAVE が自社 QR code 特許として列挙している番号群 | DENSO WAVE | **列挙 page に存続・失効の記載は一切無い。** 利用方針として "Everyone can use the QR Code freely as long as following the standards for QR Codes in JIS or ISO." のみが記されている | https://www.qrcode.com/en/patent.html |

### 発見した特許 — 日本

**すべて FreePatentsOnline の JP コレクションで書誌と英訳要約まで確認したものです。特記しない限り請求項は読めていません。法的状態は 1 件も一次確認していません。**

#### 本 project との技術的な近さが高いもの

| 番号 | 種別 | 題名 | 出願人 | 書誌 (確認できた範囲) | 請求項 | 確認 URL |
| --- | --- | --- | --- | --- | --- | --- |
| **JP7429542B2** (特許第7429542号) | 特許公報 | 拡張現実、仮想現実、ロボティクスに適したグラフィカルな基準マーカ識別 | Nvidia Corporation (代理人「あさむら特許事務所」と表示) | 出願番号 特願2020-003331、出願日 2020-01-14、公報発行日 2024-02-08、IPC G06T7/60・G06T7/136。優先日 2019-01-15 (US11113819B2 と整合)。引用文献に AprilTag 2 (Wang & Olson, IROS 2016) と柴田ら SumiTag (GPGPU による AR マーカ, 2011)。r.jina.ai 経由の Google Patents 表示は Status Active、満了予測 2040-01-14 | **請求項 1 の全文を読んだ (下の「日本の請求項 1 の全文」を参照)。** 構成要件は色空間変換、境界画素の識別、候補コーナ点の検出、しきいピクセル距離による絞り込み、多角形の判定、フィルタによる基準マーカ境界の識別。**US11113819B2 の claim 1 と同じ構成である** | [FPO](https://www.freepatentsonline.com/JP7429542B2.html) |
| JP2020119550A (特開2020-119550) | 公開公報 | 同上 | NVIDIA CORP | 出願番号 特願2020-003331 (JP7429542B2 と同一出願)、公開日 2020-08-06 | **未読。** 抄録に「計算コストの高い処理を GPU 122 で実装」「複数の処理ブロックを GPU 上で並列実装」「従来手法に比べ計算時間を削減」と明記されており、**GPU/CPU の処理分担が公開されている** | [FPO](https://www.freepatentsonline.com/JP2020119550A.html) |
| **JP7459051B2** (特許第7459051号) | 特許公報 | 角検出のための方法および装置 | Magic Leap, Inc. | 出願番号 JP2021503756、出願日 2019-07-23、公報発行日 2024-04-01。**US11430212B2 (出願 16/520,119、Filed Jul. 23, 2019) と出願日・題名・出願人・要約が一致し、日本 family member であることを確認した** | **請求項 1 の全文を読んだ (下の「日本の請求項 1 の全文」を参照)。** 前提部は**「ユーザによって頭部に装着されるために構成される装置」**であり、画面とカメラシステムを構成要件に含む。**US claim 1 の頭部装着装置の前提部は日本の請求項にも存在する。** 一方 **Harris 系定数 k1 は日本の請求項 1 に現れず**、代わりに「重複する検出された角の除去」とその実行順序が限定として入っている | [FPO](https://www.freepatentsonline.com/JP7459051B2.html) |
| JP2021530816A (特表2021-530816) | 公開公報 | 同上 | Magic Leap, Inc. | 出願番号 JP2021503756 (JP7459051B2 と同一出願)、公開日 2021-11-11 | **未読。** 要約が日本語で「第１の解像度」「より低い解像度の第２の画像」と明示 | [FPO](https://www.freepatentsonline.com/JP2021530816A.html) |
| JP2024019662A | 公開公報 | METHOD AND DEVICE FOR ANGLE DETECTION (角検出のための方法および装置) | MAGIC LEAP INC | 出願番号 JP2023215648、出願日 2023-12-21、公開日 2024-02-09。**親出願の登録公報発行 (2024-04-01) の直前という時期関係から分割出願と見られるが、分割である旨の記載は確認していない** | **Google Patents 表示では 2025-02-25 に拒絶理由通知、2025-08-21 に拒絶査定。** 拒絶査定不服審判の請求の有無は未確認。**査定が確定していれば権利は発生しない** | [FPO](https://www.freepatentsonline.com/JP2024019662A.html) |
| JP6493163B2 (特許第6493163号) | 特許公報 | 粗密探索方法および画像処理装置 | オムロン株式会社 | 出願番号 JP2015218761A、出願日 2015-11-06、公報発行日 2019-04-03。**US10366307B2 が優先権主張している日本出願 2015-218761 に対応する登録** | **未読。** 要約は「第1画像からテンプレートマッチングで検出し、解像度の異なる第2画像で検出位置に対応する領域を探索領域として設定し、SIMD 命令により n×m 回未満の演算で照合する」 | [FPO](https://www.freepatentsonline.com/JP6493163B2.html) |
| JP2017091103A (特開2017-091103) | 公開公報 | 同上 | オムロン株式会社 | 同一出願 (JP2015218761A) の公開公報、公開日 2017-05-25 | **未読** | [FPO](https://www.freepatentsonline.com/JP2017091103A.html) |
| 特許第7496546号 / 特開2021-189697 | 特許公報 | 画像処理方法、プログラム及び画像処理システム | パナソニックIPマネジメント株式会社 | 出願 特願2020-093719 | **請求項 1 の全文を J-GLOBAL で読めた。** 要旨は「1 回目は第 1 の圧縮率の対象画像から第 1 候補画像を探索し、k+1 回目は圧縮率のより小さい画像で k 回目の候補を含む予備候補に絞り込んで探索する。さらに相関情報に基づくサブ候補画像の追加を含む」 | [J-GLOBAL](https://jglobal.jst.go.jp/detail?JGLOBAL_ID=202403017263535351) |
| 特許第3842515号 / 特開2001-273091 | 特許公報 | 多重解像度画像解析による指示位置検出 | セイコーエプソン株式会社 | 出願 特願2000-084101 (2000 年出願) | **請求項 1 の抜粋を J-GLOBAL で読めた。** 「撮像画像から段階的に解像度を下げた低解像度画像を作製し、低解像度画像から指示位置に対応する画素を求め、撮像画像内の小区画に基づいて指示位置を求める」 | [J-GLOBAL](https://jglobal.jst.go.jp/detail?JGLOBAL_ID=201103068314341500) |
| JP2023539810A (特表2023-539810) | 公開公報 (PCT 国内移行の公表公報。**登録公報ではない**) | ツール、使用、関連プロセスの動的位置特定を備えた、拡張現実又は仮想現実システム | シーベリー・ソルシオネス・ソシエダッド・リミターダ (SEABERY SOLUCIONES SL) | 出願 特願2023-509749、出願日 2021-07-30 (PCT/ES2021/070582)、優先権 ES P202030858 (2020-08-10)、公表日 2023-09-20 | **請求項 1-19 の全文を lens.org で読めた。** 請求項 1 は「オブジェクト + ツール + AR/VR ビューアー + 第 1/第 2 の光学手段 + 処理装置」という装置構成。**マーカの検出手順 (二値化、候補四角形抽出、Dictionary 照合、縮小率の決定、pyramid による四隅復元) に相当する限定は請求項 1-19 のいずれにも現れない。** マーカに関する限定は請求項 8 の「LED、QR コード、バーコード、再帰反射球、プリントされたマーカ、キーポイント」という列挙のみ | [lens.org](https://www.lens.org/lens/patent/JP_2023539810_A) |
| JP2024507819A (特表2024-507819) | 公開公報 (**登録公報ではない**) | 拡張現実感環境又は仮想現実感環境において使用するのに適しているシミュレーションデバイス | シーベリー・ソルシオネス・ソシエダッド・リミターダ | 出願 特願2023-549990、出願日 2022-02-18 (WO2022175583)、公表日 2024-02-21。**第 1 次調査に記載が無かった 2 件目の Seabery 日本出願** | **請求項 1-2 を patentfield で読めた。** 請求項 1 はロッド・タッチアクチュエータ・指の動きの符号化手段からなる物理的な入力デバイス。マーカは請求項 2 で空間的位置手段の選択肢の 1 つとして現れるのみで、検出手順の限定は無い。IPC は G06F3/0346・G06F3/01 で画像処理ではなく入力装置の分類。**発明者に Garrido Jurado は含まれない** | [patentfield](https://patentfield.com/patents/JP2023549990A) |

Seabery 2 件の経過情報 (patentfield 表示、2026-08-29 時点) は次のとおりです。**いずれも係属中で未登録であり、日本には有効な登録請求項が存在しません。**

- JP2023539810A: 審査請求 2024-06-07 → 拒絶理由通知 2025-08-19 → 補正 2025-11-18 → 拒絶理由通知 2026-02-17 → 補正 2026-05-15 → 拒絶理由通知 2026-08-18。最終処分「未処分」、登録日なし。**読めたのは公表公報 A の請求項であり、3 回の拒絶理由通知を経た現在の請求項は未確認。**
- JP2024507819A: 審査請求 2025-01-17 → 調査報告 2025-12-26 → 拒絶理由通知 2026-01-13 → 補正 2026-04-13 → 拒絶理由通知 2026-07-14。最終処分「未処分」、登録日なし。

#### 日本の請求項 1 の全文

第 2 次調査の報告時点では日本の請求項を 1 件も読めていませんでしたが、**その後 patents.google.com が復旧したため、対象市場で最も近い 2 件の請求項 1 を取得しました。** 取得日は 2026-08-29、経路は `patents.google.com/patent/<番号>/ja` です。

**JP7429542B2 (NVIDIA、出願 2020-01-14、登録 2024-02-08、Google Patents 表示 Active)**

> 画像を表す画像データを受け取ることと、前記画像データを、高次元の色空間から、低次元の色空間の変換画像に変換することと、前記変換画像内の境界に相当する第１のピクセルを識別することと、候補コーナ点を表す第２のピクセルを検出することと、前記第１のピクセルと前記第２のピクセルとに基づいて、前記境界のうちの少なくとも１つまでのしきいピクセル距離内にある前記候補コーナ点のセットを判定することと、前記候補コーナ点の前記セットを分析して、少なくとも１つのポリゴンのコーナを表す前記候補コーナ点のサブセットを判定することと、少なくとも１つのフィルタを、前記少なくとも１つのポリゴンに適用して、基準マーカの基準マーカ境界に対応するものとして、前記少なくとも１つのポリゴンのうちのポリゴンを識別することとを備えた、方法。

本 project との**事実の差異**は次のとおりです。commit `65f1f41` に対して grep で確認しました。

| 請求項 1 の構成要件 | 本 project | 差異の確実さ |
| --- | --- | --- |
| 高次元の色空間から低次元の色空間への変換 | **無い。** 公開 API は 8 bit grayscale の `ImageViewU8` のみを受け取る。`cvtColor` / `COLOR_` / `rgb` / `yuv` / `hsv` の grep は 0 件 | **高い。** 入力型が grayscale に限定されており、変換すべき高次元の色空間が最初から存在しない |
| 候補コーナ点を表す第 2 のピクセルの検出 | **有るとも読める。** `src/core/quad_extract.cu` の `store_corner_kernel` は、連結成分ごとに極点探索 (重心からの最遠点 c0 → c0 からの最遠点 c2 → 直線 c0c2 の左右それぞれの最遠点 c1, c3) で求めた**画素の index から座標を書き出している**。Harris・FAST・goodFeatures といった**名前の付いた corner 検出器は使っていない**が、「候補コーナ点を表すピクセルを検出している」という記載には読まれうる | **低い。** 「corner 検出器を持たない」ことと「候補コーナ点を表すピクセルを検出していない」ことは別である。**ここを差異として主張することはできない** |
| 第 1 のピクセルと第 2 のピクセルに基づく、境界までのしきいピクセル距離内にある候補コーナ点のセットの判定 | **無い。** 本 project は境界画素の集合と候補コーナ点を**別々に求めて突き合わせる**構成を持たない。四隅は連結成分そのものから導かれるため、「境界までの距離が閾値以内か」で候補コーナ点を絞る段が存在しない | **中程度。** 段として存在しないことは確かだが、「連結成分に属する画素から四隅を取る」ことが実質的に同じ絞り込みにあたると解釈される余地は残る |

> [!WARNING]
> **「色空間変換が無い」という差異は、library の境界でしか成り立ちません。** 本 project の公開 API は 8 bit grayscale しか受け取りませんが、実際の製品では camera が色情報を出し、どこかで grayscale へ変換されます。**侵害の主体と対象を製品全体で見る場合、その変換は製品の中に存在します。** 本記録の対比はすべて library 単体を対象にしており、製品として組み上げたときに構成要件がそろうか (直接侵害・間接侵害・複数主体) は検討していません。**想定する製品形態を特定しない限り、この差異を根拠にできません。**

> [!WARNING]
> **当初この表は「候補コーナ点の検出も無い」と記載していました。これは誤りでした。** `harris` / `goodFeatures` の grep が 0 件であることを根拠にしていましたが、名前の付いた corner 検出器を使っていないことと、請求項の言う「候補コーナ点を表すピクセルを検出する」に該当しないこととは別の問題です。**grep の結果を請求項の充足判断に流用したのが誤りの原因です。** この種の読み替えは、専門家の判断を待たずに有利な結論へ寄せる典型例として記録します。

**JP7459051B2 (Magic Leap、出願 2019-07-23、登録 2024-04-01、Google Patents 表示 Active)**

> ユーザによって頭部に装着されるために構成される装置であって、前記ユーザのためにグラフィックを提示するように構成される画面と、前記ユーザが位置する環境を視認するように構成されるカメラシステムと、前記カメラシステムに結合される処理ユニットであって、前記処理ユニットは、第１の解像度を有する第１の画像を取得することであって、前記第１の画像は、第１の角を有する、ことと、第２の解像度を有する第２の画像を決定することであって、前記第２の画像は、前記第１の画像内の第１の角と対応する第２の角を有し、前記第２の画像は、前記第１の画像に基づき、前記第２の解像度は、前記第１の解像度未満である、ことと、前記第２の画像内の第２の角を検出することと、前記第２の画像内の第２の角の位置を決定することと、前記第２の画像内の第２の角の決定された位置に少なくとも部分的に基づいて、前記第１の画像内の第１の角の位置を決定することとを行うように構成される、処理ユニットとを備え、前記処理ユニットは、前記第２の解像度を有する前記第２の画像から検出された角のセット内の１つ以上の重複する検出された角を除去するように構成され、前記角のセットは、前記第２の角を備え、前記処理ユニットは、前記第２の画像内の前記第２の角が検出された後かつ前記第１の解像度を有する前記第１の画像内の第１の角の位置が決定される前に、前記第２の解像度を有する前記第２の画像と関連付けられた前記１つ以上の重複する検出された角を除去するように構成される、装置。

**この請求項は装置クレームであり、前提部が「ユーザによって頭部に装着されるために構成される装置」です。** 画面とカメラシステムを構成要件に含みます。本 project は library であり、画面もカメラも頭部装着の構成も持ちません。

一方、**「低い解像度で角を検出し、その位置に基づいて高い解像度の画像中の角の位置を決定する」という手順そのものは、本 project の S10 (四隅だけを段を登りながら精密化する) と構造が似ています。** 米国の対応特許 `US11430212B2` の claim 1 が持つ Harris 系定数 `k1` の限定は日本の請求項 1 には現れず、代わりに「重複する検出された角の除去」とその実行順序が入っています。**日本と米国で限定が異なります。**

**本 project を頭部装着型 AR 機器へ組み込んだ製品形態では、前提部の評価が変わりうる**点は米国と同じです。想定する製品形態を特定したうえでの評価が要ります。

#### 隣接分野として記録するもの

| 番号 | 種別 | 題名 | 出願人 | 本 project との関係 (要約から読める範囲) | 確認 URL |
| --- | --- | --- | --- | --- | --- |
| JP7365986B2 / JP2022053629A | 特許公報 / 公開公報 | カメラキャリブレーション装置、方法およびプログラム | KDDI CORPORATION | 「既知寸法の四角形マーカーを用いて四隅の点を検出」。マーカー抽出部・**線分検出部**・**交点計算部**を構成に持つ。本 project の「連結成分の全画素に対する距離最大化で極点を取る」方式とは到達手段が異なる。出願日 2020-09-25 | [FPO](https://www.freepatentsonline.com/JP7365986B2.html) |
| JP7448281B2 / JP2021157537A | 特許公報 / 公開公報 | 二次元マーカの認識装置、方法、プログラム及びシステム | NEC コミュニケーションシステムズ | 「**周辺単色領域**と**特性量表示領域**からなる**矩形領域**の誤認識を防止」。外周の黒枠 + 内部 code という構造は本 project が扱う正方形 2 値マーカーそのもの。border 検証 (S9 相当) と論点が重なりうる。2024-03-12 発行 | [FPO](https://www.freepatentsonline.com/JP7448281B2.html) |
| JP7239301B2 / JP2020080068A | 特許公報 / 公開公報 | 二次元マーカ、二次元マーカ認識処理システム | 株式会社日立製作所 | 「矩形パターンブロックがモザイク状に印刷された二次元マーカ」。**マーカーという「物」および認識処理システムを対象**とする類型。出願日 2018-11-13、2023-03-14 発行 | [FPO](https://www.freepatentsonline.com/JP7239301B2.html) |
| JP2025056491A | 公開公報 (**登録ではない**) | 位置補正装置・位置補正方法・位置補正用マーカ | デンソーウェーブ | QR code ではなく**マーカーによる位置補正**。「複数の特徴点が第 1 の配置パターンで配列され、各特徴点は第 2 の配置パターンで配列された細かい画像のセットで構成」。階層的な特徴点配置と視野変更を伴う。2023-09-27 出願、2025-04-08 公開 | [FPO](https://www.freepatentsonline.com/JP2025056491A.html) |
| JP7782311B2 / JP2023125675A | 特許公報 / 公開公報 | 位置推定装置 | 新東工業 | 「マーカーの深度情報と 4 つの角の位置情報から平面距離、方位角、深度距離を計算」。本 project の出力 (四隅 + ID) の下流。2022-02-28 出願、2025-12-09 発行で権利期間が長く残りうる | [FPO](https://www.freepatentsonline.com/JP7782311B2.html) |
| JP4250391B2 | 特許公報 | 指標検出装置、指標検出方法 | キヤノン | **題名がまさに指標 (marker) 検出そのもの。** Canon は marker を「指標 (index)」と表記するため `marker` 語では捕捉できず、語を変えて初めて出た。出願日 2002-09-13、2009-04-08 発行。**同じ検索軸で 24 件が出ており、Canon の指標系 portfolio は表面をなぞったにすぎない** | [FPO](https://www.freepatentsonline.com/JP4250391B2.html) |
| JP3802737B2 / JP2002056371A | 特許公報 / 公開公報 | 情報判別マーカ、その検出方法ほか | 電力中央研究所 | 「**正方形の四隅に配置された色付きマーク**と、それら外側の縁を結んで形成された表示領域内のアイコン」。四隅が色付きである点で本 project の 8 bit grayscale 入力とは前提が異なる。出願日 2000-08-07 | [FPO](https://www.freepatentsonline.com/JP3802737B2.html) |
| JP2024114339A | 公開公報 (**登録ではない**) | AUTONOMOUS VEHICLE SYSTEM, MARKER, AUTONOMOUS VEHICLE... | リコー | AGV が「2 次元コードと位置マーカー」で指定位置の動作を実行。marker を**使う側の車両制御**に重心 | [FPO](https://www.freepatentsonline.com/JP2024114339A.html) |
| JP7682444B2 | 特許公報 | 走行車システム、及びマーカのズレ量検出方法 | 村田機械 | マーカーの**設置誤差の管理**が対象。画像からの marker 検出 algorithm ではない。2025-05-26 発行 | [FPO](https://www.freepatentsonline.com/JP7682444B2.html) |
| JP2025538690A | 公開公報 (**登録ではない**) | ピッキングステーション上の識別マーカーを検出すること | Ocado Innovation Limited | 「**第 1 および第 2 のニューラルネットワークを連続で使用**して識別マーカーの検出とマーカー情報の認識を行う」。日本企業ではないが日本国内出願として記録。2025-11-28 公開 | [FPO](https://www.freepatentsonline.com/JP2025538690A.html) |
| JP2026021866A | 公開公報 (**登録ではない**) | 識別方法、移動体制御方法ほか | 株式会社ササキコーポレーション | 「二値化 → マーカーの配列獲得 → アドレス番号取得」の流れが本 project の S3→S8→S9 と表面的に似るが、対象は路面のガイダンスライン。2026-02-12 公開 | [FPO](https://www.freepatentsonline.com/JP2026021866A.html) |
| JP2008191760A | 公開公報 | 物体検出装置、物体検出方法およびプログラム | 豊田中央研究所 / トヨタ自動車 | 「基準値に基づいて倍率を算出し画像を縮小して検出」。**「最小寸法に基づく縮小率決定」軸で日本国内に見つかった唯一の近接文献。** ただし倍率は画像から検出した顔候補領域を基準に算出され、本 project の事前パラメータのみからの決定とは異なる | [FPO](https://www.freepatentsonline.com/JP2008191760A.html) |
| JP2011014012A | 公開公報 | 二次元コード読取方法ほか | 富士通コンピュータテクノロジーズ | 「低い解像度と品質の画像から二次元コードを読み取る」。手段は二値化とフェザリングによる暗画素密度判定で、縮小・多重解像度・四隅精密化のいずれも含まない | [FPO](https://www.freepatentsonline.com/JP2011014012A.html) |
| JP7389116B2 / JP2022508072A | 特許公報 / 公開公報 | Deep neural network pose estimation system | 未確認 | **`ABST/"marker" AND ABST/"subpixel"` の全 2 件がこの 1 family だけだったことの記録。** すなわち本 project の S10 に対応する日本公報は見つからなかったという否定的結果の裏づけ。**書誌は一切確認していない** | [FPO 検索結果](https://www.freepatentsonline.com/result.html?p=1&srch=xprtsrch&query_txt=ABST%2F%22marker%22+AND+ABST%2F%22subpixel%22&jp=on&date_range=all&stemming=on&sort=relevance&search=) |

#### GPU・並列画像処理の日本公報

| 番号 | 種別 | 題名 | 出願人 | 本 project との関係 (要約から読める範囲) | 確認 URL |
| --- | --- | --- | --- | --- | --- |
| JP7799551B2 | 特許公報 | グラフを使用したメモリ割振り | Nvidia Corporation | 出願番号 JP2022-073081、公報日 2026-01-15。本 project の「1 frame の kernel 発行列の CUDA Graph 化」「bump pointer arena」に語として最も近い日本の登録特許。**第 1 次調査で「NVIDIA 自身が graph 実行に関する特許を持つ可能性を否定できていない」と書かれていた点について、日本にその種の登録が実在することを確認した。** ただし対象は graph code ノードを生成する側 (CUDA ランタイム/ドライバ) と読める | [FPO](https://www.freepatentsonline.com/JP7799551B2.html) |
| JP2023003386A | 公開公報 | MEMORY ALLOCATION USING GRAPHS | NVIDIA (検索結果一覧の表示) | JP7799551B2 の対応公開公報とみられるが**個別 page 未取得** | [FPO 検索結果](https://www.freepatentsonline.com/result.html?p=1&srch=xprtsrch&query_txt=ABST%2F%22graph+code%22&uspat=off&usapp=off&eurpat=off&jp=on&pctap=off&depat=off&date_range=all&stemming=on&sort=relevance&search=) |
| JP2024514372A | 公開公報 (**登録ではない**) | グラフ・コードの表現を生成するためのアプリケーション・プログラミング・インターフェース | Nvidia Corporation | 出願番号 JP2022-526176、公開日 2024-04-02。CUDA Graph の graph code を扱う | [FPO](https://www.freepatentsonline.com/JP2024514372A.html) |
| JP2024514371A | 公開公報 | Application programming interface for locating incomplete graph code | NVIDIA (検索結果一覧の表示) | 同じ graph code API 群。**個別 page 未取得** | [FPO 検索結果](https://www.freepatentsonline.com/result.html?p=1&srch=xprtsrch&query_txt=ABST%2F%22graph+code%22&uspat=off&usapp=off&eurpat=off&jp=on&pctap=off&depat=off&date_range=all&stemming=on&sort=relevance&search=) |
| JP7739222B2 | 特許公報 | 演算を組み合わせるための技法 | Nvidia Corporation | GPU の reduction 演算の融合。ただし対象は深層学習コンパイラによる自動融合で、本 project は手書き kernel。2025-09-16 発行 | [FPO](https://www.freepatentsonline.com/JP7739222B2.html) |
| JP2022167854A | 公開公報 | TECHNIQUES FOR PARALLEL EXECUTION | NVIDIA CORP | 「GPU 上の並列実行の特定 + メモリ割り当て計画の生成」。主体は深層学習コンパイラ。2022-11-04 公開 | [FPO](https://www.freepatentsonline.com/JP2022167854A.html) |
| JP4237046B2 | 特許公報 | 画像処理装置 | 株式会社リコー | **日本における「並列プロセッサ上の連結成分ラベリング」の登録特許として本調査で最も近い。** SIMD の PE 列で副走査方向のみ並列化し主走査方向は逐次という 1 次元並列で、本 project の block union-find + 昇順採番とは並列化の軸が異なる。出願 2003-12-24 で**出願日から 20 年は 2023-12 に経過している (満了の可能性。未確認)** | [FPO](https://www.freepatentsonline.com/JP4237046B2.html) |
| JP2005267362A | 公開公報 | SIMD プロセッサを用いた画像処理方法および画像処理装置 | 株式会社リコー | 同じリコーの SIMD ラベリング系列。出願 2004-03-19 で 20 年は 2024-03 に経過 | [FPO](https://www.freepatentsonline.com/JP2005267362A.html) |
| JP2013191164A | 公開公報 | 画像処理装置及び方法、並びに電子装置 | RICOH CO LTD | 「ラベリング結果から Feret 径を SIMD で並列計算」。本 project の「64 bit packing した atomicMax による極点探索」と目的の面で近いが、求める量もデータ構造も異なる | [FPO](https://www.freepatentsonline.com/JP2013191164A.html) |
| JP2007334431A | 公開公報 | ラベリング処理方法ほか | シャープ株式会社 | ラベルテーブルによる等価併合。方式はラスタスキャンの逐次パイプラインで並列・GPU の記載が要約に無い。出願 2006-06 で 20 年は 2026-06 に経過する見込み | [FPO](https://www.freepatentsonline.com/JP2007334431A.html) |
| JP6998198B2 | 特許公報 | マルチバイナリゼーション画像処理 | Konica Minolta Laboratory U.S.A. | 「二値化 + 連結成分ラベリング + 連結成分の**木構造** (親子・世代)」。本 project の S9 (候補の包含木) と概念的に近い。ただし文字認識用途で、並列化・GPU の記載が要約に無い。**第 1 次調査で B 節の包含木が「発見なし」だった欄に対する、日本での最も近い所在** | [FPO](https://www.freepatentsonline.com/JP6998198B2.html) |
| JP7230294B2 | 特許公報 | オブジェクト検出のためのシステム及び方法 | Texas Instruments | 出願 JP2020156507 (2020-09-17)。**US11132569B2 (出願 16/420,152、2019-05-22) とは出願日も主題も異なり、同一 family とは確認できない。** 主題は物体検出の classifier 側で、integral image を計算する hardware accelerator 回路ではない | [FPO](https://www.freepatentsonline.com/JP7230294B2.html) |
| JP7754648B2 / JP2022002947A | 特許公報 / 公開公報 | Machine learning-based seat belt detection and usage recognition using fiducial markings | NVIDIA (JP2022002947A の出願人は未確認) | NVIDIA 名義で明細書に fiducial marker を含む日本公報。用途は車内のシートベルト検出で、正方形マーカーの検出 pipeline とは異なる。**個別 page 未取得** | [FPO 検索結果](https://www.freepatentsonline.com/result.html?p=1&srch=xprtsrch&query_txt=AN%2F%22nvidia%22+AND+SPEC%2F%22fiducial+marker%22&jp=on) |
| JPH01228071A | 公開公報 | 二値画像連結成分数計算方式 | 未確認 | 1989 年公開で権利としては消滅済み。技術としての先行例にはなりうるが clearance 上の意味は無い | [FPO 検索結果](https://www.freepatentsonline.com/result.html?p=1&srch=xprtsrch&query_txt=ABST%2F%22label%22+AND+ABST%2F%22equivalence%22&uspat=off&usapp=off&eurpat=off&jp=on&pctap=off&depat=off&date_range=all&stemming=on&sort=relevance&search=) |

#### 二次的確認にとどまるもの (個別 page を取得していない)

- **Magic Leap の corner detection には日本に 2 系統の family がある**ことが判りました。本記録が扱った US11430212B2 系 (JP2021530816A / JP7459051B2 / JP2024019662A) のほかに、neural network を併用する US11686941B2 系として **JP2022532238A / JP7422785B2 / JP2024028494A / JP7616796B2** が検索結果一覧に現れました。**一覧に表示された番号と題名を読んだだけの二次的確認です。** 本 project は neural network を使わないため優先度を下げましたが、日本を対象とする以上、別途確認する価値があります。
- NVIDIA の視線検出 family として JP7403293B2 / JP2021033997A が `AN/"nvidia" AND SPEC/"fiducial marker"` の 4 件のうち 2 件を占めていました。用途が異なるため追跡していません。
- 村田機械の走行車系として JP7396532B2 / JP2023080729A / JP2017027505A / JP2014006835A が同じ検索で表示されましたが、**個別 page を取得しておらず書誌未確認です。**

#### 誤った手掛かりを潰した記録

**特許第6708462号 (キヤノン)** は、web 検索の要約文が「粗密探索 (coarse-to-fine) の特許」として提示した番号ですが、J-GLOBAL の当該ページを実際に取得したところ内容は**金属含有色材による重ね刷りの画像形成**であり、粗密探索とも多重解像度とも無関係でした。**検索エンジンの要約を根拠に番号を報告してはならない実例として明示的に記録します。** (確認 URL: https://jglobal.jst.go.jp/detail?JGLOBAL_ID=202003000007176272)

### 技術特徴と特許の対照表

本 project の技術特徴ごとに、本調査で関係しうる特許を発見できたかを対照します。「**発見なし**」は「存在しない」ではなく「2026-08-29 の検索範囲では見つからなかった」という意味です。番号のうしろの括弧は claim または要約を読んだうえでの**事実の差異**であり、侵害・非侵害の評価ではありません。

**日本欄には重要な限界があります。** 日本の請求項をほとんど読めていないため、日本欄の記述は**要約から読める構成**にとどまり、**請求項との対照は原則として行えていません。**

#### A. ArUco3 検出戦略と pipeline 段 (S0-S11)

| 技術特徴 | 由来 | 米国 | 日本 |
| --- | --- | --- | --- |
| 想定最小マーカー辺長からの縮小率決定 `fxfy = S / (S + max(W,H) * tau)` | ArUco3 論文 + OpenCV 観測仕様 | **発見なし。** claim に「最小マーカー寸法から縮小率を逆算する」限定を持つ特許は見つからなかった | **発見なし。** `ABST/"reduction ratio" AND ABST/detection AND ABST/size` の 71 件は上位 50 件を確認した限り製鉄・複写機・回転検出器・欠陥検査で該当なし。最も近いのは JP2008191760A (豊田中研) だが、倍率は画像から検出した顔候補領域を基準に算出される点で異なる |
| 縮小画像のみでの候補探索 (segmentation 画像の単一化) | ArUco3 論文 + OpenCV 観測仕様 | **発見なし。** `ACLM/"fiducial marker" AND (ACLM/"reduced resolution" OR "downsampled" OR "downscaled")` は 2 件のみで深層学習の姿勢推定系 | **発見なし。** 見つかった粗密系の日本特許 (オムロン JP6493163B2、パナソニック 特許第7496546号) はいずれも**各解像度段で探索を繰り返す**構成で、候補探索を縮小画像 1 枚に単一化する限定を持たない。パナソニック件は請求項 1 の全文でこれを確認した |
| 検出下限辺長が 2 パラメータで規定されること | ArUco3 戦略の帰結 | **発見なし** | **発見なし** |
| 原寸からの scale-2 grayscale pyramid 構築 | ArUco3 + OpenCV | **発見なし。** `ACLM/"fiducial marker" AND ACLM/"pyramid"` は 6 件で無関係 | **発見なし。** `ABST/"image pyramid"` は 36 件で該当なし |
| 候補辺長に応じた pyramid level の選択的復号 (S7) | ArUco3 + OpenCV | **発見なし** | **発見なし** |
| **四隅だけを段を登りながら upsampling し各段で subpixel 補正 (S10)** | ArUco3 の中核 + OpenCV | **US11430212B2 (Magic Leap)。** claim 1 に「低解像度画像で corner を検出し、その位置に基づいて第 1 画像中の corner 位置を決定する」がある。ただし (a) screen と camera を備える**頭部装着装置**を前提部とし、(b) Harris 系定数 `k1 = R/(1+R)^2` を必須要件とする。継続の **US11605223B2** は独立 claim 1 / 19 / 20 の 3 本すべてが head-worn を前提部に持ち、低解像度→高解像度の corner 位置伝播型の限定は**独立 claim に現れない** (公報原本で確認)。本 project は library であり装置を持たず、corner 補正は `cornerSubPix` 相当の勾配正規方程式であって `k1` に相当する定数を持たない | **JP7459051B2 (Magic Leap、登録、Google Patents 表示 Active)。請求項 1 の全文を取得した。** 「低解像度画像で角を検出し、その位置から元の高解像度画像中の角位置を決定する」という手順は **ArUco3 中核戦略の後半そのもの**であり、本 project の S10 と構造が似る。**限定は日米で異なる。** 前提部の**頭部装着装置は日本にも存在する**が、米国 claim 1 の **Harris 系定数 `k1` は日本には現れず**、代わりに**重複コーナの除去とその実行順序**が入る。本 project は library であり画面もカメラも頭部装着の構成も持たない。従属請求項は未読。分割とみられる **JP2024019662A は 2025-08-21 に拒絶査定** (審判請求の有無は未確認) |
| 一般の coarse-to-fine 探索 (背景技術) | — | **US10366307B2 (OMRON)。** 独立 claim 1 / 5 は物体検出手段を **template matching** に限定し、さらに work memory 上の data 並べ替えと **SIMD 命令**による照合を必須要件とする。本 repository に template matching は無い。**US7769236B2 (Fiala / 現 Millennium Three)** の claim 12 / 15 は **edge detector** による edge / edgel 検出を必須要件とし、本 repository に edge 検出は無い | **JP6493163B2 / JP2017091103A (オムロン)** が US10366307B2 の日本 member として実在する。**特許第7496546号 (パナソニック)** は請求項 1 が多段解像度の候補伝播を正面から限定するが、(a) モデル画像の特徴を含む抽出画像の探索を前提とし本 project はテンプレートを持たない、(b) 各段で候補探索を繰り返す、(c) 相関情報に基づくサブ候補画像の追加を必須要件とする、(d) 最小マーカー寸法からの縮小率決定の限定は無い。**特許第3842515号 (セイコーエプソン)** は請求項 1 が段階的解像度低減 + 原画像小区画での精密化という骨格を持つが、対象は単一の指示位置 |
| ArUco3 有効時の corner refinement 強制 subpix 化 | OpenCV 由来 | **発見なし** (検索軸として立てていない) | **発見なし** (同左) |
| 補正窓半径を pyramid 段の寸法から決める | OpenCV 由来 | **発見なし** | **発見なし** |

#### B. 候補抽出・二値化・候補整理 (S3-S6, S9)

| 技術特徴 | 由来 | 米国 | 日本 |
| --- | --- | --- | --- |
| **連結成分ラベリングと極点探索による四隅抽出 (案 A)** | **本 project 独自** (ADR-0003) | **US11113819B2 / US12322114B2 / US20250292411A1 (NVIDIA)。** 技術分野が完全に一致する。**第 2 次調査で公報原本により claim ごとの所在が確定した。** US12322114B2 は独立 claim 1 / 11 / 14 の**すべてが「候補 corner 点と境界との距離を閾値画素距離と比較する」を必須要件**とする一方、**角度計算と色空間変換は従属 claim にのみ現れる** (角度は claim 10、色空間変換は claim 4 / 5 / 17)。US20250292411A1 は独立 claim 1 / 10 / 18 の**すべてが角度計算を必須要件**とし、**claim 10 のみ `corner points` ではなく単に `points` と記載され claim 1 / 18 より広い**。本 repository には corner 検出器も角度計算も無い | **JP7429542B2 (NVIDIA、登録、Google Patents 表示 Active)。請求項 1 の全文を取得した。** 構成要件は US11113819B2 claim 1 と同じ 3 要素、すなわち**高次元から低次元の色空間への変換、候補コーナ点を表すピクセルの検出、境界までのしきいピクセル距離による絞り込み**である。**3 要件のうち差異が明確なのは色空間変換だけである。** 候補コーナ点については、本 project も極点探索で四隅の画素を検出しており (`quad_extract.cu` の `store_corner_kernel`)、**「候補コーナ点を表すピクセルの検出」に該当しないとは言えない。** しきいピクセル距離による絞り込みは段として存在しないが、連結成分からの導出が実質的に同じ機能を果たすと解釈される余地は残る。従属請求項は未読。なお **NVIDIA marker family の日本の分割・継続に相当する後続出願は見つからなかった** (米国では US12322114B2 と係属中の US20250292411A1 がある) |
| 8 bit grayscale のみを入力とする構成 | 本 project の公開 API | **US11113819B2 claim 1 (NVIDIA)** が前段に「image data を高次元の色空間から低次元の色空間へ変換する」を置く。**ただし継続の US12322114B2 では色空間変換は独立 claim から外れ従属 claim へ移っており、この差異は US12322114B2 の独立 claim に対しては働かない。** 本 project の公開 API は 8 bit grayscale の `ImageViewU8` のみを受け取る | **未対照。** JP7429542B2 の抄録には色空間変換が現れるが、日本の請求項に置かれているかは未確認 |
| 四角形らしさを 2 つの比で判定 (内側比・辺の裏付け比) | **本 project 独自** | **US20250292411A1 (係属中)。** 独立 claim 3 本すべてが**辺の角度**を必須要件とし、従属 claim に辺長比 (claim 6 / 15) と辺長比の set ratio との比較 (claim 7 / 16) が現れる。本 project の判定は面積の内側比と辺の画素支持比であり、角度も辺長比も使わない。**係属中で claim が確定していない** | **発見なし。** 日本に対応する後続出願は見つからなかった |
| 複数 window の適応的二値化を並列次元として同時処理 | OpenCV + 本 project 独自 | **発見なし。** `ACLM/"marker" AND ACLM/"adaptive threshold" AND ACLM/"quadrilateral"` は 0 件 | **発見なし。** `ABST/"plurality of thresholds" AND ABST/"binariz"` は **0 件**、`ABST/"adaptive threshold" AND ABST/"parallel"` は 1 件のみで内容は磁気センサ軸受の閾値回路 |
| 分離型 row-sum / col-sum による適応的二値化 | OpenCV 意味論 + 本 project 独自 GPU 実装 | **US11132569B2 / US11682099B2 (Texas Instruments)。** 親 US11132569B2 の独立 claim 1 / 8 / 14 は integral image の計算と hardware accelerator 回路 / SOC を前提部に置く。**継続の US11682099B2 は前提部から `integral image` と `hardware accelerator` の語が外れ、`A video component` および方法 claim になっている点が事実として重要。** ただし router / segment row sum computation component / row pivot computation component という名前付き構成要素と segment ごとの row pivot を必須要件とする。本 project は integral image を作らず、行を segment へ分割せず pivot も持たない | **発見なし。** `AN/"Texas Instruments" AND SPEC/"integral image"` の JP ヒットは JP7230294B2 の 1 件のみで、出願日も主題も異なり同一 family と確認できない。`TTL/"hardware accelerator"` の JP 48 件にも integral image の accelerator は無かった。**TI は日本にも画像 accelerator 系の出願を現に持つのに、integral image accelerator だけは日本公報として見つからない**という形の否定的結果 |
| **決定的な連結成分ラベリング (block union-find + 昇順採番)** | **本 project 独自** (8 近傍の選択のみ OpenCV 由来) | **US9558560B2 (Intel)。第 2 次調査で公報原本により独立 claim 5 / 9 / 13 を確認し、第 1 次調査の残件が解消した。** claim 1 のみが `a display to present an image` を持ち、**claim 5 (方法) / 9 (記録媒体) / 13 (装置) には display の記載が一切無い** (claim 13 の前提部は `a shared memory; and a processor` のみで、claim 1 の `graphics processor` すら `processor` に緩和されている)。**したがって「本 project は library であり display を持たない」という事実は claim 5 / 9 / 13 に対しては差異として働かない。** 一方、**均質条件による fast / generic 2 経路切替は 4 本すべてが必須要件として保持**しており、本 repository にはこれに相当する記述が無い。claim 総数は原本で 16 (FPO の 25 表示は原本と一致しない) | **発見なし。** `SPEC/"connected component labeling" AND SPEC/"graphics processor"` の 66 件に現れた同 family は US9558560、US20150262369、**ドイツの DE102015001699B4** のみで JP 公報は無い。`AN/"Intel" AND SPEC/"connected component"` の JP 12 件はすべて別主題。**この family は米国とドイツに展開されているが日本には展開されていない可能性が高い**と読める (不存在の証明ではない) |
| ラベリングの一般手法 | — | **US9042652B2 (Intel)。** 独立 claim 1 / 10 / 18 は category が違うだけで実体的限定は同一。**running buffer**、その支持する更新済み label 数の上限到達を契機とする読み出し、label position の mark、running buffer 内の利用可能最小 label の判定という 4 要素を必須要件とする。本 repository に `running.?buffer` / `label.?connection` は 0 件 | **JP4237046B2 / JP2005267362A / JP2013191164A (リコー) が日本における並列ラベリングの系列として存在する。** ただし SIMD の PE 列による 1 次元並列であり、本 project の block union-find + 昇順採番とは並列化の軸が異なる。JP4237046B2 と JP2005267362A は**出願日から 20 年を既に経過している (満了の可能性。未確認)**。JP2007334431A (シャープ) も 2026-06 に 20 年経過の見込み |
| label 統計の SoA 配置と整数 atomics のみによる集計 | **本 project 独自** | **発見なし** | **発見なし** |
| 距離と画素 index を 64 bit へ packing した atomicMax | **本 project 独自の適用** | **発見なし。** 同一手法が学術論文 (arXiv) で公開技術として記載されているのを確認した | **発見なし。** 目的の面で近いのは JP2013191164A (リコー、Feret 径の並列計算) だが、求める量もデータ構造も異なる |
| 四隅折れ線の chain code 長を周長の代用にする | **本 project 独自** | **発見なし** | **発見なし** |
| 順位空間 union-find による近接候補統合と最大周長代表の選択 | OpenCV 規則 + 本 project 独自表現 | **発見なし** | **発見なし** |
| 候補の包含木と 64 bit 整数による内外判定、打ち切り段数への縮約 | OpenCV 規則 + 本 project 独自 GPU 定式化 | **発見なし** | **JP6998198B2 (Konica Minolta Laboratory U.S.A.)。** 「二値化 + 連結成分ラベリング + 連結成分の木構造 (親子・世代)」の組み合わせが概念的に近い。ただし文字認識用途で、元画像と反転画像を順次処理する構成、並列化・GPU の記載が要約に無く、木の選択基準も「文字を定義するか」という別の判断。**第 1 次調査で「発見なし」だった欄に対する、日本での最も近い所在** |
| 二値化 window 間の候補連結を device 側 counter で行う構成 | **本 project 独自** | **発見なし** | **発見なし** |
| **frame 間追跡を持たない単一 frame 検出** | 本 project の構成 | **US11100649B2 (Millennium Three / Fiala)** は独立 claim 1 / 9 が「先行 frame での検出」「先行 frame のマーカー線分 edge の追跡」「連続画像列であること」を必須要件とする。**第 2 次調査で継続出願 US11887312B2 の独立 claim 1 / 7 を公報原本で確認したところ、両 claim とも frame 間追跡を要件としていなかった。したがって「本 project は 1 frame 完結で frame 間状態を持たない」という事実は US11887312B2 に対しては差異として働かない。** 代わりに両 claim は AR system という前提部、display screen 付き capture means、marker patterns そのもの、remote / central location との送受信手段を必須要件とし、検出手順は `recognition means` という means-plus-function 形式の機能的記載にとどまる。本 project は camera も display も通信手段も持たない | **発見なし。** `AN/"Millennium Three"` の JP は **0 件** (全 collection では 11 件でいずれも US)。`TTL/"fiducial marker"` の JP 10 件にも Millennium Three / Fiala 由来のものは無い。**日本にはこの権利者の権利が存在しない可能性がある** (不存在の証明ではない) |

#### C. marker Dictionary — 生成方法 (手順 3 の「生成」側)

方針文書の手順 3 に従い、生成と検出を分けて確認しました。

| 技術特徴 | 本 project での事実 | 米国 | 日本 |
| --- | --- | --- | --- |
| 符号集合 (codeword 集合) の探索・設計 | **実装していない。** repository 全体を grep した結果、codeword を探索・最適化する処理は存在しない。`extendDictionary()` の呼び出しは 0 件。MILP solver も制約定式化も無い | **本 project は Dictionary を生成していない**ため、生成方法を対象とする方法 claim には対応する実施行為が無い。ただし marker という「物」や、Dictionary を含む装置・system claim には同じ論法が及ばない可能性がある (専門家判断) | 同左。**加えて日本では marker という「物」の請求項を持つ公報が実在する** (JP7239301B2 日立「二次元マーカ、二次元マーカ認識処理システム」)。**請求項未読のため対照はしていない** |
| 回転を含む inter-marker Hamming 距離の最大化による符号設計 | 実装していない (MIP 論文由来の思想だが手順は未実装) | **US7769236B2 (Fiala)。** 明細書に「全 marker 間を全ての相対回転を考慮して最近接 2 marker の Hamming 距離の最小値を最大化するよう符号を選ぶ」という**開示**がある。独立 claim は 1 / 12 / 15 の 3 本で、claim 1 は marker という**物**の claim、claim 12 / 15 は edge detector を必須要件とする検出方法 | **発見なし。** `ABST/"marker" AND ABST/"Hamming"` は **0 件**で、第 1 次調査の米国側 (`ACLM/"fiducial marker" AND ACLM/"Hamming distance"` が 0 件) と整合する |
| MILP による Dictionary 生成 | 実装していない | **発見なし。** 該当技術は学術文献としてのみ確認できる | **発見なし** |
| OpenCV `bytesList` から 64 bit packed への変換、4 回転の事前展開 table | **本 project 独自のデータ表現** | **発見なし** | **発見なし** |
| 生成時の回転規則自己検証、`--check` による byte 一致検証 | **本 project 独自の build 方式** | **発見なし** | **発見なし** |
| marker image の描画 | **自前実装を持たない。** 描画はすべて OpenCV の API 呼び出しで、用途は test fixture と評価 corpus 生成に限られる | US10817743B2 (未検証)、US11107243B2 (未検証)。本 project は marker を設計も生成もしない | **発見なし** (この軸で日本を検索していない) |

#### D. marker Dictionary — 検出・識別方法 (手順 3 の「検出」側)

| 技術特徴 | 由来 | 米国 | 日本 |
| --- | --- | --- | --- |
| canonical 画像全体への Otsu 閾値決定と GPU 3 相分割 | Otsu 法は公知 + OpenCV 意味論 + 本 project 独自の分解 | **発見なし。** GPU での Otsu 並列化を claim する特許は見つからず、Otsu を引用する古い二値化特許と学術論文のみ | **発見なし。** `ABST/"Otsu"` の 37 件は**すべて日本語の「乙」(焼酎の乙類、乙型土留) による noise** で検索が機能しない。代替語 `ABST/"discriminant analysis" AND ABST/"threshold"` の 22 件はいずれも逐次処理で、並列ヒストグラムによる Otsu の GPU 化を対象とするものは無かった |
| 内側領域の平均・母標準偏差による低分散分岐 | OpenCV 由来 | **発見なし** | **発見なし** |
| セル単位の白画素比 (bit へ潰さない中間表現) | OpenCV 由来 + 本 project 独自のデータ構造 | **発見なし** | **発見なし** |
| 外周セルによる border 検証 | OpenCV 由来 | **発見なし** | **JP7448281B2 (NEC コミュニケーションシステムズ)** が「周辺単色領域と特性量表示領域からなる矩形領域」の誤認識防止を要約に掲げており、border 検証と論点が重なりうる。**請求項未読のため対照はしていない** |
| 2 mask による 3 値表現と分岐なし Hamming 距離式 + popcount | OpenCV 意味論 + 本 project 独自の packed 表現 | **発見なし。** `ACLM/"fiducial marker" AND ACLM/"Hamming distance"` は **0 件**。`ACLM/"fiducial marker" AND (ACLM/"codeword" OR ACLM/"dictionary")` は 1 件のみで MRI 中の金属物体位置決定に関するもの | **発見なし。** `ABST/"marker" AND ABST/"Hamming"` は **0 件**。`ABST/"marker" AND ABST/"dictionary"` の 62 件は**全て電子辞書・かな漢字変換・語学学習機器**で、marker 辞書との照合を扱うものは 1 件も無かった |
| 「ID 昇順で最初に許容距離を満たした ID」を採る規則の atomicMin による再現 | OpenCV 規則 + 本 project 独自の並列等価実装 | **発見なし** | **発見なし** |
| 4 回転を事前展開した packed codeword の device 常駐化 | **本 project 独自** | **発見なし** | **発見なし** |
| 射影変換とセル sampling の演算順序まで一致させた canonical 画像生成 | OpenCV 由来 + 本 project 独自の GPU 構成 | **発見なし** | **発見なし** |
| 照合で得た回転による四隅の並べ替え打ち消し | OpenCV 由来 | **発見なし** | **発見なし** |
| ID による重複除去を行わない出力規則 | OpenCV 由来の挙動 | **発見なし** | **発見なし** |

#### E. GPU 実装に固有の構成

| 技術特徴 | 由来 | 米国 | 日本 |
| --- | --- | --- | --- |
| 全段 GPU 常駐の marker 検出 pipeline | **本 project 独自の段構成** | **US11113819B2 (NVIDIA)。** 明細書が GPU 上での並列実行を前提とし、`THRESHOLDER` / `IMAGE SEGMENTER` / `CORNER DETECTOR` / `CORNER FILTER` / `QUAD FITTER` / `DECODER` / `GPU(s)` を構成要素として列挙する。**pipeline 全体像として最も近い family** | **JP2020119550A (NVIDIA)** の抄録が「計算コストの高い処理を GPU で実装し、複数の処理ブロックを GPU 上で並列実装する」と明記しており、**GPU 常駐 pipeline という本 project の構成と技術分野が正面から重なる**。`ABST/"marker" AND (ABST/"GPU" OR ABST/"graphics processing unit")` の JP ヒットは **1 件のみ**で、それがこの NVIDIA family だった |
| device 常駐結果を同期なしで返す出力 API | **本 project 独自** | **発見なし** | **発見なし** |
| bump pointer arena による frame ごとの確保回避 | **本 project 独自の適用** | **発見なし** | **JP7799551B2 (NVIDIA、グラフを使用したメモリ割振り)** が語として近い。対象は graph code ノードを生成する側と読める。**請求項未読** |
| 設定上限からの最悪値事前確保と単調性による上界導出 | **本 project 独自** | **発見なし** | **JP2022167854A (NVIDIA)** の要約に「GPU 上で並列実行しうる演算の特定 + メモリ割り当て計画の生成」があり表面上重なるが、主体は深層学習コンパイラ |
| 上限付き device buffer と 2 本の counter による overflow 通知 | **本 project 独自** | **発見なし** | **発見なし** |
| **1 frame の kernel 発行列の CUDA Graph 化と無効化契機の管理** | CUDA 機能 + 本 project 独自の無効化規則 | **発見なし。** 「command buffer / execution graph / kernel launch replay」で検索したが特許は 1 件も現れず、NVIDIA 公式文書・blog・論文のみ | **JP7799551B2 (登録)、JP2024514372A / JP2024514371A (公開)。第 1 次調査が「NVIDIA 自身が graph 実行に関する特許を持つ可能性を否定できていない」と書いていた点について、日本に該当する登録と公開出願が実在することを確認した。** ただし対象は CUDA ランタイム/ドライバ側で graph code を生成する構成と読め、CUDA Graph API を呼び出すだけの application との関係は請求項を読まないと判らない。**請求項未読** |
| 入力内容に依存しない固定 kernel 発行列 | **本 project 独自** | **発見なし** | **発見なし** |
| device の SM 数から起動 block 数を導く自動調整 | **本 project 独自** | **発見なし** | **発見なし** |
| 1 block が 1 隅を担当し grid-stride で隅を跨ぐ割り当て | **本 project 独自** | **発見なし** | **発見なし** |
| 自前 3 段 scan による stream compaction | 公知の古典技法 + 本 project の自前実装 | **US7725518B1 (NVIDIA)。** 独立 claim 1 / 9 / 15 の 3 本すべてが同一の限定を含み、部分列長を `K*(J+1)+L` (K は local memory 構造の interleave size、(J+1) は同時処理する thread 数、L は factor) という式で限定する。本 project の sub-list 長は `constexpr int kScanThreads = 256` の 1 定数で決まり、この形の寸法規定を持たない。3 段構成 (b)(c)(d) は本 project の `scan_within_block_kernel` → `scan_block_sums_kernel` → `add_block_offsets_kernel` と概念上対応する | **発見なし。** US7725518B1 は Google Patents 上で US のみが表示され **JP member が無い**。`ABST/"prefix sum" OR ABST/"parallel prefix"` の JP 19 件は秘密計算 (NTT 系の秘匿集計) と ReLU 圧縮が中心で、画像処理の stream compaction を対象とするものは無かった |
| 数値再現性を保証したまま並列化する規約 (`-fmad=false`、順序保存 reduction) | **本 project 独自の方法論** | **発見なし** | **発見なし** |
| memory 空間の型による区別と device 属性に基づく経路選択 | **本 project 独自の設計** | **発見なし** | **発見なし** |
| caller-owned stream と最小同期契約 | **本 project 独自** | **発見なし** | **発見なし** |
| CPU / GPU ハイブリッド経路との切替可能構成 | **本 project 独自の構成判断** | **US11113819B2 claim 16 (NVIDIA)。** 「GPU が候補 corner 点と境界画素を求め、閾値距離以内の集合を GPU memory から CPU memory へ複写し、以降を CPU が行う」。本 project の案 C は labeling まで GPU、四隅抽出を CPU の輪郭追跡へ委譲する構成 | **JP2020119550A** の抄録が GPU と CPU の処理分担を明記しており、**US claim 16 に対応する構成が日本でも開示されている**ことを示す。**日本の請求項に置かれているかは未確認** |

### 本記録の比較は文言上の対比にとどまる

**本記録が「本 project には無い」と書いているのは、すべて請求項の文言と実装を直接つき合わせた結果です。** 次の 2 つは一切検討していません。

1. **均等論。** 日本 (最高裁平成 10 年ボールスプライン事件が示す 5 要件) でも米国 (doctrine of equivalents) でも、**文言上の差異があっても均等の範囲に入ると判断されることがあります。** 本記録はこの検討をしていません。したがって「構成要件が文言上そろわない」ことは、それだけでは何の結論にもなりません。
2. **クレーム解釈。** 請求項の用語が明細書と審査経過に照らしてどこまでの範囲を指すかは、確定していません。実例として、本記録は当初 NVIDIA `JP7429542B2` の「候補コーナ点を表すピクセルの検出」を「本 project には無い」と記載していましたが、**これは誤りでした** (上の WARNING)。`harris` の grep が 0 件であることを、請求項の用語の充足判断に流用したためです。

**この 2 つを検討しない比較は、clearance の結論を出す手続ではありません。** 本記録の位置づけは「専門家がどこを見るべきかを絞り込むための材料」であり、それ以上ではありません。文言上の差異が多く見つかったことを、リスクが低いことの根拠として読まないでください。

### 否定的結果の記録

clearance では否定的結果それ自体が成果物です。**すべて「存在しない」ではなく「2026-08-29 の検索範囲で見つからなかった」です。**

#### 米国・日本に共通するもの

1. **ArUco / ArUco3 の著者を発明者とする、marker 検出・生成の特許は 1 件も見つかりませんでした。** Rafael Muñoz-Salinas、Sergio Garrido-Jurado、Manuel J. Marín-Jiménez、Rafael Medina-Carnicer、Francisco J. Romero-Ramírez のいずれかを発明者とし、正方形 fiducial marker の生成または検出を対象とするものは、米国・日本のいずれでも発見できませんでした。
2. **Universidad de Córdoba を出願人とする marker 関連特許も見つかりませんでした。** 米国側は patentados.com の UCO 名義 109 件を確認し、内容はバイオ・診断、材料、農業機械に集中していました。日本側は FPO の JP コレクションで `AN/"universidad de cordoba"` が **0 件**です。**この 0 件は陽性対照 (`AN/"universidad nacional de cordoba"` が 2 件) により出願人のフレーズ検索が機能していることを確認したうえでの結果であり、比較的信頼できます。**
3. **論文著者名は「引用される側」としてのみ特許文献に現れました。** Google Patents の全文検索で `"Munoz-Salinas"` 37 件、`"Garrido-Jurado"` 91 件がヒットしましたが、いずれも他社特許が ArUco 論文を非特許文献として引用しているものでした。
4. **ArUco3 論文の中核 (縮小率の逆算 + 縮小画像単一化 + pyramid による四隅復元) を claim する特許は、米国でも日本でも見つかりませんでした。** これは本 repository の S1-S10 に最も近い技術特徴であり、該当なしという結果は clearance 上の意味があります。
5. **Dictionary 生成手法 (MILP による符号集合設計) を claim する特許は見つかりませんでした。**
6. **claim / 要約中に `fiducial marker` (または `marker`) と `Hamming distance` を同時に含むものは、米国 claim 検索でも日本要約検索でも 0 件でした。**
7. UCO の研究グループ AVA の ArUco portfolio ページには**特許への言及が一切なく**、GPLv3 と商用問い合わせ先のみが記載されています。これは技術が特許ではなく license (著作権) で管理されていることを示唆しますが、非公開の出願や失効した出願の不存在を意味しません。

#### 日本に固有のもの

8. **日本の一次 database (J-PlatPat) には到達できませんでした。** 内部 REST API の endpoint・JSON schema・検索項目コードまで特定しましたが、WAF により拒否されました。**「J-PlatPat で確認した」と言える結果は本記録に 1 件もありません。**
9. **和文の請求項を読める汎用の経路は 1 つも見つかりませんでした。** 到達できた日本公報の主要経路 (FPO の JP コレクション) は書誌と英訳要約までで、Claims section を持ちません。JP 公報の PDF も配信されていません。**本記録が日本の請求項を読めたのは、番号を知ったうえで lens.org / patentfield / J-GLOBAL がたまたま応答した 4 件だけです。**
10. **日本の法的状態 (存続・失効・年金納付・無効審判・異議申立) は 1 件も確認していません。** 登録公報 (B2) が発行された事実は確認できますが、それは発行時点の話であって現在の存続を意味しません。
11. **NVIDIA marker family の日本の分割出願・継続に相当する後続出願は見つかりませんでした。** 米国では US12322114B2 と係属中の US20250292411A1 が同 family にありますが、FPO の JP 収録範囲ではこれらに対応する日本出願は現れません。
12. **GPU で正方形マーカーを検出する日本特許は、NVIDIA のもの以外に見つかりませんでした。** `SPEC/"fiducial marker" AND SPEC/"graphics processing unit"` の JP 69 件のうち 1 位が JP7429542B2 で、残りは医療 (手術ナビゲーション、血管アクセス、眼科)、産業 (実装機、3D 印刷、除草)、航空機でした。
13. **`ABST/"fiducial marker"` の JP 124 件は、上位 50 件を確認した限り外科手術ナビゲーション、MRI・ガンマ線による位置同定、歯科インプラント、顕微鏡校正、ロボット誘導などの医療・計測系に集中しており、正方形 2 値マーカーの検出高速化を扱うものは 1 件もありませんでした。**
14. **`SPEC/"AprilTag"` の JP 検索は 110 件あり、そのうち本 project の技術分野に近いのは JP7429542B2 と JP2020119550A のみでした。** 他は医療 AR、ロボット姿勢計測、物流、点検が中心です。
15. **出願人軸で空振りだったもの。** ファナック・安川電機・キーエンスを出願人とする `marker` 出願 94 件は、大半がキーエンスの「レーザマーカ」(刻印装置) と三次元座標測定機で、正方形 2 値マーカーの検出に関するものは 1 件も見つかりませんでした。オムロン・三菱電機・デンソーを含む 228 件、セイコーエプソン・ソニー・東芝・パナソニック・リコー・富士通を含む 110 件も、**上位 25 件の範囲では** marker 検出 algorithm を対象とするものを特定できませんでした。**これは各社が該当特許を持たないことの証明ではなく、要約の英訳語彙に依存した検索の限界です。**
16. **米国 family のうち、日本に対応出願が見つからなかったもの。** Intel US9558560B2 (米国とドイツのみ)、Millennium Three の portfolio (`AN/"Millennium Three"` の JP が 0 件)、Texas Instruments US11132569B2 (JP family 番号が表示されない)、NVIDIA US7725518B1 (US のみ)。**いずれも不存在の証明ではありません。**

### 番号の実在性検証と、その限界

- **報告された特許番号のうち、実在しなかったものは 1 件もありませんでした。** 第 1 次調査の 8 件に加え、第 2 次調査では第 1 次で「二次的確認」にとどまっていた **JP7429542B2 を独立の公報 page で実在確認**し (Google Patents の family 一覧の二次情報ではなく FPO の個別公報 page)、番号の読み取り誤りでないことを確認しました。
- **第 1 次調査の誤りが 2 つ判明しました。** (a) US20250292411A1 と「US 19/225,463」を別個の 2 件として列挙していましたが、**両者は同一の出願**です (公開番号と出願番号の関係)。(b) US20200211198A1 を US11100649B2 と別の候補として列挙していましたが、**同一出願の公開公報**でした。同様に US20220366688A1 は US11605223B2 の公開公報です。
- **US11605223B2 の claim 1 の情報源間不一致は決着しました。** 公報原本 30 欄を確認したところ、**FreePatentsOnline の読み (定数 `k1` 型) が原本と一致**し、Google Patents 経由で報告された「第 1 解像度 / 第 2 解像度」型の claim 1 は US11605223B2 の claim 1 ではありませんでした (それは親 US11430212B2 側の記載です)。**同じ経路 (Google Patents) で得た他の記載も、同様に原本での確認を要します。**
- **FreePatentsOnline の書誌値が原本と食い違う実例を確認しました。** US9558560B2 の総 claim 数を 25 と表示しましたが、公報原本は claim 16 で終わっています。claim 本文自体は原本と一致しました。
- **検索エンジンの要約が誤った特許番号を提示する実例を確認しました。** 特許第6708462号 (キヤノン) が「粗密探索の特許」として要約文中に現れましたが、実際に page を取得すると金属含有色材による画像形成の特許でした。
- **表記ゆれで検索結果が変わることを 2 通り実測しました。** (a) 発音区別符号: `IN/"munoz salinas"` (ñ なし) は 0 件、`IN/"muñoz salinas"` (ñ あり) は 4 件。(b) カタカナ表記: 日本公報での Garrido Jurado の表記は「**ガルリド フラド**」であり、試した「ガリード フラド」では 0 件になります。**「0 件」という結果は表記ゆれの影響を受けており、不存在の証明ではありません。**
- **FPO の日本コレクションの発明者索引に欠落があることを実測しました。** JP2023539810A の公報原本には発明者 3 名 (ガルリド フラド，セルヒオ ほか) が明記されているのに、FPO の同公報ページには Inventors 欄自体が存在しません。陽性対照 `IN/"suzuki"` は 553,039 件返るので索引自体は生きていますが、**新しい公表公報の一部が発明者索引に入っていません。したがって日本側の `IN/` の 0 件は不存在の証明になりません。**
- **日本公報の英訳題名は機械翻訳です。** 実際に `corner detection` が `angle detection` と訳されており (JP7459051B2 の英訳題名)、`TTL/"corner detection"` では日本の family を取り逃がします。**題名ベースの 0 件は訳語のゆれの影響を強く受けます。**
- 検索結果の抜粋にのみ現れ、page を取得できなかったため**意図的に報告しなかった番号が複数あります。** これらは「存在しないと判断した」のではなく「確認できなかったので書かなかった」ものです。
- US4939354 (Data Matrix の基本特許とされるもの) は、**他の公報の References Cited 欄に印字された番号を読んだだけ**で、内容も権利状況も未検証です。

### 本調査で確認できなかったこと

| 項目 | 状態 |
| --- | --- |
| **日本の請求項** | **本 project に最も近い 2 件の独立請求項 1 は取得した** (JP7429542B2、JP7459051B2)。ほかに読めたのは 4 件 (JP2023539810A、JP2024507819A、特許第7496546号、特許第3842515号)。**従属請求項はいずれも未読。JP6493163B2 (オムロン) の請求項も未読。** 日本の公報全体で見れば、請求項を読めたのはごく一部にとどまる |
| **日本の法的状態** | **1 件も確認していない。** 存続・失効・年金納付・権利者移転・無効審判・異議申立のいずれも未確認。J-PlatPat の経過情報に到達できなかった |
| **日本の分類検索** | **未実施。** FI / F-term / CPC による網羅検索を一切行っていない。すべて英訳要約の技術語による検索であり、訳語のゆれによる漏れが構造的に残る |
| **日本の請求項ベース検索** | **未実施。** FPO の JP コレクションに対する `ACLM/` 検索は信頼できないと判断した。第 1 次調査で米国について行った claim 限定検索と同等のことを日本については行えていない |
| **日本の下位ヒット** | **未確認。** FPO は逐次取得が必要で、各検索の上位 25-50 件しか確認していない。総件数が 100 件を超える軸 (`unmanned` 166、`AR marker` 167、`position and orientation` 106、`fiducial marker` 124、`binarization AND parallel` 187、`GPU AND image` 346 ほか) は下位が未確認 |
| **日本国内のみの出願** | **網羅できていない。** 使えた検索経路が FPO の JP コレクション 1 本しかなく、米国 family を持たない日本国内出願は捕捉しきれていない |
| **米国の一次法的状態** | **未確認。** USPTO の維持年金 storefront・Patent Center・Assignment Search・bulk data のいずれにも到達できなかった。本記録の維持年金・譲渡・審査経過は **Google Patents の Legal Events 表 (USPTO の PALM トランザクションコードの転載) と Unified Patents API という二次情報**に依拠している |
| 米国の次回年金期限 | **未確認。** 登録日から計算はできるが、計算値と USPTO の記録は区別すべきであるため本記録には書いていない |
| 米国の譲渡記録原本 | **未確認。** REEL/FRAME 番号は読んだが、USPTO Assignment Search で記録原本を照会していない。とくに Magic Leap 案件の Citibank 宛 security interest が現在も有効かは未確認 |
| 米国の包袋 | **未読。** US20250292411A1 と US20230316669A1 について、拒絶理由通知・意見書・補正書の実物を 1 件も読んでいない。審査の現況は Legal Events のステータスコード列から読み取ったもの |
| 米国の IPR / PGR | **未確認。** Unified Patents の `num_challenged=0` を見ただけで、PTAB の一次記録では確認していない |
| 従属 claim | **大半が未読。** 全文を読んだのは US12322114B2 と US20250292411A1 のみ。US9558560B2 の claim 2-4 / 6-8 / 10-12 / 14-16、US11605223B2 の claim 2-18、US11887312B2 の claim 2-6 / 8-23、US7769236B2 の 15 件、US9042652B2 の 21 件、US7725518B1 の 18 件、US11132569B2 / US11682099B2 の各 17-18 件は未読 |
| 米国の CPC 分類による網羅検索 | **未実施。** すべて技術語による検索であり、語のゆれによる漏れが生じている |
| 譲受人 (assignee) 名での網羅検索 | **未実施。** ARToolKit 系 (HITLab / DAQRI)、PTC / Vuforia、Qualcomm、Microsoft、Apple、Boeing、Cognex、Zebra については該当特許を 1 件も特定できていないが、これは不在の根拠にならない。**第 2 次調査で US7769236B2 の現権利者が Millennium Three Technologies であることが判明し、同社は US10504231B2 / US11100649B2 / US11887312B2 も保有するため、出願人軸での網羅に価値があるが実施していない** |
| 引用・被引用関係の追跡 | **未実施。** とくに US11113819B2 の被引用を辿ると同領域の後発出願が出る可能性が高い |
| 公開前の出願 | **原理上確認不能。** 出願から 18 か月未満の出願はどの database でも見えない |
| **欧州・スペイン・中国の権利** | **対象外 (判断による範囲外化)。** 販売予定国に含まれないため調査していない。ES2894549A1 / ES2894549B2、EP4195182A4、CN111435438B、CN116075875A、DE102015001699B4、DE102020100684B4 の claim と法的状態は未確認のまま。**販売予定国を広げる場合、これらは未調査の状態から再開する必要がある** |

## 目標

- **日本の請求項を入手できる状態にする。** J-PlatPat を対話型 browser で開くか、公報固定アドレスサービスを申し込む (INPIT 知財情報部 情報提供担当) か、有償 database または専門家へ委託する。**本記録の残件のうち最も重いものです。**
- **日本の法的状態 (存続・年金・審判) を一次 database で確認できる状態にする。** 現状は 1 件も確認できていません。
- **米国の法的状態を USPTO の一次記録で裏づける。** 現状は Google Patents と Unified Patents という二次情報に依拠しています。
- 本記録を入力資料として、専門家による freedom-to-operate review の要否 (手順 4) を判断できる状態にする。
- 判断の結果を [知的財産・ライセンス方針](ip-and-licensing.md) の `Patent review` 欄と [Code Provenance 記録](code-provenance.md) へ反映し、実装単位ごとの review 状態を追跡可能にする。

## 未確定事項

### 対象国の確定により解消した事項

- ~~販売予定国が未定である~~ → **日本と米国に確定しました。** これにより手順 2 の範囲が定まり、欧州・スペイン・中国は対象外となりました。**対象外は判断による範囲外化であり、権利の不存在を意味しません。**
- ~~US11605223B2 の claim 1 の情報源間不一致~~ → **公報原本で決着しました** (FreePatentsOnline の k1 型が正しい)。
- ~~US11887312B2 の claim が未読~~ → **独立 claim 1 / 7 を公報原本で読みました。** frame 間追跡は要件ではありませんでした。
- ~~US9558560B2 の独立 claim 5 / 9 / 13 が未読~~ → **公報原本で読みました。** display 要件を持たない独立 claim が実在しました。
- ~~US20250292411A1 と US 19/225,463 が別件かどうか~~ → **同一出願でした。**
- ~~US20200211198A1 が US11100649B2 と別の候補かどうか~~ → **同一出願の公開公報でした。**

### 残っている未確定事項

- **均等論とクレーム解釈を一切検討していません。** 本記録の比較は文言上の対比にとどまります (上の「本記録の比較は文言上の対比にとどまる」)。**これは残件のうち最も本質的なもので、専門家でなければ埋められません。**

- **JP7429542B2 (NVIDIA) と JP7459051B2 (Magic Leap) の請求項 1 は取得しました** (上の「日本の請求項 1 の全文」)。**ただし従属請求項と他の独立請求項は未読で、法的状態も一次確認していません。** Google Patents の Active 表示に依拠しています。維持年金の納付、審査経過での補正の経緯、無効審判の有無はいずれも未確認です。
- 同一出願の公開公報 JP2020119550A の請求項が未読です。公開時と登録時で請求項は異なりうるため、補正の経緯を追うには両方が要ります。
- JP7429542B2 の**従属請求項**が未読です。独立請求項 1 との差異が本 project の構成に及ぶかは判断できていません。
- JP7459051B2 (Magic Leap) の**従属請求項と他の独立請求項が未読**です。請求項 1 は取得済みで、頭部装着装置の前提部は日本にも存在し、Harris 系定数 `k1` は日本には現れないことを確認しました。法的状態は Google Patents の Active 表示に依拠しており、一次確認をしていません。
- JP2024019662A (Magic Leap の分割とみられる公開公報) は、Google Patents 表示では **2025-02-25 に拒絶理由通知、2025-08-21 に拒絶査定**です。**拒絶査定不服審判が請求されたかは未確認**であり、査定が確定していれば権利は発生しません。請求項も未読です。
- **JP6493163B2 (オムロン、粗密探索) の請求項が未確認です。** 対応する US10366307B2 の claim は template matching と SIMD 命令を必須要件としますが、日本の請求項が同じ限定を持つとは限りません。
- **Magic Leap の corner detection には日本に 2 系統目の family (US11686941B2 系: JP2022532238A / JP7422785B2 / JP2024028494A / JP7616796B2) が存在します。** 検索結果一覧で番号と題名を読んだだけで、個別 page を取得していません。
- **NVIDIA の CUDA graph 系日本案件 (JP7799551B2 登録、JP2024514372A / JP2024514371A 公開) の請求項が未確認です。** 本 project の CUDA Graph 化との関係は、請求項を読まないと判りません。
- **日本において、ArUco 著者 5 名のカタカナ表記による網羅検索が未完です。** 正しい表記が「ガルリド フラド」であることは判りましたが、patentfield の未ログイン検索回数制限に達したため、この表記での検索を実行できていません。マリン ヒメネス、メディナ カルニセル、ロメロ ラミレスのカタカナ検索も未実施です。
- **日本の請求項ベースの検索 (米国の `ACLM/` 検索に相当するもの) が一切実施できていません。**
- **patent clearance の実施担当が未定です。** [知的財産・ライセンス方針](ip-and-licensing.md) の未確定事項に挙がっている項目のうち、対象国は確定しましたが**実施担当は依然として未定**です。
- 専門家による freedom-to-operate review を依頼するか否かが未決定です。本文書は判断材料の提示にとどめています。
- **係属中の出願は請求項が今後補正されうるため、現時点の読みは暫定です。** 対象は US20250292411A1 (NVIDIA、審査未着手のまま係属)、US20230316669A1 (Seabery、最終拒絶後に審査継続中)、JP2023539810A / JP2024507819A (Seabery、いずれも 3 回・2 回の拒絶理由通知を経て未処分)、JP2024019662A (Magic Leap、状態未確認)、JP2025056491A / JP2024114339A / JP2025538690A / JP2026021866A です。
- `docs/design/detector-pipeline.md` が未確定事項として残している「適応的二値化を integral image 方式へ変更するか」は、**US11132569B2 と継続の US11682099B2** の claim との関係を確認してから決める必要があります。**継続では前提部から `integral image` と `hardware accelerator` の語が外れ方法 claim も存在するため、親より射程の評価が難しくなっています。**
- 本 project が変換元とする `DICT_ARUCO_MIP_36h12` の設計者と、Seabery 出願の発明者 `Sergio Garrido Jurado` が同一人物かは**依然として未確認**です。第 2 次調査で 2 つの状況証拠が得られました。(a) JP2023539810A の公報原本の (72) 発明者欄に「ガルリド フラド，セルヒオ／スペイン１４００７コルドバ、カリェ／プラテロ・ペドロ・デ・バレス２４－１－３」とあり**住所がコルドバ (スペイン)** である。(b) US20230316669A1 の譲渡記録 (REEL/FRAME 063109/0600) の譲渡人に **GARRIDO JURADO, SERGIO** が明記されている。**いずれも同姓同名・同市在住という状況証拠にとどまり、同一人物であることの確認ではありません。**
- **Google Patents の到達可否が不安定です。** 第 2 次調査では米国公報 (`/patent/US.../en`) について HTTP 200 に回復した担当がいた一方、日本公報 (`/patent/JP.../ja`) は複数の担当が 503 のままでした。**同じ日のうちに結果が分かれており、安定した経路として計画に組み込めません。**

## 専門家へ依頼するとしたら

以下は依頼事項の**候補**であり、依頼するか否かの判断は行っていません。**販売予定国が日本と米国に確定したため、この 2 国に絞って整理します。** 優先度は本記録の未確認事項の重さに基づきます。

### 最優先 — 日本 (本調査で最も欠落が大きい)

1. **日本の請求項の入手 (残りの部分)。** 本 project に最も近い 2 件の**独立請求項 1 は本調査で取得しました** (上の「日本の請求項 1 の全文」)。残りを J-PlatPat、公報固定アドレスサービス (INPIT)、または有償 database で入手することを依頼します。
   - `JP7429542B2` (NVIDIA) の**従属請求項および他の独立請求項**。請求項 1 は取得済みで、色空間変換・候補コーナ点の検出・しきいピクセル距離という 3 つの構成要件を持ち、**本 project にはいずれも存在しません** (commit `912a17c` に対する grep で確認)。従属請求項がこの評価を変えるかは未確認です。
   - 同一出願の公開公報 `JP2020119550A` の請求項。公開時と登録時の差から**審査経過での補正**を追うために要ります。
   - `JP7459051B2` (Magic Leap) の**従属請求項および他の独立請求項**。請求項 1 は取得済みで、**頭部装着装置の前提部は日本にも存在し、米国 claim 1 の Harris 系定数 `k1` は日本には現れず、代わりに重複コーナの除去とその実行順序が入っている**ことを確認しました。**日本と米国で限定が異なります。**
   - `JP6493163B2` (オムロン、粗密探索) の登録請求項全文。**未読です。**
2. **日本の法的状態の確認。** 上記 3 件と、本記録に挙げた日本の登録公報 (JP7448281B2、JP7239301B2、JP7365986B2、JP7782311B2、JP4250391B2、JP7682444B2、JP3802737B2、JP7799551B2、JP7739222B2、JP4237046B2、JP6998198B2 ほか) について、**年金納付状況、存続/失効、権利者移転、無効審判・異議申立の有無**を確認することを依頼します。**本調査ではこれらを 1 件も確認していません。** とくに 2000-2006 年出願のもの (JP3802737B2、JP4250391B2、JP4237046B2、JP2005267362A、JP2007334431A) は存続期間満了の可能性がありますが、確認していません。
3. **日本の分類検索と請求項ベース検索。** FI / F-term / CPC による網羅検索と、請求項を対象とした検索を依頼します。**本調査は英訳要約の技術語による検索のみで、日本語の技術語をそのまま検索する手段も、請求項を検索する手段もありませんでした。** 検索軸として、ArUco3 の中核 (最小マーカー寸法からの縮小率決定、縮小画像単一化、四隅の段階的精密化)、正方形 2 値マーカーの検出 pipeline、GPU 並列の連結成分ラベリング・適応的二値化・Otsu 法を挙げます。
4. **日本の未確認 family の追跡。** Magic Leap の 2 系統目 (`JP2022532238A` / `JP7422785B2` / `JP2024028494A` / `JP7616796B2`)、NVIDIA の CUDA graph 系 (`JP7799551B2` / `JP2024514372A` / `JP2024514371A`)、および村田機械の走行車系 (`JP7396532B2` ほか) の書誌と請求項を確認することを依頼します。**いずれも検索結果一覧で番号と題名を読んだだけです。**
5. **日本の出願人軸の網羅検索。** DENSO WAVE、Canon (marker を「指標 = index」と表記するため語を変える必要があります)、Sony、Seiko Epson、Keyence、Omron、Toshiba、Hitachi、NEC、Ricoh、村田機械、新東工業について、正方形マーカー検出・GPU 画像処理に関する出願を網羅することを依頼します。**本調査は各社について上位 25 件しか確認できていません。**
6. **日本における ArUco 著者・UCO の再検索。** カタカナ表記「ガルリド フラド」ほか正しい表記での発明者検索を依頼します。**本調査では表記を誤っていたか、回数制限で実行できていません。** また FPO の日本コレクションの発明者索引に欠落があることを実測しているため、**発明者名の 0 件は不存在の証明になりません。**

### 次点 — 米国 (claim 解釈が必要なもの)

7. **NVIDIA family の claim chart 作成。** `US11113819B2` / `US12322114B2` / 係属中の `US20250292411A1` の各独立 claim と、本 project の S3-S6 を要素ごとに対照することを依頼します。**第 2 次調査で claim ごとの要件の所在が公報原本により確定しています** — US12322114B2 は独立 claim 3 本すべてが「候補 corner 点と境界の閾値画素距離比較」を必須要件とし角度計算と色空間変換は従属 claim のみ、US20250292411A1 は独立 claim 3 本すべてが角度計算を必須要件とし claim 10 のみ `points` と記載されて広い。**争点は `candidate corner points` の解釈です。** 本 project は Harris 等の名前の付いた corner 検出器を使いませんが、極点探索で候補四角形の四隅にあたる画素を検出しています。「corner 検出器を使っていない」ことを差異として主張できるかは、まさに専門家の判断を要します。**確実に差異と言えるのは色空間変換の不在です** (公開 API が 8 bit grayscale に限定)。角度計算が無いこと (`atan2` / `acos` / `asin` の grep 0 件、commit `912a17c`) は US20250292411A1 の独立 claim に対しては意味を持ちます。従属 claim および継続出願の claim 変遷の監視方針も含めて依頼します。
8. **Magic Leap の corner detection family の claim 解釈。** `US11430212B2` / `US11605223B2` の独立 claim はいずれも前提部が頭部装着装置であり、本 project は library です。**本 library を頭部装着型 AR 機器へ組み込んだ製品形態では前提部の評価が変わりうる**ため、想定する製品形態を specify したうえでの評価を依頼します。**日本の JP7459051B2 の請求項 1 にも同じ頭部装着装置の前提部があることは確認済みです。** したがって同じ論点が日本でも成り立ちます。本 project を頭部装着型 AR 機器へ組み込む製品形態を想定するかどうかが、日米いずれでも分岐点になります。
9. **Intel `US9558560B2` の claim 5 / 9 / 13 に対する評価。** **第 2 次調査で display 要件を持たない独立 claim が実在することが確定しました。** したがって「本 project は library であり display を持たない」という差異はこれらの claim には働きません。残る差異は**均質条件による fast / generic 2 経路切替**のみで、これは 4 本すべてに残ります。本 project の block union-find + thread 単位の局所 label 表 + global 表への atomic 更新との対照を依頼します。**なおこの family には日本の対応出願が見つかっていません** (米国とドイツのみ)。
10. **Millennium Three `US11887312B2` の means-plus-function 解釈。** **第 2 次調査で独立 claim 1 / 7 が frame 間追跡を要件としないことが確定しました。** 代わりに AR system の前提部、display screen 付き capture means、marker patterns そのもの、remote / central location との送受信手段を必須要件とし、検出手順は `recognition means for recognizing said fiducial markers` という機能的記載にとどまります。**means-plus-function の範囲が明細書の対応構造に照らしてどう解釈されるか**の評価を依頼します。あわせて、同社が保有する `US7769236B2` / `US10504231B2` / `US11100649B2` / `US11887312B2` を**出願人軸で網羅する**ことを依頼します。**日本にはこの権利者の権利が見つかっていません。**
11. **integral image 方式への設計変更を検討する場合の再確認。** `US11132569B2` と継続の `US11682099B2` (TI) の claim を再確認することを依頼します。**継続では前提部から `integral image` と `hardware accelerator` の語が外れ方法 claim も存在するため、親より射程の評価が難しくなっています。** 日本には対応 family が見つかっていません。

### 調査手法の補完 (本調査の手段では実施できなかったもの)

12. **米国の一次 database での法的状態の確認。** USPTO Patent Center の維持年金納付記録、Assignment Search の REEL/FRAME 原本、包袋 (file wrapper)、PTAB の一次記録で存続性と審査経過を確認することを依頼します。**本記録の状態欄はすべて Google Patents の Legal Events 表と Unified Patents API という二次情報に依拠しています。** とくに `US7725518B1` (2010 年登録) と `US9042652B2` (2015 年登録) は年金不納による失効の可能性を排除できていません。
13. **CPC 分類ベースの網羅検索 (米国)。** `G06V 10/24`、`G06K 7/14`、`G06T 7/73` 等の分類による検索を依頼します。本調査はすべて技術語ベースであり、語のゆれによる漏れが構造的に残っています。
14. **譲受人軸での網羅検索 (米国)。** ARToolKit の権利を取得した経緯がある DAQRI、PTC / Vuforia、Qualcomm、Cognex、Zebra、Seabery、Millennium Three Technologies を含む出願人名での検索を依頼します。
15. **被引用関係の追跡。** `US11113819B2` および ArUco3 論文を引用する後発出願の追跡を依頼します。論文は他社特許の引用文献として広く流通していることが確認できており、同領域の後発出願を効率的に洗い出せる見込みがあります。

### 論点の切り分け

16. **Dictionary の「生成」と「検出」の切り分けに関する評価。** 本 project は Dictionary を生成しておらず、OpenCV 収録済み Dictionary の**変換物**を保持しているだけです (`extendDictionary()` の呼び出し 0 件、MILP solver 不在)。この事実が生成方法を対象とする方法 claim に対して持つ意味と、**marker という「物」の claim や Dictionary を含む装置・system claim に対して同じ論法が及ぶか**について評価を依頼します。**日本には marker という「物」の請求項を持つ登録公報が実在します** (JP7239301B2 日立)。あわせて、**marker を印刷・配布する行為**と、**software が code table を保持する行為**とで実施の性質が異なるかについての整理も依頼します。
17. **freedom-to-operate review の要否と範囲の判断 (手順 4)。** 本記録を入力として、日本と米国について full FTO を行うか、特定 family (NVIDIA / Magic Leap / Intel / Millennium Three) に絞った限定調査にとどめるかの判断を依頼します。

### 対象外とした国について

18. **欧州・スペイン・中国は販売予定国に含まれないため、本記録では対象外としました。これは判断による範囲外化であり、「調べたが権利が無かった」ではありません。** 第 1 次調査で番号だけが判明している `ES2894549A1` / `ES2894549B2` (Seabery、Google Patents 表示は Active、満了予測 2040-08-10)、`EP4195182A4`、`CN111435438B`、`CN116075875A`、`DE102015001699B4` (Intel、connected component labeling の family)、`DE102020100684B4` (NVIDIA family) は、いずれも claim も法的状態も未確認のままです。**販売予定国を欧州または中国へ広げる決定がなされた場合、これらは未調査の状態から再開する必要があります。** とくに ArUco の著者はスペインの大学に所属しており、ES 国内出願のみで PCT に乗らなかった案件は本記録では一切捕捉していません。