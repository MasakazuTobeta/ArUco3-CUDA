# AGENTS Guide

このリポジトリで作業するエージェントは、最初に `CONTRIBUTING.md` を確認してください。共通ルールの正本は `CONTRIBUTING.md` です。

## 優先順位

規約が衝突する場合は、次の順に適用します。

1. ユーザーの明示的な指示
2. `CONTRIBUTING.md`
3. repo root または対象ディレクトリの `AGENTS.md`
4. 対象領域の `README.md` と `docs/**`

## First Read Checklist

- `CONTRIBUTING.md` を読む。
- 文書を変更する場合は `docs/AGENTS.md` と `docs/terminology.md` を読む。
- 変更対象に対応する設計文書、ADR、評価計画を読む。
- 非自明な変更では、目的、依存関係、リスク、検証方法、更新対象文書を整理する。

## Working Principles

- 現在の実装と目標を混在させない。
- CUDA が常に高速という前提を置かず、CPU、CUDA、ハイブリッドを測定結果で比較する。
- カーネル単体時間と入力準備・同期・結果取得を含む時間を分けて記録する。
- DGX Spark と Jetson Orin で同じアルゴリズムを使用し、機種固有最適化は明示的なオプションとして分離する。
- しきい値や上限値を source の固定値だけにせず、設定から上書きできる構造にする。
- 外部実装のコードを無断でコピーしない。論文、公開仕様、互換ライセンスの実装を参照した場合は出典と影響を記録する。
- 重要な設計判断は ADR または関連文書へ残す。

## Documentation Rules

- 仕様、設計、評価文書には、少なくとも `目的`、`対象範囲`、`現状`、`目標`、`未確定事項` を設ける。
- 振る舞い、設定、公開 API、評価条件が変わる場合は、関連する文書も同じ変更で更新する。
- C++ の同一 stem に属する `.cpp` / `.hpp` / `.h` / `.cc` / `.cxx` の合計が 300 行以上になる場合は、同じディレクトリへ sidecar `*.md` を追加する。
- 概要図には差分確認しやすい Mermaid を優先する。

## Completion Checklist

- 関連する CPU 基準実装との互換性を確認したか。
- 正常系、異常系、境界値のテストを追加したか。
- CUDA エラーと非同期実行の失敗を検出できるか。
- カーネル単体と end-to-end の測定を混同していないか。
- DGX Spark と Jetson Orin の少なくとも対象側で検証したか、未検証理由を記録したか。
- README、設計、評価条件、ADR に矛盾がないか。
- secret、生成物、大容量データセットを commit に含めていないか。

## Operational Know-How

作業中に再発しうる失敗や環境固有の注意事項が判明した場合は、この節へ原因と正しい手順を追記してください。

### GitHub App で空 repository を初期化する場合

- 完全に空の repository では、Git Data API の blob 作成が `409 Git Repository is empty` で失敗する。
- 最初に Contents API で `README.md` 等の 1 file を作成して default branch を初期化する。
- 初期化後は、blob、tree、commit、ref update の順で複数 file を 1 commit にまとめられる。
