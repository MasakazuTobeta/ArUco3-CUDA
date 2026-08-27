# dictgen

## 目的

OpenCV 4.x の定義済み Dictionary から、CUDA 側で使用する packed codeword を生成し、C++ source として出力します。生成物が OpenCV と整合していることを継続的に確認する `--check` も提供します。

## 対象範囲

定義済み Dictionary の読み出し、bit 配列の packed 表現への変換、回転規則の検証、C++ source の生成、生成物と OpenCV の整合確認を対象とします。custom Dictionary の生成、MILP による codeword 探索は対象外です。

## 現状

- 対応する定義済み Dictionary は 18 種です。
- 生成対象は 1 回の実行につき 1 Dictionary です。
- 生成物は `src/dictionary/generated/` へ置き、repository へ commit します。

## 実装上の判断

### 生成物を repository へ commit する

build 時に OpenCV から生成すると、`core` の build に OpenCV が必要になります。[アーキテクチャ](../../docs/architecture.md) は core の OpenCV 依存を最小化する方針であり、これと矛盾します。生成物を commit し、OpenCV との一致は `--check` と `test/reference/test_dictionary_conformance.cpp` で継続的に検証します。

### bytesList を自前で展開しない

OpenCV の `bytesList` は `CV_8UC4` ですが、memory 配置は channel の interleave ではなく回転ごとの連続 block です。公開 header の記述どおり `bytesList.ptr(i)[k*nbytes + j]` が i 番目 marker の k 回転目の j byte 目です。この配置を誤って interleave として読むと、回転 0 以外が壊れた値になります。誤読を避けるため、byte 列を自前で展開せず `Dictionary::getBitsFromByteList()` に `rotationId` を渡して取得します。

### 回転規則を生成時に検証する

CUDA 側は 4 回転を事前展開した table を使い、照合結果の rotation を OpenCV と比較します。table の回転順序が OpenCV の定義と一致していなければ、この比較が成立しません。生成時に `rotate_marker_code()` の連鎖が table の rotation 1 から 3 を再現すること、および 4 回まわすと元へ戻ることを確認し、一致しない場合は生成を中止します。

### 生成物を整形対象から外す

`clang-format` を適用すると生成器の出力と byte 単位で一致しなくなり、`--check` による再生成の検証が成立しません。`cmake/Aruco3CudaOptions.cmake` が `/generated/` を整形対象から除外します。

### 名前空間スコープの const へ extern を付ける

C++ では名前空間スコープの `const` は既定で internal linkage になります。`registry.cpp` から参照できるよう、生成物の定義へ `extern` を明示します。

### 識別子を kPascalCase で生成する

`CONTRIBUTING.md` は定数を `kPascalCase` と定めます。Dictionary 名の `_` 区切りを語の境界として扱い、`DICT_ARUCO_MIP_36h12` から `kDictArucoMip36h12Codes` を生成します。

## 目標

- 複数 Dictionary を 1 回の実行で生成できるようにする。
- 生成物の hash を [Code Provenance 記録](../../docs/code-provenance.md) へ自動で反映する。
- `registry.cpp` の宣言も生成し、Dictionary 追加時の手作業を無くす。

## 関連

- [Dictionary 方針](../../docs/dictionaries.md)
- [検出パイプライン設計](../../docs/design/detector-pipeline.md)
- [Code Provenance 記録](../../docs/code-provenance.md)
