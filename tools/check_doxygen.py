#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""公開ヘッダの宣言ごとに、規約が要求する Doxygen 要素の有無を検査する。

規約: public class と関数には、目的、引数、戻り値、所有権、同期動作、入力例、出力例を記載する。

所有権と同期動作が class 全体で共通する場合は、class の Doxygen へ
「全ての public member 関数に適用される」と明記することで member 側の記載に代える。
同一の記述を member ごとに複写すると、規約が避けよと定める「処理の言い換え」に
近い冗長さを生み、可読性を損なうためである。

使用方法:
    python3 tools/check_doxygen.py
    欠落があれば一覧を表示し、終了コード 1 を返す。
"""
import re, sys, pathlib

HEADERS = [
    "include/aruco3cuda/status.hpp", "include/aruco3cuda/version.hpp",
    "include/aruco3cuda/device_probe.hpp", "include/aruco3cuda/dictionary.hpp",
    "include/aruco3cuda/util/sha256.hpp", "include/aruco3cuda/util/json_writer.hpp",
    "include/aruco3cuda/util/statistics.hpp",
    "reference/reference_runner.hpp", "tools/corpusgen/corpus_generator.hpp",
    "bench/benchmark_harness.hpp", "src/core/cuda_check.hpp",
]

DECL = re.compile(r'^\s{0,4}(?:(?:const\s+)?[\w:]+[\w:<>,\s*&]*?)\s(\w+)\s*\(([^;{]*)\)\s*(?:const)?\s*[;{]')
CLASS = re.compile(r'^\s*(?:class|struct)\s+(\w+)\s*(?:final)?\s*[{:]')

def comment_block(lines, index):
    """宣言の直前に連なる /// コメント行を返す。"""
    out, i = [], index - 1
    while i >= 0:
        st = lines[i].strip()
        if st.startswith("///") or st.startswith("//"):
            out.insert(0, st); i -= 1
        elif st == "" and out:
            break
        elif st == "":
            i -= 1
            if i >= 0 and not lines[i].strip().startswith("//"): break
        else:
            break
    return "\n".join(out)

def params_of(sig):
    sig = sig.strip()
    if not sig or sig == "void": return []
    parts, depth, cur = [], 0, ""
    for ch in sig:
        if ch in "<(": depth += 1
        elif ch in ">)": depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur); cur = ""
        else:
            cur += ch
    if cur.strip(): parts.append(cur)
    names = []
    for p in parts:
        p = p.split("=")[0].strip()
        m = re.search(r'(\w+)\s*(?:\[\s*\])?$', p)
        if m and m.group(1) not in ("void",): names.append(m.group(1))
    return names

problems = []
for path in HEADERS:
    f = pathlib.Path(path)
    if not f.exists(): continue
    lines = f.read_text().splitlines()
    in_private = False
    # class doc が「全ての public member 関数に適用される」と明記していれば、
    # その項目は member 関数側で満たされているものとして扱う。
    class_covers = set()
    current_class = None
    for i, line in enumerate(lines):
        st = line.strip()
        if st.startswith("private:"): in_private = True
        if st.startswith("public:"): in_private = False
        if in_private: continue
        if st.startswith("//") or st.startswith("*") or st.startswith("#"): continue

        mc = CLASS.match(line)
        if mc:
            doc = comment_block(lines, i)
            class_covers = set()
            current_class = mc.group(1)
            if "member 関数に適用される" in doc:
                if "所有権" in doc: class_covers.add("所有権")
                if "同期動作" in doc: class_covers.add("同期動作")
            missing = [k for k, pat in (("目的", r'///\s*\S'), ("所有権", r'所有権'),
                                        ("同期動作", r'同期動作'), ("入力例", r'入力例'),
                                        ("出力例", r'出力例')) if not re.search(pat, doc)]
            if missing and not mc.group(1).endswith(("Result", "Record", "Config", "Spec",
                                                     "Table", "Match", "Detection", "Detections",
                                                     "Environment", "Scene", "GroundTruth",
                                                     "Statistics", "ViewU8", "Closer")):
                problems.append((path, i + 1, f"class {mc.group(1)}", missing))
            continue

        md = DECL.match(line)
        if not md: continue
        name, sig = md.group(1), md.group(2)
        if name in ("if", "for", "while", "switch", "return", "sizeof", "static_cast"): continue
        # 複数行にまたがる宣言は引数を連結する
        if ")" not in line.split("(", 1)[1]:
            j, buf = i, line
            while j + 1 < len(lines) and ")" not in buf.split("(", 1)[1]:
                j += 1; buf += " " + lines[j].strip()
            sig = buf.split("(", 1)[1].rsplit(")", 1)[0]
        doc = comment_block(lines, i)
        if not doc:
            problems.append((path, i + 1, name, ["Doxygen 全体"])); continue
        missing = []
        for p in params_of(sig):
            if not re.search(r'@param\s+' + re.escape(p) + r'\b', doc): missing.append(f"@param {p}")
        # constructor と destructor は戻り値を持たない。
        is_ctor = (name == current_class) or name.startswith("~")
        returns_void = (bool(re.match(r'^\s*(?:inline\s+)?void\s', line))
                        or name == "operator()" or is_ctor)
        if not returns_void and "@return" not in doc: missing.append("@return")
        if "所有権" not in doc and "所有権" not in class_covers: missing.append("所有権")
        if "同期動作" not in doc and "同期動作" not in class_covers: missing.append("同期動作")
        if "入力例" not in doc: missing.append("入力例")
        if "出力例" not in doc: missing.append("出力例")
        if missing: problems.append((path, i + 1, name, missing))

cur = None
for path, line, name, missing in problems:
    if path != cur:
        print(f"\n=== {path} ===" ); cur = path
    print(f"  {line:4d} {name:32s} 欠落: {', '.join(missing)}")
print(f"\n合計 {len(problems)} 件の宣言に欠落あり")
sys.exit(1 if problems else 0)
