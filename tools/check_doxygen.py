#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check every declaration in the public headers for the Doxygen elements the
convention requires.

Convention: a public class or function documents its purpose, parameters, return
value, ownership, synchronization behavior, an example input, and an example output.

When ownership and synchronization are the same across the whole class, stating
"applies to all public member functions" in the class Doxygen stands in for
repeating them on each member. Copying identical text onto every member produces
the same kind of redundancy as the "restating the code" the convention tells us to
avoid, and it hurts readability.

Usage:
    python3 tools/check_doxygen.py
    Prints a list of what is missing and returns exit code 1 if anything is.
"""
import re, sys, pathlib

# The targets are discovered automatically. A hand-maintained list would leave a
# newly added header out of the check, so missing documentation would still pass.
HEADER_ROOTS = ["include", "reference", "tools", "bench", "src", "hybrid", "examples"]


def discover_headers():
    found = []
    for root in HEADER_ROOTS:
        # .h as well as .hpp: the C ABI under include/aruco3cuda/c is public API
        # and is held to the same documentation rule as the C++ headers.
        for pattern in ("*.hpp", "*.h"):
            for path in sorted(pathlib.Path(root).rglob(pattern)):
                if "generated" in path.parts or "build" in path.parts:
                    continue
                found.append(str(path))
    return sorted(found)


HEADERS = discover_headers()

DECL = re.compile(r'^\s{0,4}(?:(?:const\s+)?[\w:]+[\w:<>,\s*&]*?)\s(\w+)\s*\(([^;{]*)\)\s*(?:const)?\s*[;{]')
CLASS = re.compile(r'^\s*(?:class|struct)\s+(\w+)\s*(?:final)?\s*[{:]')

def comment_block(lines, index):
    """Return the run of /// comment lines immediately above the declaration."""
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
    # If the class doc states "applies to all public member functions", treat those
    # items as already satisfied for each member function.
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
            if "applies to all public member functions" in doc:
                if "Ownership" in doc: class_covers.add("Ownership")
                if "Synchronization" in doc: class_covers.add("Synchronization")
            missing = [k for k, pat in (("purpose", r'///\s*\S'), ("Ownership", r'Ownership'),
                                        ("Synchronization", r'Synchronization'),
                                        ("Example input", r'Example input'),
                                        ("Example output", r'Example output')) if not re.search(pat, doc)]
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
        # Join the parameters of a declaration that spans several lines.
        if ")" not in line.split("(", 1)[1]:
            j, buf = i, line
            while j + 1 < len(lines) and ")" not in buf.split("(", 1)[1]:
                j += 1; buf += " " + lines[j].strip()
            sig = buf.split("(", 1)[1].rsplit(")", 1)[0]
        doc = comment_block(lines, i)
        if not doc:
            problems.append((path, i + 1, name, ["the whole Doxygen block"])); continue
        missing = []
        for p in params_of(sig):
            if not re.search(r'@param\s+' + re.escape(p) + r'\b', doc): missing.append(f"@param {p}")
        # Constructors and destructors have no return value.
        is_ctor = (name == current_class) or name.startswith("~")
        # The optional leading macro is the export attribute a C header puts in
        # front of the return type. Without allowing it, every void function in
        # the C ABI is asked for an @return it does not have.
        returns_void = (bool(re.match(r'^\s*(?:[A-Z][A-Z0-9_]*\s+)?(?:inline\s+)?void\s', line))
                        or name == "operator()" or is_ctor)
        if not returns_void and "@return" not in doc: missing.append("@return")
        if "Ownership" not in doc and "Ownership" not in class_covers: missing.append("Ownership")
        if "Synchronization" not in doc and "Synchronization" not in class_covers: missing.append("Synchronization")
        if "Example input" not in doc: missing.append("Example input")
        if "Example output" not in doc: missing.append("Example output")
        if missing: problems.append((path, i + 1, name, missing))

print(f"checking {len(HEADERS)} headers")
cur = None
for path, line, name, missing in problems:
    if path != cur:
        print(f"\n=== {path} ===" ); cur = path
    print(f"  {line:4d} {name:32s} missing: {', '.join(missing)}")
print(f"\n{len(problems)} declarations have missing elements in total")
sys.exit(1 if problems else 0)
