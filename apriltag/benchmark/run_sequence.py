# SPDX-License-Identifier: Apache-2.0
"""Run one equal-layout AprilTag sequence. Store generated evidence outside Git."""

import argparse
import hashlib
import json
import math
import subprocess
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--backend", choices=["nvidia", "project"], required=True)
    parser.add_argument("--scope", choices=["host", "device"], default="device")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--max-tags", type=int, default=1024)
    parser.add_argument("--tile", type=int, default=4, help="NVIDIA detector's tile size")
    parser.add_argument("--decimate", type=int, choices=range(5),
                        help="project segmentation spacing, 0 selects automatic sampling")
    args = parser.parse_args()
    if args.backend == "project" and args.decimate is None:
        parser.error("the project route requires --decimate")
    if not 0 <= args.warmup <= 1000000 or not 1 <= args.samples <= 1000000:
        parser.error("expected 0–1000000 warmup cycles and 1–1000000 timed cycles")
    if not 1 <= args.max_tags <= 65535:
        parser.error("--max-tags must be between 1 and 65535")
    if args.out.exists():
        parser.error("output directory already exists")
    manifest_bytes = args.manifest.read_bytes()
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    if manifest["dictionary"] != "DICT_APRILTAG_36h11":
        raise ValueError("paired sequence requires the common 36h11 family")
    scenes = manifest["scenes"]
    if (not scenes or any(not isinstance(s["name"], str) or not s["name"] for s in scenes)
            or len({s["name"] for s in scenes}) != len(scenes)):
        raise ValueError("manifest must have distinct nonempty scenes")
    files = [args.manifest.parent / s["path"] for s in scenes]
    route_parameter = args.tile if args.backend == "nvidia" else args.decimate
    command = [str(args.executable.resolve()), args.scope, str(args.warmup),
               str(args.samples), str(route_parameter), str(args.max_tags)]
    binary_digest = hashlib.sha256(args.executable.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="apriltag-input-") as snapshot:
        snapshots = []
        for i, (path, scene) in enumerate(zip(files, scenes)):
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != scene["sha256"]:
                raise ValueError(f"input digest mismatch for {path}")
            # The native loader reopens its inputs. Give it exactly the bytes we checked.
            copy = Path(snapshot) / f"{i}.pgm"
            copy.write_bytes(data)
            snapshots.append(str(copy))
        completed = subprocess.run([*command, *snapshots], check=True, stdout=subprocess.PIPE, text=True)
    if hashlib.sha256(args.executable.read_bytes()).hexdigest() != binary_digest:
        raise ValueError("benchmark executable changed during the run")
    rows = [json.loads(line) for line in completed.stdout.splitlines()]
    if len(rows) != args.samples * len(scenes):
        raise ValueError("incomplete result sequence")
    qualities = []
    timings = []
    for ordinal, row in enumerate(rows):
        cycle, frame = divmod(ordinal, len(scenes))
        if row.get("backend") != args.backend:
            raise ValueError("benchmark backend mismatch: select or rebuild the matching executable")
        if type(row["cycle"]) is not int or type(row["frame"]) is not int:
            raise ValueError("invalid response index")
        if (row["cycle"], row["frame"], row["scope"]) != (cycle, frame, args.scope):
            raise ValueError("response sequence mismatch")
        if type(row["ms"]) not in (int, float) or not math.isfinite(row["ms"]) or row["ms"] <= 0:
            raise ValueError("invalid duration")
        name = scenes[frame]["name"]
        timings.append({"cycle": cycle, "scene": name, "ms": row["ms"]})
        observed = []
        if not isinstance(row["tags"], list) or len(row["tags"]) >= args.max_tags:
            raise ValueError("invalid or potentially truncated tag list")
        for tag in row["tags"]:
            if type(tag["id"]) is not int or not 0 <= tag["id"] < 587:
                raise ValueError("tag ID outside the 36h11 dictionary")
            corners = tag["corners"]
            if (len(corners) != 4 or any(len(c) != 2 for c in corners)
                    or not all(type(x) in (int, float) and math.isfinite(x) for point in corners for x in point)):
                raise ValueError("invalid corners")
            # cuAprilTags starts two corners later in the canonical tag orientation.
            if args.backend == "nvidia":
                corners = corners[2:] + corners[:2]
            observed.append({"id": tag["id"], "corners": [x for c in corners for x in c]})
        observations = {"scene": name, "manifest_sha256": manifest_digest,
                        "detections": sorted(observed, key=lambda tag: (tag["id"], tag["corners"]))}
        if cycle == 0:
            qualities.append(observations)
        elif observations != qualities[frame]:
            # Scheduling may change detection ordering. Any differences that survive
            # sorting must be retained for review, not silently folded into one result.
            args.out.mkdir(parents=True, exist_ok=False)
            (args.out / "changed-observation.json").write_text(json.dumps(
                {"cycle": cycle, "first": qualities[frame], "changed": observations}))
            raise ValueError("changing detection on exact replay")
    args.out.mkdir(parents=True, exist_ok=False)
    for name, values in [("observations", qualities), ("timings", timings)]:
        (args.out / f"{name}.jsonl").write_text("".join(json.dumps(v) + "\n" for v in values))
    (args.out / "conditions.json").write_text(json.dumps({
        "command": [*command, *map(str, files)], "backend": args.backend,
        "binary_sha256": binary_digest,
        "manifest_sha256": manifest_digest
    }, indent=2))


if __name__ == "__main__":
    main()
