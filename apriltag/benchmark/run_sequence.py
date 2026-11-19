# SPDX-License-Identifier: Apache-2.0
"""Run one equal-layout AprilTag sequence. Store generated evidence outside Git."""

import argparse
import hashlib
import json
import math
import subprocess
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
    manifest = json.loads(args.manifest.read_text())
    if manifest["dictionary"] != "DICT_APRILTAG_36h11":
        raise ValueError("paired sequence requires the common 36h11 family")
    scenes = manifest["scenes"]
    if len({s["name"] for s in scenes}) != len(scenes) or not scenes:
        raise ValueError("manifest must have distinct nonempty scenes")
    files = [args.manifest.parent / s["path"] for s in scenes]
    for path, scene in zip(files, scenes):
        if hashlib.sha256(path.read_bytes()).hexdigest() != scene["sha256"]:
            raise ValueError(f"input digest mismatch for {path}")
    route_parameter = args.tile if args.backend == "nvidia" else args.decimate
    command = [str(args.executable.resolve()), args.scope, str(args.warmup),
               str(args.samples), str(route_parameter), str(args.max_tags), *map(str, files)]
    completed = subprocess.run(command, check=True, stdout=subprocess.PIPE, text=True)
    rows = [json.loads(line) for line in completed.stdout.splitlines()]
    if len(rows) != args.samples * len(scenes):
        raise ValueError("incomplete result sequence")
    args.out.mkdir(parents=True, exist_ok=False)
    qualities = []
    timings = []
    for ordinal, row in enumerate(rows):
        cycle, frame = divmod(ordinal, len(scenes))
        if (row["cycle"], row["frame"], row["scope"]) != (cycle, frame, args.scope):
            raise ValueError("response sequence mismatch")
        if not math.isfinite(row["ms"]) or row["ms"] <= 0:
            raise ValueError("invalid duration")
        name = scenes[frame]["name"]
        timings.append({"cycle": cycle, "scene": name, "ms": row["ms"]})
        observed = []
        for tag in row["tags"]:
            corners = tag["corners"]
            if (len(corners) != 4 or any(len(c) != 2 for c in corners)
                    or not all(math.isfinite(x) for point in corners for x in point)):
                raise ValueError("invalid corners")
            # cuAprilTags starts two corners later in the canonical tag orientation.
            if args.backend == "nvidia":
                corners = corners[2:] + corners[:2]
            observed.append({"id": tag["id"], "corners": [x for c in corners for x in c]})
        observations = {"scene": name, "detections": sorted(observed,
                                                           key=lambda tag: (tag["id"], tag["corners"]))}
        if cycle == 0:
            qualities.append(observations)
        elif observations != qualities[frame]:
            # Scheduling may change detection ordering. Any differences that survive
            # sorting must be retained for review, not silently folded into one result.
            (args.out / "changed-observation.json").write_text(json.dumps(
                {"cycle": cycle, "first": qualities[frame], "changed": observations}))
            raise ValueError("changing detection on exact replay")
    for name, values in [("observations", qualities), ("timings", timings)]:
        (args.out / f"{name}.jsonl").write_text("".join(json.dumps(v) + "\n" for v in values))
    (args.out / "conditions.json").write_text(json.dumps({
        "command": command, "backend": args.backend,
        "binary_sha256": hashlib.sha256(args.executable.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    }, indent=2))


if __name__ == "__main__":
    main()
