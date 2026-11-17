#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Aggregate paired per-scene CPU results across independent processes."""
import argparse
import collections
import json
import math
import pathlib
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=pathlib.Path)
    parser.add_argument("--candidate", default="optimized")
    args = parser.parse_args()
    environment = json.loads((args.directory / "environment.json").read_text())
    files = sorted(args.directory.glob("run-*.jsonl"))
    if len(files) != environment["independent_processes"]:
        raise ValueError("The independent measurement processes are incomplete")
    grouped = collections.defaultdict(lambda: collections.defaultdict(list))
    for path in files:
        seen = set()
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["type"] != "measurement":
                continue
            key = (row["scene"], row["variant"])
            if key in seen:
                raise ValueError(f"Duplicate measurement {key} in {path}")
            seen.add(key)
            grouped[row["scene"]][row["variant"]].append(row)
    scenes = []
    for scene, variants in sorted(grouped.items()):
        reference = variants["opencv"]
        candidate = variants[args.candidate]
        if len(reference) != len(files) or len(candidate) != len(files):
            raise ValueError(f"Missing paired measurement: {scene}")
        if len({row["sha256"] for row in reference + candidate}) != 1:
            raise ValueError(f"Inputs differ between measurements: {scene}")
        baseline_ms = statistics.median(row["p50_ms"] for row in reference)
        candidate_ms = statistics.median(row["p50_ms"] for row in candidate)
        scenes.append({
            "scene": scene, "sha256": reference[0]["sha256"],
            "reference_detections": reference[0]["detections"],
            "candidate_detections": candidate[0]["detections"],
            "opencv_p50_ms": baseline_ms, "candidate_p50_ms": candidate_ms,
            "opencv_p95_ms": statistics.median(row["p95_ms"] for row in reference),
            "candidate_p95_ms": statistics.median(row["p95_ms"] for row in candidate),
            "opencv_p99_ms": statistics.median(row["p99_ms"] for row in reference),
            "candidate_p99_ms": statistics.median(row["p99_ms"] for row in candidate),
            "speedup": baseline_ms / candidate_ms,
            "compatibility_discrepancies": sorted({row.get("compatibility_discrepancy", "")
                                                    for row in candidate} - {""}),
            "opencv_run_spread_pct": 100 * (max(row["p50_ms"] for row in reference) -
                                             min(row["p50_ms"] for row in reference)) / baseline_ms,
            "candidate_run_spread_pct": 100 * (max(row["p50_ms"] for row in candidate) -
                                                min(row["p50_ms"] for row in candidate)) / candidate_ms,
        })
        if "previous" in variants:
            previous = variants["previous"]
            if len(previous) != len(files) or any(
                    row["sha256"] != reference[0]["sha256"] for row in previous):
                raise ValueError(f"Missing or different previous input: {scene}")
            previous_ms = statistics.median(row["p50_ms"] for row in previous)
            scenes[-1].update({
                "previous_p50_ms": previous_ms,
                "previous_p95_ms": statistics.median(row["p95_ms"] for row in previous),
                "previous_p99_ms": statistics.median(row["p99_ms"] for row in previous),
                "speedup_over_previous": previous_ms / candidate_ms,
            })
    def describe(rows):
        ratios = [row["speedup"] for row in rows]
        return {"scenes": len(rows), "wins": sum(value > 1 for value in ratios),
                "min_speedup": min(ratios), "median_speedup": statistics.median(ratios),
                "geometric_mean_speedup": math.exp(statistics.mean(map(math.log, ratios))),
                "max_speedup": max(ratios)} if rows else {}
    summary = {"candidate": args.candidate, "independent_processes": len(files),
               "aggregation": "Median across independent process percentiles, per scene; not pooled percentiles",
               "all_scenes": describe(scenes),
               "scenes_with_detections": describe([row for row in scenes if row["reference_detections"] > 0]),
               "scenes_without_detections": describe([row for row in scenes if row["reference_detections"] == 0]),
               "scenes": scenes}
    incremental = [dict(row, speedup=row["speedup_over_previous"])
                   for row in scenes if "speedup_over_previous" in row]
    if incremental:
        summary["over_previous_all_scenes"] = describe(incremental)
        summary["over_previous_scenes_with_detections"] = describe(
            [row for row in incremental if row["reference_detections"] > 0])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
