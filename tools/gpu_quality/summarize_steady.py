#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarize completed steady_benchmark runs without pooling process tails."""
import argparse
import csv
import json
from pathlib import Path
from statistics import median

from steady_benchmark import validate_percentiles


def validate_command_rows(command, rows, environment):
    """Recheck the frozen plan against retained logs before aggregating them."""
    cpu = command["variant"].startswith("cpu")
    names = command["names"]
    if not names or len(set(names)) != len(names):
        raise ValueError("empty or duplicate input names in command plan")
    count = len(names) * (2 if cpu else 1)
    if len(rows) != count + 1 or rows[0].get("type") != "environment":
        raise ValueError("incomplete measurement rows: " + command["key"])
    run_environment = rows[0]
    if cpu and (run_environment["opencv_threads"] != int(command["variant"][-1]) or
                run_environment["aruco3"] != bool(command["mode"]) or
                run_environment["iterations"] != environment["iterations"] or
                run_environment["warmup"] != environment["warmup"] or
                not run_environment["save_samples"]):
        raise ValueError("CPU measurement conditions mismatch")
    resident = command["variant"].endswith("resident")
    route = "CUDA-Resident" if resident else "CUDA-E2E"
    expected = {(name, variant) for name in names
                for variant in (["opencv", "optimized"] if cpu else [route])}
    hashes = {item["name"]: item["sha256"]
              for item in environment["input_manifest"]["inputs"]}
    observed = set()
    for row in rows[1:]:
        if row.get("type") != "measurement":
            raise ValueError("unexpected measurement record type")
        name = row["scene"] if cpu else Path(row["image_path"]).name
        identity = (name, row["variant"] if cpu else row["route"])
        if identity not in expected or identity in observed:
            raise ValueError("unexpected or duplicate measurement identity")
        observed.add(identity)
        if row["sha256" if cpu else "image_sha256"] != hashes[name]:
            raise ValueError("measured input hash mismatch")
        validate_percentiles(row, cpu, environment["iterations"])
        if cpu:
            if row["compatibility_discrepancy"] or row["max_corner_error_px"] != 0:
                raise ValueError("CPU output mismatch")
        else:
            conditions = row["conditions"]
            if (conditions["max_candidates"] != command["candidate_budget"] or
                    conditions["max_markers"] != 1024 or
                    conditions["use_aruco3_detection"] != bool(command["mode"]) or
                    conditions["latency_iterations"] != environment["iterations"] or
                    conditions["warmup_iterations"] != environment["warmup"]):
                raise ValueError("GPU measurement conditions mismatch")
            if row["memory_mode"] != ("M-Device" if resident else "M-Pageable"):
                raise ValueError("GPU memory scope mismatch")
            if command["variant"].startswith("new-"):
                if any(conditions.get(key) != value for key, value in
                       environment["expected_new_metadata"].items()):
                    raise ValueError("compiled GPU identity mismatch")
            elif conditions.get("baseline_adapter") != "badb673 benchmark budgets only":
                raise ValueError("baseline adapter identity mismatch")
    if observed != expected:
        raise ValueError("incomplete measurement identities")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    complete = json.loads((args.run / "complete.json").read_text())
    environment = json.loads((args.run / "environment.json").read_text())
    commands = json.loads((args.run / "commands.json").read_text())
    if (not complete["complete"] or not complete["binaries_and_inputs_unchanged"] or
            len(commands) != complete["commands"] or not commands or
            len({case["key"] for case in commands}) != len(commands)):
        raise ValueError("incomplete benchmark")
    families = {item["name"]: item["family"]
                for item in environment["input_manifest"]["inputs"]}
    groups = {}
    for command in commands:
        cpu = command["variant"].startswith("cpu")
        rows = [json.loads(line) for line in
                (args.run / (command["key"] + ".jsonl")).read_text().splitlines()]
        validate_command_rows(command, rows, environment)
        for row in rows[1:]:
            name = row["scene"] if cpu else Path(row["image_path"]).name
            route = command["variant"] + "-" + row["variant"] if cpu else command["variant"]
            key = (command["mode"], name, route)
            groups.setdefault(key, []).append(row)
    summaries = []
    for (mode, name, route), rows in sorted(groups.items()):
        if len(rows) != environment["processes"]:
            raise ValueError("unexpected process count")
        cpu = route.startswith("cpu")
        stats = rows if cpu else [row["end_to_end"] for row in rows]
        counts = [row["detections" if cpu else "detection_count"] for row in rows]
        if len(set(counts)) != 1:
            raise ValueError("detection count changed between processes")
        item = {"aruco3": mode, "image": name, "family": families[name],
                "route": route, "detections": counts[0], "processes": len(rows)}
        for percentile in ("p50_ms", "p95_ms", "p99_ms"):
            values = [row[percentile] for row in stats]
            item[percentile] = median(values)
            item[percentile + "_process_min"] = min(values)
            item[percentile + "_process_max"] = max(values)
        summaries.append(item)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"wrote {len(summaries)} per-image summaries to {args.output}")


if __name__ == "__main__":
    main()
