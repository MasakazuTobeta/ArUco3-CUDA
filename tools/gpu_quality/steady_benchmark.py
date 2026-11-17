#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run paired, startup-excluded CPU/GPU measurements on a fixed input manifest."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def validate_percentiles(row, cpu, iterations):
    samples = row["latency_samples_ms" if cpu else "end_to_end_samples_ms"]
    if len(samples) != iterations or any(not math.isfinite(x) or x < 0 for x in samples):
        raise ValueError("missing, nonfinite, or negative latency samples")
    ordered = sorted(samples)
    statistics = row if cpu else row["end_to_end"]
    tolerance = 1e-8 if cpu else 0.00011
    for name, fraction in (("p50_ms", .50), ("p95_ms", .95), ("p99_ms", .99)):
        expected = ordered[math.ceil(len(ordered) * fraction) - 1]
        if abs(statistics[name] - expected) > tolerance:
            raise ValueError(f"{name} disagrees with retained samples")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("old-bench", "new-bench", "cpu-bench", "inputs", "output", "expected-new-metadata"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--cpu-list", required=True, help="Four allowed logical CPU IDs")
    parser.add_argument("--processes", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cpus = [int(x) for x in args.cpu_list.split(",")]
    if len(cpus) != 4 or len(set(cpus)) != 4 or min(cpus) < 0:
        raise ValueError("--cpu-list must contain four distinct nonnegative CPU IDs")
    if args.processes < 1 or args.iterations < 1 or args.warmup < 0:
        raise ValueError("invalid measurement counts")
    expected_new = json.loads(args.expected_new_metadata.read_text())
    required = {"gpu_resize_profile", "gpu_resize_backend", "gpu_carotene_x_fma",
                "gpu_strategy_identity"}
    if set(expected_new) != required or any(not isinstance(v, str) or not v
                                            for v in expected_new.values()):
        raise ValueError("expected metadata must contain exactly four nonempty profile/strategy strings")
    inputs = args.inputs.resolve()
    manifest = json.loads((inputs / "input-manifest.json").read_text())
    items = manifest["inputs"]
    if not items or len({item["name"] for item in items}) != len(items):
        raise ValueError("empty or duplicate input manifest")
    for item in items:
        name = item["name"]
        if Path(name).name != name or not name.endswith(".png"):
            raise ValueError("manifest names must be plain PNG basenames")
        if digest(inputs / name) != item["sha256"]:
            raise ValueError("input hash mismatch: " + name)
    binaries = {"old": args.old_bench.resolve(), "new": args.new_bench.resolve(),
                "cpu": args.cpu_bench.resolve()}
    binary_hashes = {name: digest(path) for name, path in binaries.items()}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    off_names = [item["name"] for item in items if item["family"] == "synthetic"]
    off_names += sorted(item["name"] for item in items
                        if item["family"] == "industrial photograph")[:3]
    off_names.sort()
    off_inputs = output / "aruco3-off-inputs"
    off_inputs.mkdir()
    for name in off_names:
        shutil.copyfile(inputs / name, off_inputs / name)
    write_json(output / "environment.json", {
        "started_unix": time.time(), "hostname": platform.node(), "platform": platform.platform(),
        "cpu_list": cpus, "binary_paths": {k: str(v) for k, v in binaries.items()},
        "binary_sha256": binary_hashes, "runner_sha256": digest(Path(__file__)),
        "input_manifest": manifest, "processes": args.processes,
        "warmup": args.warmup, "iterations": args.iterations,
        "scope": "wall latency; startup excluded; GPU-resident and host-input/result separate",
        "gpu_equal_budget_max_candidates": 16384, "gpu_default_control_max_candidates": 4096,
        "gpu_max_markers": 1024, "expected_new_metadata": expected_new,
        "baseline_note": "old equal-budget algorithm is not CPU-quality equivalent",
        "external_workloads": "unchanged; capture host power/load separately",
        "aruco3_off_names": off_names})
    commands = []
    all_names = sorted(item["name"] for item in items)
    for mode, directory, names in ((1, inputs, all_names), (0, off_inputs, off_names)):
        for run in range(1, args.processes + 1):
            variants = ["cpu1", "old-resident", "new-resident", "old-default-resident",
                        "old-e2e", "new-e2e", "cpu4"]
            if run % 2 == 0:
                variants.reverse()
            for variant in variants:
                if variant.startswith("cpu"):
                    threads = int(variant[-1])
                    affinity = str(cpus[0]) if threads == 1 else args.cpu_list
                    cmd = ["taskset", "-c", affinity, str(binaries["cpu"]),
                           "--input-dir", str(directory), "--variants", "opencv,optimized",
                           "--threads", str(threads), "--warmup", str(args.warmup),
                           "--iterations", str(args.iterations), "--aruco3", str(mode),
                           "--throughput-frames", "0", "--save-samples"]
                else:
                    version, route = variant.split("-")[0], variant.split("-")[-1]
                    budget = 4096 if variant == "old-default-resident" else 16384
                    cmd = [str(binaries[version]), "--route",
                           "CUDA-Resident" if route == "resident" else "CUDA-E2E",
                           "--cpu-list", str(cpus[0]), "--threads", "1", "--warmup", str(args.warmup),
                           "--latency-iterations", str(args.iterations), "--throughput-frames", "0",
                           "--use-aruco3", str(mode), "--max-candidates", str(budget),
                           "--max-markers", "1024", "--save-samples"]
                    if route == "e2e":
                        cmd += ["--memory-mode", "M-Pageable"]
                    for name in names:
                        cmd += ["--input", str(directory / name)]
                commands.append({"key": f"mode{mode}-{variant}-{run}", "command": cmd,
                                 "variant": variant, "mode": mode, "names": names,
                                 "candidate_budget": None if variant.startswith("cpu") else budget})
    write_json(output / "commands.json", commands)
    if args.dry_run:
        print(f"prepared {len(commands)} commands; no benchmark executed", flush=True)
        return
    expected_hashes = {item["name"]: item["sha256"] for item in items}
    for case in commands:
        key = case["key"]
        cpu = case["variant"].startswith("cpu")
        begin = time.monotonic()
        with (output / (key + ".jsonl")).open("x") as stdout, \
                (output / (key + ".stderr")).open("x") as stderr:
            result = subprocess.run(case["command"], stdout=stdout, stderr=stderr, check=False)
        if result.returncode:
            raise RuntimeError(f"{key} failed ({result.returncode}); partial logs retained")
        rows = [json.loads(line) for line in (output / (key + ".jsonl")).read_text().splitlines()]
        expected_count = len(case["names"]) * (2 if cpu else 1)
        if len(rows) != expected_count + 1 or rows[0]["type"] != "environment":
            raise ValueError(key + ": incomplete row count")
        environment = rows[0]
        if cpu and (environment["opencv_threads"] != int(case["variant"][-1]) or
                    environment["aruco3"] != bool(case["mode"]) or
                    environment["iterations"] != args.iterations or
                    environment["warmup"] != args.warmup or not environment["save_samples"]):
            raise ValueError(key + ": CPU measurement conditions mismatch")
        observed = set()
        for row in rows[1:]:
            name = row["scene"] if cpu else Path(row["image_path"]).name
            identity = (name, row["variant"] if cpu else row["route"])
            if identity in observed or name not in case["names"]:
                raise ValueError(key + ": duplicate/unexpected image")
            observed.add(identity)
            if row["sha256" if cpu else "image_sha256"] != expected_hashes[name]:
                raise ValueError(key + ": measured input changed")
            validate_percentiles(row, cpu, args.iterations)
            if cpu:
                if row["variant"] not in ("opencv", "optimized"):
                    raise ValueError(key + ": unexpected CPU variant")
                if row["compatibility_discrepancy"] or row["max_corner_error_px"] != 0:
                    raise ValueError(key + ": CPU output mismatch")
            else:
                conditions = row["conditions"]
                if (conditions["max_candidates"] != case["candidate_budget"] or
                        conditions["max_markers"] != 1024 or
                        conditions["use_aruco3_detection"] != bool(case["mode"]) or
                        conditions["latency_iterations"] != args.iterations or
                        conditions["warmup_iterations"] != args.warmup):
                    raise ValueError(key + ": GPU measurement conditions mismatch")
                resident = case["variant"].endswith("resident")
                if (row["route"] != ("CUDA-Resident" if resident else "CUDA-E2E") or
                        row["memory_mode"] != ("M-Device" if resident else "M-Pageable")):
                    raise ValueError(key + ": route or memory scope mismatch")
                if case["variant"].startswith("new-"):
                    if any(conditions.get(k) != v for k, v in expected_new.items()):
                        raise ValueError(key + ": compiled GPU identity mismatch")
                elif conditions.get("baseline_adapter") != "badb673 benchmark budgets only":
                    raise ValueError(key + ": missing baseline adapter identity")
        with (output / "completed.jsonl").open("a") as stream:
            stream.write(json.dumps({"key": key, "rows": expected_count,
                                     "elapsed_seconds": time.monotonic() - begin}) + "\n")
        print(key, expected_count, "rows verified", flush=True)
    if {name: digest(path) for name, path in binaries.items()} != binary_hashes:
        raise ValueError("binary changed during measurement")
    for item in items:
        if digest(inputs / item["name"]) != item["sha256"]:
            raise ValueError("input changed during measurement")
    write_json(output / "complete.json", {"complete": True, "commands": len(commands),
                                         "binaries_and_inputs_unchanged": True})


if __name__ == "__main__":
    main()
