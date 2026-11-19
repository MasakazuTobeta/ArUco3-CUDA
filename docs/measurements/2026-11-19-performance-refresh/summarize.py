# SPDX-License-Identifier: Apache-2.0
"""Verify retained raw timing samples and summarize complete benchmark commands."""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def json_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def distribution(values):
    if not values or any(type(value) not in (int, float) or not math.isfinite(value)
                         or value <= 0.0 for value in values):
        raise ValueError("expected positive finite timing samples")
    ordered = sorted(values)
    return {f"p{k}_ms": ordered[math.ceil(k * len(ordered) / 100) - 1] for k in (50, 95, 99)}


def summarize_replicates(sets):
    return {key: {
        "processes": len(sets_for_key),
        "samples": sum(samples for _, samples in sets_for_key),
        **{metric: statistics.median(row[metric] for row, _ in sets_for_key)
           for metric in sets_for_key[0][0]},
        "process_p50_range_ms": [min(row["p50_ms"] for row, _ in sets_for_key),
                                 max(row["p50_ms"] for row, _ in sets_for_key)],
    } for key, sets_for_key in sorted(sets.items())}


def april_runs(root):
    runs = defaultdict(list)
    observations = {}
    for path in sorted(root.glob("*/*/timings.jsonl")):
        case = path.parent.parent.name
        rep, scope, backend = path.parent.name.split("-", 2)
        key = f"{case}/{scope}/{backend}"
        rows = json_lines(path)
        timing = [row["ms"] for row in rows]
        observed = json_lines(path.parent / "observations.jsonl")
        scenarios = [ob["scene"] for ob in observed]
        if not scenarios or len(scenarios) != len(set(scenarios)) or len(rows) % len(scenarios):
            raise ValueError(f"wrong sequence length in {path}")
        for index, row in enumerate(rows):
            cycle, frame = divmod(index, len(scenarios))
            if (row["cycle"], row["scene"]) != (cycle, scenarios[frame]):
                raise ValueError(f"incorrect sequence index in {path}")
        if observations.setdefault(key, observed) != observed:
            raise ValueError(f"observation changed in {key}")
        measured = distribution(timing)
        measured["serial_calls_s"] = 1000.0 * len(timing) / math.fsum(timing)
        runs[key].append((measured, len(rows)))
    return summarize_replicates(runs)


def check_distribution(values, reported, tolerance, samples, path):
    if len(values) != samples:
        raise ValueError(f"wrong sample count in {path}")
    measured = distribution(values)
    if any(abs(reported[metric] - expected) > tolerance for metric, expected in measured.items()):
        raise ValueError(f"incorrect reported statistics in {path}")
    return measured


def aruco_runs(root, cpu_only=False):
    runs = defaultdict(list)
    counts = {}
    sequence_observations = {}
    for job in json_lines(root / "commands.jsonl"):
        if job["exit"] != 0:
            continue
        name = job["name"]
        source = (root / name).with_suffix(".jsonl")
        group, label = name.split("/", 1)
        rows = json_lines(source)
        if group == "video":
            case, _, route = label.rsplit("-", 2)
            if cpu_only and route != "cpu":
                continue
            if route == "cpu":
                for row in rows:
                    if row["type"] != "video_measurement":
                        continue
                    if not row["exact_ids_and_ordered_corners"] or row["timed_frames_checked"] != row["frames"]:
                        raise ValueError(f"incomplete video validation in {source}")
                    key = f"{group}/{case}/{row['route']}/{row['layout']}"
                    samples = [frame["service_ms"] for frame in rows if frame["type"] == "video_frame"
                               and frame["route"] == row["route"] and frame["layout"] == row["layout"]]
                    reported = {f"p{k}_ms": row[f"service_p{k}_ms"] for k in (50, 95, 99)}
                    measured = check_distribution(samples, reported, 0.000001, row["frames"], source)
                    measured["saturated_fps"] = row["saturated_throughput_fps"]
                    runs[key].append((measured, len(samples)))
            else:
                paths = job["command"][9:]
                passes = int(job["command"][7])
                if len(paths) < 2 or len(rows) != passes * len(paths):
                    raise ValueError(f"incomplete changing-frame measurement in {source}")
                detected = []
                for index, row in enumerate(rows):
                    cycle, frame = divmod(index, len(paths))
                    if (row["cycle"], row["frame"]) != (cycle, frame):
                        raise ValueError(f"incorrect changing-frame index in {source}")
                    markers = sorted(row["detections"], key=lambda tag: (tag["id"], tag["corners"]))
                    if cycle == 0:
                        detected.append(markers)
                    elif detected[frame] != markers:
                        raise ValueError(f"changed observation on replay in {source}")
                if sequence_observations.setdefault(case, detected) != detected:
                    raise ValueError(f"changed observation across timing scopes or trials in {source}")
                timing = [row["ms"] for row in rows]
                measured = distribution(timing)
                measured["serial_calls_s"] = 1000.0 * len(timing) / math.fsum(timing)
                runs[f"{group}/{case}/{route}"].append((measured, len(rows)))
            continue
        if group == "quality":
            continue
        mode, _, route = label.split("-", 2)
        if cpu_only and route not in ("cpu1", "cpu4"):
            continue
        seen = set()
        for row in rows:
            if row["type"] != "measurement":
                continue
            if route in ("cpu1", "cpu4"):
                if row["compatibility_discrepancy"]:
                    raise ValueError(f'ArUco CPU compatibility failure in {source}')
                key = f"{group}/{mode}/{route}/{row['variant']}/{row['scene']}"
                samples = row["latency_samples_ms"]
                measured = check_distribution(samples, row, 0.000001, rows[0]["iterations"], source)
                count = row["detections"]
            else:
                key = f"{group}/{mode}/{route}/{Path(row['image_path']).name}"
                samples = row["end_to_end_samples_ms"]
                measured = check_distribution(samples, row["end_to_end"], 0.000101,
                                              row["conditions"]["latency_iterations"], source)
                count = row["detection_count"]
            if key in seen or counts.setdefault(key, count) != count:
                raise ValueError(f"duplicate row or unstable counts in {source}: {key}")
            seen.add(key)
            runs[key].append((measured, len(samples)))
    result = summarize_replicates(runs)
    for key, count in counts.items():
        result[key]["detections"] = count
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--april", type=Path, action="append", default=[])
    parser.add_argument("--aruco", type=Path, action="append", default=[])
    parser.add_argument("--aruco-cpu", type=Path, action="append", default=[])
    args = parser.parse_args()
    result = {"april": {str(root): april_runs(root) for root in args.april},
              "aruco": {str(root): aruco_runs(root) for root in args.aruco},
              "aruco_cpu": {str(root): aruco_runs(root, cpu_only=True) for root in args.aruco_cpu}}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
