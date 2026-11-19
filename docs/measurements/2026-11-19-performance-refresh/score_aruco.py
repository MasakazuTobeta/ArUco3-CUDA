# SPDX-License-Identifier: Apache-2.0
"""Score external ArUco audit frames against the exact prepared PNG corpus."""

import argparse
from collections import Counter
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import re

from PIL import Image


ROUTES = ("opencv", "optimized_cpu", "gpu")
DICTIONARY = "DICT_ARUCO_MIP_36h12"
CORNER_TOLERANCE_PX = 0.001


def load_scorer():
    source = Path(__file__).resolve().parents[3] / "apriltag/evaluation/score.py"
    spec = importlib.util.spec_from_file_location("ground_truth_score", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_corpus(manifest):
    raw = manifest.read_bytes()
    data = json.loads(raw)
    scenes = data["scenes"]
    if data["dictionary"] != DICTIONARY or not scenes:
        raise ValueError("expected an annotated ArUco MIP 36h12 corpus")
    if len({scene["name"] for scene in scenes}) != len(scenes):
        raise ValueError("duplicated scene name in the manifest")
    identities = {}
    for scene in scenes:
        filename = scene["path"]
        if Path(filename).name != filename or filename in identities:
            raise ValueError(f"nonlocal or repeated image in the manifest: {filename}")
        encoded = (manifest.parent / filename).read_bytes()
        if digest(encoded) != scene["sha256"]:
            raise ValueError(f"encoded image hash mismatch: {filename}")
        size = (scene["width_px"], scene["height_px"])
        with Image.open(io.BytesIO(encoded)) as image:
            if image.mode != "L" or image.format != "PNG" or image.size != size:
                raise ValueError(f"expected a Gray8 PNG with the declared dimensions: {filename}")
            pixel_sha256 = digest(image.tobytes())
        identities[filename] = (scene["name"], size, pixel_sha256)
    return scenes, identities, digest(raw)


def read_record(manifest, audit, mode, manifest_sha256):
    directory = audit.parent.parent
    command_name = f"quality/{audit.stem}"
    matching = [row for row in read_jsonl(directory / "commands.jsonl") if row["name"] == command_name]
    if len(matching) != 1 or matching[0]["exit"] not in (0, 2):
        raise ValueError(f"expected one complete audit command record for {command_name}")
    record = matching[0]
    command = record["command"]

    def option(key):
        if command.count(key) != 1 or command.index(key) + 1 >= len(command):
            raise ValueError(f"missing or ambiguous audit option: {key}")
        return command[command.index(key) + 1]

    if (len(command) < 4 or Path(command[3]).name != "aruco3cuda_gpu_quality_audit"
            or option("--aruco3") != str(mode) or "--direct" in command
            or Path(option("--images")).name != manifest.parent.name
            or re.fullmatch(r"[0-9a-f]{64}", record["binary_sha256"]) is None):
        raise ValueError("audit command, manifest or mode mismatch")

    # The command journal records a finished case before the full study has finished.
    # The PNG and decoded-pixel hashes still bind each frame during that interval.
    path = directory / "inventory.json"
    if path.is_file():
        entry = json.loads(path.read_text())["input_sets"][manifest.parent.name]
        if entry["manifest_sha256"] != manifest_sha256:
            raise ValueError("run inventory manifest hash mismatch")
    return record, int(option("--max-candidates")), int(option("--max-markers"))


def observations(audit, identities, mode, max_candidates, max_markers, marker_row):
    raw = audit.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    if len(rows) < 2 or rows[-1]["type"] != "completion":
        raise ValueError("missing audit completion record")
    completion = rows.pop()
    if completion["frames"] != len(rows):
        raise ValueError("audit completion frame count mismatch")

    output = {route: {} for route in ROUTES}
    order = []
    blank_count = 0
    blank_detections = Counter({route: 0 for route in ROUTES})
    blank_hashes = {size: digest(bytes([245]) * math.prod(size))
                    for size in {entry[1] for entry in identities.values()}}
    configuration = ("configuration", "opencv_version", "gpu_resize_profile",
                     "gpu_resize_backend", "gpu_carotene_x_fma", "gpu_strategy_identity")
    first_configuration = {key: rows[0][key] for key in configuration}
    for sequence, row in enumerate(rows):
        if (row["type"] != "frame" or row["sequence"] != sequence or row["dictionary"] != DICTIONARY
                or row["aruco3"] is not bool(mode) or row["graph"] is not True
                or row["configuration"]["max_candidates"] != max_candidates
                or row["configuration"]["max_markers"] != max_markers
                or any(row[key] != value for key, value in first_configuration.items())):
            raise ValueError(f"mixed configurations or nonsequential audit row: {sequence}")
        for route in ROUTES:
            for marker in row[route]:
                marker_row(marker, "corners")
        if row["name"] == "inserted_blank":
            size = (row["width"], row["height"])
            if (row["family"] != "sequence_cut" or row["has_ground_truth"] is not True
                    or row["truth"] or row["pixel_sha256"] != blank_hashes.get(size)
                    or len(order) != 7 * (blank_count + 1)):
                raise ValueError(f"invalid blank cut: {sequence}")
            blank_count += 1
            blank_detections.update({route: len(row[route]) for route in ROUTES})
            continue
        if row["name"] not in identities:
            raise ValueError(f"unexpected external image: {row['name']}")
        name, size, pixel_sha256 = identities[row["name"]]
        if (row["has_ground_truth"] is not False or (row["width"], row["height"]) != size
                or row["pixel_sha256"] != pixel_sha256):
            raise ValueError(f"decoded pixels or annotation mode mismatch: {name}")

        order.append(name)
        for route in ROUTES:
            detections = sorted(row[route], key=lambda marker: (marker["id"], marker["corners"]))
            prior = output[route].setdefault(name, detections)
            if prior != detections:
                raise ValueError(f"detector output changed on replay: {route}, {name}")

    count = len(identities)
    replay_count = min(8, count)
    names = {identity[0] for identity in identities.values()}
    if (set(order[:count]) != names or order[count:] != order[:replay_count]
            or len(order) != count + replay_count or blank_count != len(order) // 7):
        missing = names - set(order)
        raise ValueError(f"audit image/replay coverage mismatch. Missing scenes {sorted(missing)}")
    evidence = {
        "audit_sha256": digest(raw),
        "configuration": first_configuration,
        "replayed_frames": replay_count,
        "blank_cuts": blank_count,
        "blank_cut_false_positives": blank_detections,
        "audit_completion": completion,
    }
    return output, evidence


def compare(scorer, baseline, candidate):
    base_matches, base_fp, base_summary = baseline
    cand_matches, cand_fp, cand_summary = candidate
    shared = base_matches.keys() & cand_matches.keys()
    deltas = []
    for name, index in sorted(shared):
        base_squared, base_max = base_matches[name, index]
        cand_squared, cand_max = cand_matches[name, index]
        deltas.append({"scene": name, "truth_index": index,
                       "max_corner_delta_px": cand_max - base_max,
                       "corner_rmse_delta_px": math.sqrt(cand_squared / 4)
                       - math.sqrt(base_squared / 4)})
    worst = max(deltas, key=lambda row: row["max_corner_delta_px"], default=None)
    regression = None
    if worst is not None and worst["max_corner_delta_px"] > 0:
        key = (worst["scene"], worst["truth_index"])
        regression = {"scene": key[0], "truth_index": key[1],
                      "baseline_max_px": base_matches[key][1],
                      "candidate_max_px": cand_matches[key][1]}
    return {
        "baseline": base_summary,
        "candidate": cand_summary,
        "lost_truths": sorted(base_matches.keys() - shared),
        "extra_truths": sorted(cand_matches.keys() - shared),
        "new_fp_frames": sorted(name for name in base_fp if cand_fp[name] > base_fp[name]),
        "largest_shared_corner_regression": regression,
        "paired_corner_rmse_px": [scorer.rmse(base_matches, shared),
                                  scorer.rmse(cand_matches, shared)],
        "corner_quality_tolerance_px": CORNER_TOLERANCE_PX,
        "corner_quality_regressions": [
            row for row in deltas
            if max(row["max_corner_delta_px"], row["corner_rmse_delta_px"]) > CORNER_TOLERANCE_PX
        ],
        "shared_truth_corner_deltas": deltas,
    }


def compare_to_opencv(scorer, scenes, observations, executable):
    reference = [
        {"name": scene["name"],
         "markers": [{"id": tag["id"], "corners_px": tag["corners"]}
                     for tag in observations["opencv"][scene["name"]]]}
        for scene in scenes
    ]
    output = {}
    for route in ("optimized_cpu", "gpu"):
        matches, _, summary = scorer.score(reference, observations[route], executable)
        output[route] = {
            **summary,
            "corner_tolerance_px": CORNER_TOLERANCE_PX,
            "outside_tolerance": [key for key in sorted(matches)
                                  if matches[key][1] > CORNER_TOLERANCE_PX],
        }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("audit", type=Path)
    parser.add_argument("--scorer", type=Path, required=True)
    parser.add_argument("--aruco3", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()

    scenes, identities, manifest_sha256 = read_corpus(args.manifest)
    record, candidates, markers = read_record(args.manifest, args.audit, args.aruco3, manifest_sha256)
    scorer = load_scorer()
    routes, evidence = observations(
        args.audit, identities, args.aruco3, candidates, markers, scorer.marker_row)
    executable = args.scorer.resolve(strict=True)
    scores = {route: scorer.score(scenes, rows, executable) for route, rows in routes.items()}
    summary = {
        "manifest_sha256": manifest_sha256,
        "scorer_sha256": digest(executable.read_bytes()),
        "audit_binary_sha256": record["binary_sha256"],
        "audit_exit": record["exit"],
        "aruco3": bool(args.aruco3),
        **evidence,
        "opencv_vs_cuda": compare(scorer, scores["opencv"], scores["gpu"]),
        "opencv_vs_optimized_cpu": compare(scorer, scores["opencv"], scores["optimized_cpu"]),
        "direct_opencv_compatibility": compare_to_opencv(scorer, scenes, routes, executable),
    }
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
