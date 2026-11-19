# SPDX-License-Identifier: Apache-2.0
"""Score detector observations against rendered markers, pairing by position before ID."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess


def load_observations(path, names):
    records = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        name = row["scene"]
        if name in records or not isinstance(row["detections"], list):
            raise ValueError(f"invalid or duplicate frame: {name}")
        records[name] = row["detections"]
    if names != records.keys():
        raise ValueError(f"{path}: missing or unexpected frames: {names ^ records.keys()}")
    return records


def marker_row(marker, key):
    corners = marker[key]
    if (type(marker["id"]) is not int or marker["id"] < 0 or len(corners) != 8
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in corners)):
        raise ValueError(f"invalid marker: {marker}")
    return " ".join(map(str, [marker["id"], *corners]))


def score(scenes, observations, executable):
    rows = []
    for scene in scenes:
        truth = scene["markers"]
        detected = observations[scene["name"]]
        rows.append(f"{len(truth)} {len(detected)}")
        rows.extend(marker_row(marker, "corners_px") for marker in truth)
        rows.extend(marker_row(marker, "corners") for marker in detected)
    output = subprocess.run([executable], input="\n".join(rows) + "\n", text=True,
                            capture_output=True, check=True).stdout.splitlines()
    if len(output) != len(scenes):
        raise ValueError("incomplete score output")
    matches = {}
    false_positives = {}
    wrong_rotation = []
    wrong_id = []
    for scene, line in zip(scenes, output):
        name = scene["name"]
        values = line.split()
        if len(values) != 1 + 5 * len(scene["markers"]):
            raise ValueError(f"incomplete scores: {name}")
        false_positives[name] = int(values[0])
        for i in range(len(scene["markers"])):
            found, oriented, squared, worst, mismatch = values[1 + i*5:6 + i*5]
            key = (name, i)
            if int(mismatch):
                wrong_id.append(key)
            if int(found):
                matches[key] = (float(squared), float(worst))
                if not int(oriented):
                    wrong_rotation.append(key)
    truth_count = sum(len(s["markers"]) for s in scenes)
    return matches, false_positives, {
        "frames": len(scenes), "truth": truth_count, "tp": len(matches),
        "fn": truth_count - len(matches), "fp": sum(false_positives.values()),
        "id_errors": wrong_id, "rotation_errors": wrong_rotation,
        "corner_rmse_px": rmse(matches, matches),
        "worst_corner_px": max((v[1] for v in matches.values()), default=None),
    }


def rmse(matches, keys):
    if not keys:
        return None
    return math.sqrt(sum(matches[k][0] for k in keys) / (4 * len(keys)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--scorer", type=Path, required=True)
    args = parser.parse_args()
    scenes = json.loads(args.manifest.read_bytes())["scenes"]
    names = {s["name"] for s in scenes}
    if not scenes or len(names) != len(scenes):
        raise ValueError("empty or duplicate-scene manifest")
    for scene in scenes:
        path = args.manifest.parent / scene["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != scene["sha256"]:
            raise ValueError(f"input checksum mismatch: {path}")
    baseline, fp_base, bs = score(scenes, load_observations(args.baseline, names), args.scorer)
    candidate, fp_new, cs = score(scenes, load_observations(args.candidate, names), args.scorer)
    shared = baseline.keys() & candidate.keys()
    # A lower paired RMSE can hide a single badly localized tag in a crowded frame.
    regression = None
    if shared:
        worst = max(sorted(shared), key=lambda k: candidate[k][1] - baseline[k][1])
        if candidate[worst][1] > baseline[worst][1]:
            regression = {"scene": worst[0], "truth_index": worst[1],
                          "baseline_max_px": baseline[worst][1], "candidate_max_px": candidate[worst][1]}
    print(json.dumps({"baseline": bs, "candidate": cs, "lost_truths": sorted(baseline.keys() - shared),
                      "extra_truths": sorted(candidate.keys() - shared),
                      "new_fp_frames": sorted(n for n in names if fp_new[n] > fp_base[n]),
                      "largest_shared_corner_regression": regression,
                      "paired_corner_rmse_px": [rmse(baseline, shared), rmse(candidate, shared)]},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
