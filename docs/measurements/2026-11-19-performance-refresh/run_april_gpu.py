# SPDX-License-Identifier: Apache-2.0
"""Repeat the paired AprilTag CUDA latency study with the pinned CPP adapters."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--binaries", type=Path, required=True)
    parser.add_argument("--original-inputs", type=Path, required=True)
    parser.add_argument("--new-inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", required=True)
    parser.add_argument("--sets", nargs="+",
                        default=["640", "720", "1080", "2160", "field", "photos", "controls"])
    parser.add_argument("--rounds", type=int, default=3)
    opts = parser.parse_args()
    if opts.rounds < 1:
        parser.error("--rounds must be positive")
    opts.output.mkdir(parents=True, exist_ok=False)

    original = opts.original_inputs
    field = original / "april-field"
    if not field.is_dir():
        field = original / "field"
    cases = {
        "640": (original / "corpus-36h11", 2, 10),
        "720": (original / "corpus-720-36h11", 2, 10),
        "1080": (opts.new_inputs / "tag-1080", 2, 10),
        "2160": (opts.new_inputs / "tag-2160-balanced", 2, 10),
        "field": (field, 8, 60),
    }
    score_cpp = opts.source / "apriltag/evaluation/score.cpp"
    scorer = opts.output / "score"
    subprocess.run(["c++", "-std=c++17", "-O2", "-I" + str(opts.source / "tools/evaluate"),
                    "-I" + str(opts.source / "tools/report"), str(score_cpp),
                    str(opts.source / "tools/evaluate/accuracy.cpp"), "-o", str(scorer)], check=True)
    failures = []

    def stage_manifest(label, manifest, selected=None):
        data = json.loads(manifest.read_bytes())
        if data.get("dictionary", "DICT_APRILTAG_36h11") != "DICT_APRILTAG_36h11":
            raise ValueError(f"unexpected Dictionary in {manifest}")
        scenes = data["scenes"] if selected is None else selected
        directory = opts.output / "derived-manifests" / label
        directory.mkdir(parents=True)
        for scene in scenes:
            filename = Path(scene["path"])
            if filename.name != scene["path"]:
                raise ValueError(f"expected a local image filename in {manifest}")
            (directory / filename).symlink_to((manifest.parent / filename).resolve(strict=True))
        staged = directory / "manifest.json"
        staged.write_text(json.dumps({**data, "dictionary": "DICT_APRILTAG_36h11", "scenes": scenes}))
        return staged

    def sample(label, manifest, rounds, warmup, samples, modes):
        if not manifest.is_file():
            failures.append({"case": label, "error": f"missing {manifest}"})
            return
        root = opts.output / label
        root.mkdir()
        for rep in range(1, rounds + 1):
            for scope in ("device", "host"):
                routes = modes if rep % 2 else modes[::-1]
                paths = {}
                for name, backend, extra in routes:
                    path = root / f"{rep}-{scope}-{name}"
                    exe = opts.binaries / f"apriltag_{backend}_benchmark"
                    command = ["taskset", "-c", opts.cpu, sys.executable,
                               str(opts.source / "apriltag/benchmark/run_sequence.py"),
                               str(exe), str(manifest), str(path), "--backend", backend,
                               "--scope", scope, "--warmup", str(warmup), "--samples",
                               str(samples), *extra]
                    with (root / f"{path.name}.log").open("x") as log:
                        log.write(json.dumps(command) + "\n")
                        log.flush()
                        code = subprocess.run(command, stdout=log, stderr=log).returncode
                    print(f"{label} {path.name}: {code}", flush=True)
                    if code:
                        failures.append({"case": label, "run": path.name, "exit": code})
                    else:
                        paths[name] = path / "observations.jsonl"
                if "nvidia" in paths and "markers" in json.loads(manifest.read_bytes())["scenes"][0]:
                    for name, obs in paths.items():
                        if name == "nvidia":
                            continue
                        command = [sys.executable, str(opts.source / "apriltag/evaluation/score.py"),
                                   str(manifest), str(paths["nvidia"]), str(obs), "--scorer", str(scorer)]
                        with (root / f"{rep}-{scope}-{name}-quality.json").open("x") as quality:
                            code = subprocess.run(command, stdout=quality).returncode
                        if code:
                            failures.append({"case": label, "run": f"{rep}-{scope}-{name}-quality", "exit": code})
                if label.startswith("photo-"):
                    for name, obs in paths.items():
                        if any(json.loads(line)["detections"] for line in obs.read_text().splitlines()):
                            failures.append({"case": label, "run": f"{rep}-{scope}-{name}",
                                             "error": "detections in an industrial background"})

    modes = [("project", "project", ["--decimate", "0"]), ("nvidia", "nvidia", [])]
    for label in opts.sets:
        if label in cases:
            directory, warmup, samples = cases[label]
            variants = modes + [("fixed", "project", ["--decimate", "1"])] if label == "field" else modes
            sample(label, directory / "manifest.json", opts.rounds, warmup, samples, variants)
        elif label == "photos":
            directory = original / "april-photos"
            manifest = directory / "manifest.json"
            if not manifest.is_file():
                failures.append({"case": label, "error": f"missing {manifest}"})
                continue
            photos = json.loads(manifest.read_bytes())
            for scene in photos["scenes"]:
                # Layout is fixed inside each process, so each photo is a one-scene sequence.
                single = stage_manifest(scene["name"], manifest, [scene])
                sample("photo-" + label + "-" + scene["name"],
                       single, opts.rounds, 5, 30, modes)
        elif label == "native-photos":
            for directory, name in [("industrial-4k5k", "04_steel_pipe_storage"),
                                    ("industrial-8k", "15_brewery_machine_room")]:
                manifest = opts.new_inputs / directory / f"manifest-{name}.json"
                sample("photo-native-" + name, stage_manifest(name, manifest),
                       opts.rounds, 3, 20, modes)
        elif label == "controls":
            for name in ("nvidia-fixture", "tag-damage"):
                directory = original / name
                if not (directory / "manifest.json").is_file():
                    directory = opts.new_inputs / name
                manifest = directory / "manifest.json"
                if manifest.is_file():
                    manifest = stage_manifest(name, manifest)
                sample(name, manifest, 1, 1, 3, modes)
        else:
            parser.error(f"unknown input group {label}")

    binary_hashes = {
        name: hashlib.sha256((opts.binaries / f"apriltag_{name}_benchmark").read_bytes()).hexdigest()
        for name in ("project", "nvidia")
    }
    (opts.output / "result.json").write_text(json.dumps({
        "completed": not failures, "failures": failures, "rounds": opts.rounds,
        "binary_hashes": binary_hashes,
    }, indent=2) + "\n")
    if failures:
        raise SystemExit(f"{len(failures)} failed cases")


if __name__ == "__main__":
    main()
