# SPDX-License-Identifier: Apache-2.0
"""Run ArUco latency and changing-frame measurements with local evidence."""

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path


def digest(path):
    with path.open("rb") as source:
        sha = hashlib.sha256()
        for block in iter(lambda: source.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def inputs(directory):
    path = directory / "input-manifest.json"
    if path.is_file():
        rows = json.loads(path.read_text())["inputs"]
        pairs = [(directory / row["name"], row["sha256"]) for row in rows]
    else:
        path = directory / "manifest.json"
        rows = json.loads(path.read_text())["scenes"]
        pairs = [(directory / row["path"], row["sha256"]) for row in rows]
    if not pairs or len({p for p, _ in pairs}) != len(pairs):
        raise ValueError(f"empty or repeated inputs in {path}")
    for image, expected in pairs:
        if digest(image) != expected:
            raise ValueError(f"wrong input bytes: {image}")
    return path, [p for p, _ in pairs]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cores", required=True)
    parser.add_argument("--groups", nargs="+", required=True,
                        choices=["fixed", "historical", "memory", "video", "native", "quality",
                                 "cpu-photos"])
    parser.add_argument("--old-gpu", type=Path)
    parser.add_argument("--gpu-only", action="store_true")
    parser.add_argument("--skip-video-cpu", action="store_true")
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()

    if args.rounds < 1:
        parser.error("--rounds must be positive")
    cores = [int(core) for core in args.cores.split(",")]
    if len(cores) != 4 or len(set(cores)) != 4 or not set(cores) <= os.sched_getaffinity(0):
        parser.error("--cores requires four distinct available CPU IDs")

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    aliases = out / "input-links"
    aliases.mkdir()
    build = args.build.resolve()
    cpu = build / "cpu/aruco3cuda_cpu_benchmark"
    video = build / "cpu/aruco3cuda_cpu_video"
    gpu = build / "bench/aruco3cuda_bench"
    sequence = build / "tools/gpu_quality/aruco3cuda_gpu_sequence"
    audit = build / "tools/gpu_quality/aruco3cuda_gpu_quality_audit"
    accepted = {}
    records = []

    def input_set(directory):
        if directory not in accepted:
            path, images = inputs(args.inputs.resolve() / directory)
            accepted[directory] = {"manifest": str(path), "manifest_sha256": digest(path)}
            return images
        return inputs(args.inputs.resolve() / directory)[1]

    def run(name, binary, extra, threads=1):
        root = out / name
        root.parent.mkdir(parents=True, exist_ok=True)
        binary = binary.resolve(strict=True)
        command = ["taskset", "-c", ",".join(map(str, cores[:threads])), str(binary), *extra]
        start = time.time()
        with root.with_suffix(".jsonl").open("x") as output, root.with_suffix(".stderr").open("x") as log:
            code = subprocess.run(command, stdout=output, stderr=log).returncode
        record = {"name": name, "command": command, "start_epoch": start, "end_epoch": time.time(),
                  "binary_sha256": digest(binary), "exit": code}
        records.append(record)
        with (out / "commands.jsonl").open("a") as journal:
            journal.write(json.dumps(record) + "\n")
        print(name, code, flush=True)
        return code

    def cpu_input_dir(label, images):
        target = aliases / label
        target.mkdir(exist_ok=True)
        for image in images:
            link = target / image.name
            if not link.exists():
                link.symlink_to(image)
        return target

    def flags(images):
        return [val for path in images for val in ("--input", str(path))]

    def still(label, images, mode, warmup, samples, routes):
        cpu_dir = cpu_input_dir(f"{label}-{mode}", images)
        for rep in range(1, args.rounds + 1):
            for route in (routes if rep % 2 else routes[::-1]):
                name = f"{label}/{mode}-{rep}-{route}"
                if route in ("cpu1", "cpu4"):
                    if args.gpu_only:
                        continue
                    threads = int(route[-1])
                    run(name, cpu, ["--input-dir", str(cpu_dir), "--aruco3", str(mode),
                                    "--warmup", str(warmup), "--iterations", str(samples),
                                    "--threads", str(threads), "--variants", "opencv,optimized",
                                    "--save-samples"], threads)
                    continue
                bench = args.old_gpu if route.startswith("old-") else gpu
                scope, memory = {
                    "hybrid": ("Hybrid", "M-Device"),
                    "hybrid-host": ("Hybrid", "M-Pageable"),
                    "resident": ("CUDA-Resident", "M-Device"),
                    "host": ("CUDA-E2E", "M-Pageable"),
                    "host-pinned": ("CUDA-E2E", "M-Pinned"),
                    "host-managed": ("CUDA-E2E", "M-Managed"),
                    "old-resident": ("CUDA-Resident", "M-Device"),
                    "old-host": ("CUDA-E2E", "M-Pageable"),
                    "old-default": ("CUDA-Resident", "M-Device"),
                }[route]
                run(name, bench, ["--route", scope, "--memory-mode", memory,
                                  "--use-aruco3", str(mode), "--threads", "1",
                                  "--max-candidates", "4096" if route == "old-default" else "16384",
                                  "--max-markers", "1024", "--warmup", str(warmup),
                                  "--latency-iterations", str(samples), "--throughput-frames", "0",
                                  "--min-marker-length-ratio", "0.05", "--save-samples", *flags(images)])

    def check_sequence(path, images, passes):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != len(images) * passes:
            raise ValueError(f"incomplete sequence in {path}")
        first = []
        for index, row in enumerate(rows):
            cycle, frame = divmod(index, len(images))
            if ((row["cycle"], row["frame"]) != (cycle, frame)
                    or not math.isfinite(row["ms"]) or row["ms"] <= 0):
                raise ValueError(f"invalid sample in {path}")
            detections = sorted(row["detections"], key=lambda tag: (tag["id"], tag["corners"]))
            if cycle == 0:
                first.append(detections)
            elif first[frame] != detections:
                raise ValueError(f"varying detection on replay in {path}")

    completed = False
    try:
        for group in args.groups:
            if group == "fixed":
                images = input_set("aruco-steady")
                photos = [p for p in images if p.name[:2].isdigit()]
                synth = [p for p in images if p not in photos]
                routes = ["cpu1", "cpu4", "resident", "host"]
                if args.old_gpu:
                    routes += ["old-resident", "old-host", "old-default"]
                still(group, images, 1, 20, 200, routes)
                still(group, synth + photos[:3], 0, 20, 200, routes)
            elif group == "cpu-photos":
                photos = [p for p in input_set("aruco-steady") if p.name[:2].isdigit()]
                still(group, photos, 0, 10, 50, ["cpu1", "cpu4"])
            elif group == "historical":
                still(group, input_set("aruco-historical"), 1, 30, 200,
                      ["cpu1", "cpu4", "hybrid", "resident", "host"])
            elif group == "memory":
                images = input_set("aruco-steady")
                subset = [p for p in images if p.name in
                          {"clean_1280x720_n4_s128.png", "cpu_3840x2160_n4_clean.png"}]
                if len(subset) != 2:
                    raise ValueError("expected both memory-scope control inputs")
                still(group, subset, 1, 30, 200,
                      ["hybrid", "hybrid-host", "resident", "host", "host-pinned", "host-managed"])
            elif group == "native":
                for directory, prefix in [("industrial-4k5k", "04_"), ("industrial-8k", "15_")]:
                    images = [p for p in input_set(directory) if p.name.startswith(prefix)]
                    if len(images) != 1:
                        raise ValueError(f"expected one control photo for {prefix}")
                    still(f"{group}-{prefix.rstrip('_')}", images, 1, 10, 50,
                          ["cpu1", "cpu4", "resident", "host"])
            elif group in ("video", "quality"):
                for directory in ("aruco-1080-balanced", "aruco-2160-balanced"):
                    images = input_set(directory)
                    if group == "quality":
                        for mode in ("1", "0"):
                            run(f"{group}/{directory}-{mode}", audit,
                                ["--images", str(images[0].parent), "--aruco3", mode,
                                 "--max-candidates", "16384", "--max-markers", "1024"])
                        continue
                    for rep in range(1, args.rounds + 1):
                        for scope in (("resident", "host") if rep % 2 else ("host", "resident")):
                            name = f"{group}/{directory}-{rep}-{scope}"
                            code = run(name, sequence,
                                       [scope, "1", "2", "10", "16384", *map(str, images)])
                            if not code:
                                check_sequence((out / name).with_suffix(".jsonl"), images, 10)
                        if not args.gpu_only and not args.skip_video_cpu:
                            run(f"{group}/{directory}-{rep}-cpu", video,
                                ["--frames", str(10 * len(images)), "--warmup", str(len(images)),
                                 "--workers", "4", "--opencv-threads", "1", "--aruco3", "1",
                                 "--route", "both", "--layout", "both", "--save-frames", *flags(images)]
                                + (["--reverse-order"] if rep % 2 == 0 else []), 4)
        for directory in accepted:
            inputs(args.inputs.resolve() / directory)
        completed = True
    finally:
        failures = [row["name"] for row in records if row["exit"]]
        (out / "inventory.json").write_text(json.dumps({
            "input_sets": accepted, "groups": args.groups, "cores": cores, "rounds": args.rounds,
            "commands": len(records), "failed_commands": failures,
            "completed": completed and not failures,
        }, indent=2) + "\n")

    if failures:
        raise SystemExit(f"{len(failures)} failed benchmark commands")


if __name__ == "__main__":
    main()
