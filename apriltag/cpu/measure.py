# SPDX-License-Identifier: Apache-2.0
"""Repeat the Linux host-only AprilTag sequence and industrial background comparisons."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(binary, destination, cores, extra):
    command = ["taskset", "-c", ",".join(map(str, cores)), str(binary),
               "--threads", str(len(cores)), *extra]
    with destination.open("x") as output:
        subprocess.run(command, check=True, stdout=output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("photos", type=Path, help="directory with the 16 prepared industrial PNGs")
    parser.add_argument("output", type=Path, help="new local output directory")
    parser.add_argument("--cores", required=True, help="four distinct physical cores, e.g. 5,6,7,8")
    args = parser.parse_args()
    cores = list(map(int, args.cores.split(",")))
    if len(cores) != 4 or len(set(cores)) != 4 or not set(cores) <= os.sched_getaffinity(0):
        raise ValueError("expected four distinct available CPU IDs")
    binary = args.binary.resolve(strict=True)
    photos = sorted(args.photos.glob("[01][0-9]_*.png"))
    if [p.name[:2] for p in photos] != [f"{i:02}" for i in range(1, 17)]:
        raise ValueError("expected photographs numbered 01 through 16")

    input_hashes = {path.name: sha256(path) for path in photos}
    manifest = Path(__file__).resolve().parents[2] / "tools/gpu_quality/steady-inputs.json"
    expected = {row["name"]:row["sha256"] for row in json.loads(manifest.read_text())["inputs"]
                if row["family"] == "industrial photograph"}
    if input_hashes != expected:
        raise ValueError("industrial photos differ from the fixed input manifest")

    # Record the actual library resolved by the loader. An installed consumer
    # can use a different library than a binary run directly from its build tree.
    linkage = subprocess.check_output(["ldd", binary], text=True)
    dependencies = [Path(line.split("=>", 1)[1].split()[0]) for line in linkage.splitlines()
                    if "=>" in line and "not found" not in line]
    hashes = {str(path): sha256(path) for path in [binary, *dependencies]}
    args.output.mkdir(parents=True, exist_ok=False)
    environment = {"platform": platform.platform(), "cores": cores, "binary_hashes": hashes,
                   "input_hashes": input_hashes}
    (args.output / "environment.json").write_text(json.dumps(environment, indent=2))
    for group in [cores[:1], cores]:
        for resolution in ["720p", "1080p"]:
            for rep in [1, 2, 3]:
                destination = args.output / f"stream-{len(group)}t-{resolution}-{rep}.csv"
                run(binary, destination, group, ["--stream", resolution, "--warmup", "16",
                                                 "--samples", "192"] + (["--cpu-first"] if rep == 2 else []))
        for photo in photos:
            run(binary, args.output / f"photo-{len(group)}t-{photo.stem}.csv", group,
                ["--image", str(photo), "--warmup", "2", "--samples", "10"])
    if any(sha256(Path(path)) != digest for path, digest in hashes.items()):
        raise RuntimeError("detector binary or a linked library changed during this run")


if __name__ == "__main__":
    main()
