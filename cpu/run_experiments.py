#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run independent paired CPU measurements and retain their environment and commands."""
import argparse
import datetime
import hashlib
import json
import os
import pathlib
import platform
import subprocess


def read_text(path):
    try:
        return pathlib.Path(path).read_text().strip()
    except OSError:
        return "unavailable"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cpu-list", default="5")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--throughput-frames", type=int, default=0)
    parser.add_argument("--aruco3", choices=["0", "1"], default="1")
    parser.add_argument("--dictionary", default="DICT_ARUCO_MIP_36h12")
    parser.add_argument("--variants", default="opencv,optimized")
    parser.add_argument("--filter", default="")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--background-note", default="Other user workloads remain running")
    args = parser.parse_args()
    cpus = {int(cpu) for cpu in args.cpu_list.split(",")}
    if args.runs < 1 or args.threads < 1 or args.iterations < 1 or args.warmup < 0:
        parser.error("runs, threads, iterations must be positive; warmup must be nonnegative")
    os.sched_setaffinity(0, cpus)
    destination = pathlib.Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    command = [args.binary, "--input-dir", args.input_dir, "--iterations", str(args.iterations),
               "--warmup", str(args.warmup), "--threads", str(args.threads), "--aruco3", args.aruco3,
               "--variants", args.variants, "--filter", args.filter,
               "--dictionary", args.dictionary,
               "--throughput-frames", str(args.throughput_frames)]
    if args.profile:
        command.append("--profile")
    cpu_info = {}
    for cpu in sorted(cpus):
        base = pathlib.Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq")
        cpu_info[cpu] = {name: read_text(base / name) for name in
                         ["scaling_governor", "scaling_cur_freq", "scaling_max_freq", "cpuinfo_max_freq"]}
    metadata = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command": command, "independent_processes": args.runs, "cpu_affinity": sorted(cpus),
        "platform": platform.uname()._asdict(), "cpu_clocks_khz": cpu_info,
        "aslr": read_text("/proc/sys/kernel/randomize_va_space"),
        "cpuinfo": read_text("/proc/cpuinfo"),
        "product_name": read_text("/sys/class/dmi/id/product_name"),
        "load_average_at_start": read_text("/proc/loadavg"),
        "binary_sha256": hashlib.sha256(pathlib.Path(args.binary).read_bytes()).hexdigest(),
        "opencv_provenance": read_text("/opt/opencv/share/aruco3cuda/opencv-provenance.json"),
        "power_mode": "Host configuration retained; no power or clock settings changed",
        "background_load": args.background_note + "; paired timings alternate route order",
        "kernel_time": "Not applicable: host-only CPU detector; no CUDA execution",
        "measurement": "Single-frame latency; serial throughput only when throughput_frames > 0",
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_diff_stat": subprocess.check_output(["git", "diff", "--stat"], text=True),
    }
    source_paths = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"], text=True).splitlines()
    metadata["source_sha256"] = {path: hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
                                  for path in source_paths if pathlib.Path(path).is_file() and
                                  pathlib.Path(path).suffix in {".cpp", ".hpp", ".py", ".txt"}}
    (destination / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for run in range(1, args.runs + 1):
        path = destination / f"run-{run}.jsonl"
        if path.exists():
            raise RuntimeError(f"Refusing to overwrite existing measurement: {path}")
        with path.open("w") as output:
            subprocess.run(command, stdout=output, check=True)
        print(f"Completed {path}", flush=True)


if __name__ == "__main__":
    main()
