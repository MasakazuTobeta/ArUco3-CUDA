#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Aggregate measurement JSONL into tables and crossover points.

Purpose:
    Read the JSONL emitted by the benchmark harness and arrange it so that
    latency and throughput can be compared per route. Every measurement in every
    file that was given is included, so that favorable results cannot be
    cherry-picked.

Usage:
    aggregate.py results.jsonl [more.jsonl ...] [--format table|csv|json]
"""
import argparse
import json
import sys
from collections import defaultdict

# Schema versions that can be read. Versions may share this set only when they
# emit the same set of keys, each with the same meaning, over the same measured
# intervals. Silently mixing versions that do not would make the same key refer
# to something else, or make a table silently sparse.
#
# 4 and 5 satisfy it: no key was added or renamed and the intervals are
# identical, only the precision of cuda_toolkit changed, so the committed
# version-4 files aggregate together with anything measured from now on.
#
# 3 does not, even though its intervals match 4: a version-3 measurement line
# has no startup and no cuda_context_ms, so the startup table would silently
# drop those rows while the context cost of the version-4 rows was attributed to
# them. Version 2 and earlier differ in the interval itself - version 3 took
# image loading and the checksum out of the measured interval of the CPU route.
SUPPORTED_SCHEMA_VERSIONS = {4, 5}


def short_name(path):
    """Take only the file name of an image path, for display."""
    if not path:
        return "(unknown)"
    return path.rsplit("/", 1)[-1]


def machine_label(environment):
    """Short name that distinguishes measurements by machine.

    When results from several machines are listed together, there is nothing to
    compare unless it is clear which row belongs to which machine. Machines that
    report a board name use it; the others fall back to the GPU name.
    """
    if environment is None:
        return "(unknown)"
    model = environment.get("platform_model")
    if model:
        # "Jetson AGX Orin Developer Kit" -> "Jetson AGX Orin"
        return model.replace(" Developer Kit", "").strip()
    gpu = environment.get("gpu_name")
    if gpu:
        return gpu
    return environment.get("hostname") or "(unknown)"


def environment_key(environment):
    """Identity of the machine an environment line describes.

    Used to collapse repeated environment lines. The CUDA Toolkit is compared at
    minor-version precision only: a version-4 record says 11.4 where a version-5
    record of the same Toolkit says 11.4.315, and the same machine measured
    across that change must collapse to one entry rather than be listed twice.
    The cost is that two genuinely different patch versions of the same minor
    version collapse as well, and only the first is printed.
    """
    toolkit = ".".join((environment.get("cuda_toolkit") or "").split(".")[:2])
    return (environment.get("hostname"), environment.get("gpu_name"), toolkit,
            environment.get("power_mode"))


def load_records(paths):
    """Read the JSONL and split it into environment information and measurements.

    In the JSONL the first line is environment information and the rest are
    measurements. Each measurement is associated with the environment record that
    appeared most recently before it. This keeps the association correct, by line
    order, even when one file contains several environment records (for example
    when results from several machines were concatenated).
    """
    environments = []
    measurements = []
    for path in paths:
        current_environment = None
        with open(path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise SystemExit(f"{path}:{line_number}: not readable as JSON: {error}")
                version = record.get("schema_version")
                if version not in SUPPORTED_SCHEMA_VERSIONS:
                    raise SystemExit(
                        f"{path}:{line_number}: schema_version {version} cannot be read. "
                        f"Supported: {sorted(SUPPORTED_SCHEMA_VERSIONS)}. "
                        "Measure again, or aggregate old results separately")
                kind = record.get("type")
                if kind == "environment":
                    record["_source"] = path
                    current_environment = record
                    # Collapse repeated environment lines for the same machine.
                    key = environment_key(record)
                    if key not in {environment_key(e) for e in environments}:
                        environments.append(record)
                elif kind == "measurement":
                    record["_source"] = path
                    if current_environment is None:
                        raise SystemExit(
                            f"{path}:{line_number}: a measurement appears before any environment line")
                    record["_environment"] = current_environment
                    measurements.append(record)
                else:
                    raise SystemExit(f"{path}:{line_number}: unknown type: {kind}")
    return environments, measurements


def format_environment(environment):
    gpu = environment.get("gpu_name") or "(no GPU)"
    integrated = "integrated" if environment.get("gpu_integrated") else "discrete"
    return (
        f"host={environment.get('hostname')} os={environment.get('os')}\n"
        f"  arch={environment.get('architecture')} cores={environment.get('cpu_online_cores')}"
        f" opencv={environment.get('opencv_version')} threads={environment.get('opencv_threads')}\n"
        f"  gpu={gpu} ({integrated}) cc={environment.get('gpu_compute_capability') or '-'}"
        f" driver={environment.get('driver_version') or '-'}"
        f" cuda={environment.get('cuda_toolkit') or '-'}\n"
        f"  model={environment.get('platform_model') or '-'}"
        f" platform={environment.get('platform_release') or '-'}\n"
        f"  cpu={environment.get('cpu_topology') or '-'}"
        f" affinity={environment.get('cpu_affinity') or '-'}"
        f" aslr={environment.get('address_randomization') or '-'}\n"
        f"  cuda_context={environment.get('cuda_context_ms') or 0:.1f} ms (once per process)\n"
        f"  power_mode={environment.get('power_mode') or '(not collected)'}"
        f" clock={environment.get('gpu_max_clock_mhz') or '-'} MHz (max)"
        f" / {environment.get('gpu_current_clock_mhz') or '-'} MHz (current)"
    )


def measurement_rows(measurements):
    rows = []
    for record in measurements:
        end_to_end = record.get("end_to_end") or {}
        kernel = record.get("kernel")
        rows.append(
            {
                "machine": machine_label(record.get("_environment")),
                "route": record.get("route"),
                "memory_mode": record.get("memory_mode"),
                "resolution": f"{record.get('width_px')}x{record.get('height_px')}",
                "markers": record.get("detection_count"),
                "fxfy": (record.get("conditions") or {}).get("fxfy_effective"),
                "aruco3": (record.get("conditions") or {}).get("use_aruco3_detection"),
                "p50_ms": end_to_end.get("p50_ms"),
                "p95_ms": end_to_end.get("p95_ms"),
                "p99_ms": end_to_end.get("p99_ms"),
                "kernel_p50_ms": (kernel or {}).get("p50_ms") if kernel else None,
                "gpu_stage_p50_ms": ((record.get("stages") or {}).get("gpu") or {}).get("p50_ms"),
                "cpu_stage_p50_ms": ((record.get("stages") or {}).get("cpu") or {}).get("p50_ms"),
                "first_result_ms": (record.get("startup") or {}).get("time_to_first_result_ms"),
                "first_frame_ms": (record.get("startup") or {}).get("first_frame_ms"),
                "_context_ms": (record.get("_environment") or {}).get("cuda_context_ms"),
                "fps": record.get("throughput_fps"),
                "image": record.get("image_path"),
            }
        )
    return rows


def print_table(rows):
    header = [
        "machine", "route", "memory", "resolution", "markers", "fxfy",
        "p50_ms", "p95_ms", "p99_ms", "gpu_p50", "cpu_p50", "fps",
    ]
    widths = {name: len(name) for name in header}
    formatted = []
    for row in rows:
        values = {
            "machine": str(row["machine"]),
            "route": str(row["route"]),
            "memory": str(row["memory_mode"]),
            "resolution": str(row["resolution"]),
            "markers": str(row["markers"]),
            "fxfy": f"{row['fxfy']:.4f}" if row["fxfy"] is not None else "-",
            "p50_ms": f"{row['p50_ms']:.3f}" if row["p50_ms"] is not None else "-",
            "p95_ms": f"{row['p95_ms']:.3f}" if row["p95_ms"] is not None else "-",
            "p99_ms": f"{row['p99_ms']:.3f}" if row["p99_ms"] is not None else "-",
            "gpu_p50": f"{row['gpu_stage_p50_ms']:.3f}" if row["gpu_stage_p50_ms"] is not None else "-",
            "cpu_p50": f"{row['cpu_stage_p50_ms']:.3f}" if row["cpu_stage_p50_ms"] is not None else "-",
            "fps": f"{row['fps']:.1f}" if row["fps"] is not None else "-",
        }
        for name in header:
            widths[name] = max(widths[name], len(values[name]))
        formatted.append(values)

    print("  ".join(name.ljust(widths[name]) for name in header))
    print("  ".join("-" * widths[name] for name in header))
    for values in formatted:
        print("  ".join(values[name].ljust(widths[name]) for name in header))


def print_crossover(rows):
    """Show the ratio between routes when several routes share a condition.

    Conditions where the CPU is faster are always kept in the table, so that
    favorable results are not the only ones shown.
    """
    grouped = defaultdict(dict)
    for row in rows:
        # Comparing routes across machines means something different, so the
        # machine is part of the key.
        key = (row["machine"], row["image"], row["resolution"], row["markers"], row["fxfy"])
        grouped[key][(row["route"], row["memory_mode"])] = row

    comparable = {key: value for key, value in grouped.items() if len(value) > 1}
    if not comparable:
        print("\nOnly one route is present, so no crossover point is computed.")
        return

    print("\n=== Route comparison (ratio against the CPU p50) ===")
    for key, routes in sorted(comparable.items(), key=lambda item: str(item[0])):
        cpu = next((row for (route, _), row in routes.items() if route == "CPU"), None)
        if cpu is None or not cpu["p50_ms"]:
            continue
        print(f"{key[0]} {short_name(key[1])} {key[2]} markers={key[3]} fxfy={key[4]}")
        for (route, memory), row in sorted(routes.items()):
            if not row["p50_ms"]:
                continue
            ratio = row["p50_ms"] / cpu["p50_ms"]
            if row is cpu:
                print(f"  {route:<14} {memory:<12} p50={row['p50_ms']:.3f} ms  (baseline)")
                continue
            # Differences within 5% are buried in measurement spread, so make no
            # claim about which is faster. Getting the order of these checks
            # wrong would turn a hair above 1.00 into "CPU is faster" and leave
            # the equivalence band effective on one side only.
            if abs(ratio - 1.0) < 0.05:
                verdict = "equivalent"
            elif ratio > 1.0:
                verdict = "CPU is faster"
            else:
                verdict = "this route is faster"
            print(f"  {route:<14} {memory:<12} p50={row['p50_ms']:.3f} ms  ratio={ratio:.3f}  {verdict}")


def print_startup(rows):
    """Show the startup cost per route.

    It does not appear in the post-warm-up quantiles. For a single detection or a
    short burst, the startup cost dominates the steady-state difference. From
    that difference, also report how many frames it takes to pay the cost back.
    """
    grouped = defaultdict(dict)
    for row in rows:
        key = (row["machine"], row["image"], row["resolution"], row["markers"])
        entry = grouped[key].setdefault((row["route"], row["memory_mode"]), [])
        if row["first_result_ms"] is not None:
            entry.append(row)

    # CUDA context creation is a per-process cost and does not show up per image.
    context_ms = 0.0
    for row in rows:
        value = (row.get("_context_ms") or 0.0)
        context_ms = max(context_ms, value)

    printed = False
    for key, routes in sorted(grouped.items(), key=lambda item: str(item[0])):
        cpu_rows = routes.get(("CPU", "N/A"))
        if not cpu_rows:
            continue
        cpu = cpu_rows[0]
        for (route, memory), items in sorted(routes.items()):
            if not items or route == "CPU":
                continue
            row = items[0]
            if not printed:
                print("\n=== Startup cost (does not appear in the post-warm-up quantiles) ===")
                print("Per route: the time until the first result is available, and how many")
                print("frames the steady-state difference needs to pay that cost back.")
                printed = True
            gain = cpu["p50_ms"] - row["p50_ms"]
            # Add CUDA context creation to the CUDA routes only. It happens once
            # per process and does not show up in the per-image measurements.
            extra = (row["first_result_ms"] + context_ms) - cpu["first_result_ms"]
            print(f"{key[0]} {short_name(key[1])} {key[2]} markers={key[3]}")
            print(f"  CPU            to first frame {cpu['first_result_ms']:8.3f} ms "
                  f"(detect {cpu['first_frame_ms']:.3f} ms)  steady {cpu['p50_ms']:.3f} ms")
            print(f"  {route:<14} to first frame {row['first_result_ms']:8.3f} ms "
                  f"(detect {row['first_frame_ms']:.3f} ms)  steady {row['p50_ms']:.3f} ms  "
                  f"[{memory}]")
            print(f"    + CUDA context creation {context_ms:.1f} ms (once per process)")
            if gain > 0:
                print(f"    extra startup cost {extra:.1f} ms / gain per frame "
                      f"{gain:.3f} ms -> paid back after about {extra / gain:.0f} frames")
            else:
                print(f"    extra startup cost {extra:.1f} ms. Not faster in steady state either, so it is never paid back")


def print_run_variance(rows):
    """Show the run-to-run spread when the same condition was measured repeatedly.

    Per-process memory layout (ASLR) and core assignment move p50 itself, so the
    quantiles within a single run are not enough to judge the reproducibility of
    a measurement.
    """
    grouped = defaultdict(list)
    for row in rows:
        # The image is part of the key. If it were not, a different image with
        # the same resolution and marker count would land in the same group and
        # make the run-to-run spread look larger than it is.
        key = (row["machine"], row["route"], row["memory_mode"], row["image"],
               row["resolution"], row["markers"], row["fxfy"], row["aruco3"])
        if row["p50_ms"] is not None:
            grouped[key].append(row["p50_ms"])

    repeated = {key: values for key, values in grouped.items() if len(values) > 1}
    if not repeated:
        return

    print("\n=== Run-to-run spread (p50 when the same condition was run several times) ===")
    for key, values in sorted(repeated.items(), key=lambda item: str(item[0])):
        values = sorted(values)
        middle = values[len(values) // 2]
        spread = (values[-1] - values[0]) / middle * 100.0 if middle else 0.0
        print(f"{key[0]} {key[1]} {key[2]} {short_name(key[3])} {key[4]} "
              f"markers={key[5]} fxfy={key[6]} aruco3={key[7]}")
        print(f"  n={len(values)} median {middle:.3f} ms  range {values[0]:.3f} - {values[-1]:.3f} ms"
              f"  width {spread:.1f}%")


def print_machine_comparison(rows):
    """Show the ratio between machines when the same condition was measured on several.

    Beyond which one is faster, this makes it visible that the gap changes with
    the condition.
    """
    grouped = defaultdict(dict)
    for row in rows:
        key = (row["route"], row["resolution"], row["markers"], row["fxfy"])
        grouped[key][row["machine"]] = row

    comparable = {key: value for key, value in grouped.items() if len(value) > 1}
    if not comparable:
        return

    print("\n=== Machine comparison (same condition) ===")
    for key, machines in sorted(comparable.items(), key=lambda item: str(item[0])):
        print(f"{key[0]} {key[1]} markers={key[2]} fxfy={key[3]}")
        baseline_name, baseline = sorted(machines.items())[0]
        for name, row in sorted(machines.items()):
            if not row["p50_ms"] or not baseline["p50_ms"]:
                continue
            ratio = row["p50_ms"] / baseline["p50_ms"]
            note = "(baseline)" if name == baseline_name else f"{ratio:.2f}x baseline"
            print(f"  {name:<22} p50={row['p50_ms']:>8.3f} ms  "
                  f"fps={row['fps'] or 0:>7.1f}  {note}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="measurement JSONL")
    parser.add_argument("--format", choices=["table", "csv", "json"], default="table")
    arguments = parser.parse_args()

    environments, measurements = load_records(arguments.paths)
    if not measurements:
        raise SystemExit("no measurement records at all")

    rows = measurement_rows(measurements)

    if arguments.format == "json":
        json.dump({"environments": environments, "measurements": rows},
                  sys.stdout, ensure_ascii=False, indent=2)
        print()
        return
    if arguments.format == "csv":
        names = list(rows[0].keys())
        print(",".join(names))
        for row in rows:
            print(",".join("" if row[name] is None else str(row[name]) for name in names))
        return

    print("=== Environment ===")
    for environment in environments:
        print(format_environment(environment))
    print("\n=== Measurement results ===")
    print_table(rows)
    print_startup(rows)
    print_run_variance(rows)
    print_crossover(rows)
    print_machine_comparison(rows)


if __name__ == "__main__":
    main()
