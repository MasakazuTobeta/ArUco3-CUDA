#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""測定結果 JSONL を集計して表と crossover point を出力する。

目的:
    benchmark harness が出力した JSONL を読み、経路ごとの遅延と throughput を
    比較できる形へまとめる。有利な結果だけを選ばないよう、指定された全 file の
    全 measurement を対象とする。

使用方法:
    aggregate.py results.jsonl [more.jsonl ...] [--format table|csv|json]
"""
import argparse
import json
import sys
from collections import defaultdict


def load_records(paths):
    """JSONL を読み、環境情報と測定結果へ分ける。"""
    environments = []
    measurements = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise SystemExit(f"{path}:{line_number}: JSON として読めない: {error}")
                kind = record.get("type")
                if kind == "environment":
                    record["_source"] = path
                    environments.append(record)
                elif kind == "measurement":
                    record["_source"] = path
                    measurements.append(record)
                else:
                    raise SystemExit(f"{path}:{line_number}: 未知の type: {kind}")
    return environments, measurements


def format_environment(environment):
    gpu = environment.get("gpu_name") or "(GPU なし)"
    integrated = "統合" if environment.get("gpu_integrated") else "非統合"
    return (
        f"host={environment.get('hostname')} os={environment.get('os')}\n"
        f"  arch={environment.get('architecture')} cores={environment.get('cpu_online_cores')}"
        f" opencv={environment.get('opencv_version')} threads={environment.get('opencv_threads')}\n"
        f"  gpu={gpu} ({integrated}) cc={environment.get('gpu_compute_capability') or '-'}"
        f" driver={environment.get('driver_version') or '-'}"
        f" cuda={environment.get('cuda_toolkit') or '-'}\n"
        f"  power_mode={environment.get('power_mode') or '(未取得)'}"
    )


def measurement_rows(measurements):
    rows = []
    for record in measurements:
        end_to_end = record.get("end_to_end") or {}
        kernel = record.get("kernel")
        rows.append(
            {
                "route": record.get("route"),
                "memory_mode": record.get("memory_mode"),
                "resolution": f"{record.get('width_px')}x{record.get('height_px')}",
                "markers": record.get("detection_count"),
                "fxfy": (record.get("conditions") or {}).get("fxfy_effective"),
                "p50_ms": end_to_end.get("p50_ms"),
                "p95_ms": end_to_end.get("p95_ms"),
                "p99_ms": end_to_end.get("p99_ms"),
                "kernel_p50_ms": (kernel or {}).get("p50_ms") if kernel else None,
                "fps": record.get("throughput_fps"),
                "image": record.get("image_path"),
            }
        )
    return rows


def print_table(rows):
    header = [
        "route", "memory", "resolution", "markers", "fxfy",
        "p50_ms", "p95_ms", "p99_ms", "kernel_p50", "fps",
    ]
    widths = {name: len(name) for name in header}
    formatted = []
    for row in rows:
        values = {
            "route": str(row["route"]),
            "memory": str(row["memory_mode"]),
            "resolution": str(row["resolution"]),
            "markers": str(row["markers"]),
            "fxfy": f"{row['fxfy']:.4f}" if row["fxfy"] is not None else "-",
            "p50_ms": f"{row['p50_ms']:.3f}" if row["p50_ms"] is not None else "-",
            "p95_ms": f"{row['p95_ms']:.3f}" if row["p95_ms"] is not None else "-",
            "p99_ms": f"{row['p99_ms']:.3f}" if row["p99_ms"] is not None else "-",
            "kernel_p50": f"{row['kernel_p50_ms']:.3f}" if row["kernel_p50_ms"] is not None else "-",
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
    """同じ条件で複数経路がある場合に、経路間の比を示す。

    CPU が速い条件も必ず表に残す。有利な結果だけを選ばないため。
    """
    grouped = defaultdict(dict)
    for row in rows:
        key = (row["image"], row["resolution"], row["markers"], row["fxfy"])
        grouped[key][(row["route"], row["memory_mode"])] = row

    comparable = {key: value for key, value in grouped.items() if len(value) > 1}
    if not comparable:
        print("\n経路が 1 つのみのため crossover point は算出しない。")
        return

    print("\n=== 経路比較 (CPU の p50 を 1 とする比) ===")
    for key, routes in sorted(comparable.items(), key=lambda item: str(item[0])):
        cpu = next((row for (route, _), row in routes.items() if route == "CPU"), None)
        if cpu is None or not cpu["p50_ms"]:
            continue
        print(f"{key[1]} markers={key[2]} fxfy={key[3]}")
        for (route, memory), row in sorted(routes.items()):
            if not row["p50_ms"]:
                continue
            ratio = row["p50_ms"] / cpu["p50_ms"]
            verdict = "CPU が速い" if ratio > 1.0 else ("同等" if abs(ratio - 1.0) < 0.05 else "こちらが速い")
            print(f"  {route:<14} {memory:<12} p50={row['p50_ms']:.3f} ms  比={ratio:.3f}  {verdict}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="測定結果 JSONL")
    parser.add_argument("--format", choices=["table", "csv", "json"], default="table")
    arguments = parser.parse_args()

    environments, measurements = load_records(arguments.paths)
    if not measurements:
        raise SystemExit("measurement が 1 件も無い")

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

    print("=== 環境 ===")
    for environment in environments:
        print(format_environment(environment))
    print("\n=== 測定結果 ===")
    print_table(rows)
    print_crossover(rows)


if __name__ == "__main__":
    main()
