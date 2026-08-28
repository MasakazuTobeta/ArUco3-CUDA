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

# 読める schema 版。version が違う結果を黙って混ぜると、同じ key が違う
# 測定区間を指すことになる。version 3 で CPU 経路の測定区間から画像の
# 読み込みと checksum を外したため、2 以前の値とは比較できない。
SUPPORTED_SCHEMA_VERSIONS = {4}


def short_name(path):
    """表示用に画像 path の file 名だけを取り出す。"""
    if not path:
        return "(不明)"
    return path.rsplit("/", 1)[-1]


def machine_label(environment):
    """測定結果を機体で区別するための短い名前。

    複数機の結果を並べる場合、どの行がどの機体かが判別できないと比較にならない。
    基板名が取れる機種はそれを使い、取れない場合は GPU 名で代替する。
    """
    if environment is None:
        return "(不明)"
    model = environment.get("platform_model")
    if model:
        # "Jetson AGX Orin Developer Kit" -> "Jetson AGX Orin"
        return model.replace(" Developer Kit", "").strip()
    gpu = environment.get("gpu_name")
    if gpu:
        return gpu
    return environment.get("hostname") or "(不明)"


def load_records(paths):
    """JSONL を読み、環境情報と測定結果へ分ける。

    JSONL は 1 行目が環境情報、以降が測定結果である。測定結果には直前に
    現れた環境情報を対応付ける。1 つの file に複数の環境情報が現れる場合
    (複数機の結果を連結した場合) も、行の順序で正しく対応する。
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
                    raise SystemExit(f"{path}:{line_number}: JSON として読めない: {error}")
                version = record.get("schema_version")
                if version not in SUPPORTED_SCHEMA_VERSIONS:
                    raise SystemExit(
                        f"{path}:{line_number}: schema_version {version} は読めない。"
                        f"対応するのは {sorted(SUPPORTED_SCHEMA_VERSIONS)}。"
                        "測定し直すか、古い結果は別に集計すること")
                kind = record.get("type")
                if kind == "environment":
                    record["_source"] = path
                    current_environment = record
                    # 同一機体の環境行が繰り返し現れる場合は 1 つにまとめる。
                    key = (record.get("hostname"), record.get("gpu_name"),
                           record.get("cuda_toolkit"), record.get("power_mode"))
                    if key not in {(e.get("hostname"), e.get("gpu_name"),
                                    e.get("cuda_toolkit"), e.get("power_mode"))
                                   for e in environments}:
                        environments.append(record)
                elif kind == "measurement":
                    record["_source"] = path
                    if current_environment is None:
                        raise SystemExit(
                            f"{path}:{line_number}: 環境行より前に measurement がある")
                    record["_environment"] = current_environment
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
        f"  model={environment.get('platform_model') or '-'}"
        f" platform={environment.get('platform_release') or '-'}\n"
        f"  cpu={environment.get('cpu_topology') or '-'}"
        f" affinity={environment.get('cpu_affinity') or '-'}"
        f" aslr={environment.get('address_randomization') or '-'}\n"
        f"  cuda_context={environment.get('cuda_context_ms') or 0:.1f} ms (process ごとに 1 度)\n"
        f"  power_mode={environment.get('power_mode') or '(未取得)'}"
        f" clock={environment.get('gpu_max_clock_mhz') or '-'} MHz (max)"
        f" / {environment.get('gpu_current_clock_mhz') or '-'} MHz (現在)"
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
    """同じ条件で複数経路がある場合に、経路間の比を示す。

    CPU が速い条件も必ず表に残す。有利な結果だけを選ばないため。
    """
    grouped = defaultdict(dict)
    for row in rows:
        # 機体をまたいだ経路比較は意味が異なるため、機体を key へ含める。
        key = (row["machine"], row["image"], row["resolution"], row["markers"], row["fxfy"])
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
        print(f"{key[0]} {short_name(key[1])} {key[2]} markers={key[3]} fxfy={key[4]}")
        for (route, memory), row in sorted(routes.items()):
            if not row["p50_ms"]:
                continue
            ratio = row["p50_ms"] / cpu["p50_ms"]
            if row is cpu:
                print(f"  {route:<14} {memory:<12} p50={row['p50_ms']:.3f} ms  (基準)")
                continue
            # 5% 以内は測定のばらつきに埋もれるため、速い遅いを主張しない。
            # 判定の順序を誤ると、1.00 を僅かに超えただけで「CPU が速い」に
            # なり、同等の帯が片側にしか効かなくなる。
            if abs(ratio - 1.0) < 0.05:
                verdict = "同等"
            elif ratio > 1.0:
                verdict = "CPU が速い"
            else:
                verdict = "こちらが速い"
            print(f"  {route:<14} {memory:<12} p50={row['p50_ms']:.3f} ms  比={ratio:.3f}  {verdict}")


def print_startup(rows):
    """起動の費用を経路ごとに示す。

    warm-up 後の分位点には現れない。単発の検出や短い burst では、定常状態の
    差より起動の費用が支配する。定常との差から、元が取れる frame 数も出す。
    """
    grouped = defaultdict(dict)
    for row in rows:
        key = (row["machine"], row["image"], row["resolution"], row["markers"])
        entry = grouped[key].setdefault((row["route"], row["memory_mode"]), [])
        if row["first_result_ms"] is not None:
            entry.append(row)

    # CUDA の文脈生成は process ごとの費用であり、画像ごとには現れない。
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
                print("\n=== 起動の費用 (warm-up 後の分位点には現れない) ===")
                print("経路ごとに、1 枚目の結果が出るまでの時間と、定常との差で")
                print("元が取れる frame 数を示す。")
                printed = True
            gain = cpu["p50_ms"] - row["p50_ms"]
            # CUDA の文脈生成を CUDA 経路側にのみ加える。process ごとに 1 度
            # 発生し、画像ごとの測定には現れないためである。
            extra = (row["first_result_ms"] + context_ms) - cpu["first_result_ms"]
            print(f"{key[0]} {short_name(key[1])} {key[2]} markers={key[3]}")
            print(f"  CPU            1 枚目まで {cpu['first_result_ms']:8.3f} ms "
                  f"(検出 {cpu['first_frame_ms']:.3f} ms)  定常 {cpu['p50_ms']:.3f} ms")
            print(f"  {route:<14} 1 枚目まで {row['first_result_ms']:8.3f} ms "
                  f"(検出 {row['first_frame_ms']:.3f} ms)  定常 {row['p50_ms']:.3f} ms  "
                  f"[{memory}]")
            print(f"    + CUDA 文脈生成 {context_ms:.1f} ms (process ごとに 1 度)")
            if gain > 0:
                print(f"    起動の追加費用 {extra:.1f} ms / 1 frame あたりの利得 "
                      f"{gain:.3f} ms -> 約 {extra / gain:.0f} frame で相殺")
            else:
                print(f"    起動の追加費用 {extra:.1f} ms。定常でも速くならないため相殺しない")


def print_run_variance(rows):
    """同一条件を複数回実行した場合の、実行間ばらつきを示す。

    process ごとの memory 配置 (ASLR) や core 割り当てで p50 自体が動くため、
    1 回の実行内の分位点だけでは測定値の再現性を判断できない。
    """
    grouped = defaultdict(list)
    for row in rows:
        # 画像を key へ含める。同じ解像度でマーカー数も同じ別の画像があると、
        # 含めない場合に別条件の値が 1 つの group へ入り、実行間ばらつきを
        # 実際より大きく見せる。
        key = (row["machine"], row["route"], row["memory_mode"], row["image"],
               row["resolution"], row["markers"], row["fxfy"], row["aruco3"])
        if row["p50_ms"] is not None:
            grouped[key].append(row["p50_ms"])

    repeated = {key: values for key, values in grouped.items() if len(values) > 1}
    if not repeated:
        return

    print("\n=== 実行間ばらつき (同一条件を複数回実行した場合の p50) ===")
    for key, values in sorted(repeated.items(), key=lambda item: str(item[0])):
        values = sorted(values)
        middle = values[len(values) // 2]
        spread = (values[-1] - values[0]) / middle * 100.0 if middle else 0.0
        print(f"{key[0]} {key[1]} {key[2]} {short_name(key[3])} {key[4]} "
              f"markers={key[5]} fxfy={key[6]} aruco3={key[7]}")
        print(f"  n={len(values)} 中央 {middle:.3f} ms  範囲 {values[0]:.3f} - {values[-1]:.3f} ms"
              f"  幅 {spread:.1f}%")


def print_machine_comparison(rows):
    """同じ条件を複数機体で測った場合に、機体間の比を示す。

    どちらが速いかだけでなく、条件ごとに差が変わることを見えるようにする。
    """
    grouped = defaultdict(dict)
    for row in rows:
        key = (row["route"], row["resolution"], row["markers"], row["fxfy"])
        grouped[key][row["machine"]] = row

    comparable = {key: value for key, value in grouped.items() if len(value) > 1}
    if not comparable:
        return

    print("\n=== 機体比較 (同一条件) ===")
    for key, machines in sorted(comparable.items(), key=lambda item: str(item[0])):
        print(f"{key[0]} {key[1]} markers={key[2]} fxfy={key[3]}")
        baseline_name, baseline = sorted(machines.items())[0]
        for name, row in sorted(machines.items()):
            if not row["p50_ms"] or not baseline["p50_ms"]:
                continue
            ratio = row["p50_ms"] / baseline["p50_ms"]
            note = "(基準)" if name == baseline_name else f"基準比 {ratio:.2f}x"
            print(f"  {name:<22} p50={row['p50_ms']:>8.3f} ms  "
                  f"fps={row['fps'] or 0:>7.1f}  {note}")


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
    print_startup(rows)
    print_run_variance(rows)
    print_crossover(rows)
    print_machine_comparison(rows)


if __name__ == "__main__":
    main()
