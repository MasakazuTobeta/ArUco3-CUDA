# SPDX-License-Identifier: Apache-2.0
"""Verify existing video evidence without running a detector or collecting timing."""

import hashlib
import json
import math
from pathlib import Path


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    directory = Path(__file__).resolve().parent
    manifest = json.loads((directory / "validation.json").read_text())
    commands = json.loads((directory / "commands.json").read_text())
    for name, digest in manifest["evidence_sha256"].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest, name
    assert hashlib.sha256((directory / manifest["source_snapshot"]).read_bytes()).hexdigest() == (
        manifest["frozen_video_source_sha256"]
    )
    for case in commands["lifecycle_cases"]:
        assert case["observed_exit_code"] == case["expected_exit_code"]
        records = rows(directory / case["output"])
        if case["observed_exit_code"] == 0:
            summaries = [record for record in records if record["type"] == "video_measurement"]
            assert len(summaries) == (2 if case["name"] == "round2" else 4)
            assert all(record["validate_only"] for record in summaries)
            assert all(record["service_p50_ms"] is None for record in summaries)
            assert all(record["saturated_throughput_fps"] is None for record in summaries)
    flags = ["spatial_grouping", "sparse_pyramid", "filtered_contours",
             "cached_refinement", "exact_dictionary_lookup"]
    for case in commands["tuning_cases"]:
        records = rows(directory / case["output"])
        assert [flag for flag in flags if records[0][flag]] == case["expected_enabled_flags"]
        assert records[0]["binary_sha256"] == (
            "ed2cb02da873dacf1b9b90c357d14f0f2762e1540e8533528a030ed4e448d73c"
        )
    sanitizer = rows(directory / "asan-final.jsonl")
    assert sanitizer[0]["binary_sha256"] == manifest["sanitizer"]["binary_sha256"]
    assert len([record for record in sanitizer if record["type"] == "video_measurement"]) == 4
    smoke = rows(directory / "timed-smoke.jsonl")
    summaries = [record for record in smoke if record["type"] == "video_measurement"]
    assert len(summaries) == 4
    for summary in summaries:
        assert summary["timed_frames_checked"] == 12
        for field in ["service_p50_ms", "service_p95_ms", "service_p99_ms",
                      "saturated_span_ms", "saturated_throughput_fps"]:
            assert math.isfinite(summary[field]) and summary[field] > 0
        assert summary["service_p50_ms"] <= summary["service_p95_ms"] <= summary["service_p99_ms"]
        samples = [record for record in smoke if record["type"] == "video_frame"
                   and record["route"] == summary["route"]
                   and record["layout"] == summary["layout"]]
        assert [record["frame"] for record in samples] == list(range(12))
        assert all(0 <= record["worker"] < summary["workers"] for record in samples)
    print("12 CLI cases, 8 tuning mappings, sanitizer records and 48 smoke samples verified")


if __name__ == "__main__":
    main()
