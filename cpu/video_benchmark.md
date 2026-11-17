# Independent-frame video scheduling experiment

## Purpose

Compare persistent frame workers with parallel processing inside one frame under
the same CPU budget. This is a steady-state video scheduling experiment, not a
tracking implementation or a change to the detector's default latency policy.

## Scope

`aruco3cuda_cpu_video` runs both the actual OpenCV `ArucoDetector` and this
repository's `CpuDetector` on the same changing-frame sequence. Each worker owns
one detector for its lifetime. Every frame reruns detection and preprocessing.
Storage capacity and precomputations that do not depend on frame pixels persist.
No previous result, pyramid pixel, candidate, or marker position is used to skip
work in the next frame.

Images are decoded to grayscale and hashed before any measurement. Inputs are
shared read-only. The benchmark does not include image decoding, acquisition,
camera arrival times, or a real-time output queue. Startup and detector/thread
creation are excluded. CUDA kernel time is not applicable.

## Current state

The command has two scheduling layouts. With `--workers W --opencv-threads T`:

- `intra` uses one persistent worker and requests `W*T` OpenCV threads.
- `workers` uses `W` persistent detectors and requests `T` OpenCV threads.

The command requires exactly `W*T` CPUs in its affinity mask, bounded to 64. Use
`taskset` or a comma-separated `--cpu-list` to select the same physical cores for both layouts.
OpenCV's thread setting is process-wide; it is set once before launching a pool,
never concurrently by workers. Its internal pool is shared rather than a private
pool per detector. Requested and actual OpenCV thread counts are both recorded.

All selected inputs are prepared on every worker before measurement, followed by
the configured warmup calls. The pool persists through the validation and timed
phases. A shared atomic cursor assigns independent frame indices dynamically;
each index has exactly one writer. Results and samples stay in indexed slots,
so completion order does not change the output's frame order.

### Inputs and reproducibility

By default, sorted files matching `cpu_1280x720_` under `data/cpu-experiment`
provide six clean/noisy scenes with 1, 4 and 16 markers. Frame `i` uses input
`i % input_count`; the cycle repeats, but detection is recomputed on every call.
At least two different decoded frames must occur in the measured sequence.

`--input FILE` can be repeated to specify a sequence in explicit argument order.
Otherwise `--input-dir DIR --filter TEXT` selects sorted PNG/JPEG/PGM filenames
containing the substring. File and decoded-pixel hashes, dimensions, the complete
input list and an indexed sequence hash are emitted. Both routes/layouts use the
same sequence. Worker assignment can differ because dispatch is dynamic.

The fixed dictionary is `DICT_ARUCO_MIP_36h12`. `--aruco3 1` uses canonical minimum
side 32 and original-image ratio 0.05 with subpixel refinement; `--aruco3 0`
disables both ArUco3 and refinement. Actual configuration values, threshold
windows, error correction and CPU tuning switches are written to the environment
record, along with the executable hash and compiler/OpenCV versions. Host power,
clock state, build flags and source revision should be recorded by the enclosing
experiment runner, as for the ordinary CPU benchmark.

### Measurement and validation

Service latency is the time of one detector call producing host IDs and corners,
with timestamp overhead. It includes CPU contention while that call runs, but
excludes time waiting to receive a frame index. Nearest-rank p50/p95/p99 are
computed across all frames in the common mixed sequence. `--save-frames` retains
each indexed sample and input/worker assignment for further per-scene analysis.

Saturated throughput is `frame_count / (last_completion - first_service_start)`.
All work is available to the prepared pool. The span includes dispatch gaps and
the finite batch's final load imbalance; it excludes setup, thread teardown,
validation, printing and image loading. The numerator counts every frame once.
Do not infer it from reciprocal median latency or sum the service times.

The indexed records are written after the batch finishes. The benchmark does not
measure camera-to-result latency, queue waiting, or head-of-line delay from an
ordered real-time consumer. More workers can improve throughput while increasing
service latency; this is an optional scheduling tradeoff, not a default latency
optimization. No performance benefit is assumed without measurements.

Every frame is compared with an independently executed OpenCV reference outside
timing. Marker lists are canonicalized by ID and first-corner position, while
each marker's four ordered corners remain unchanged. IDs, corner coordinates and
rejected counts must match exactly; a discrepancy aborts the run.

Timed OpenCV calls request IDs and corners without materializing rejected-corner
coordinates, matching the ordinary CPU benchmark's output scope. Before and after
the timed phase, the same persistent workers rerun every frame with rejected
output enabled and validate its count. Timed IDs/corners are checked after the
clock interval; `CpuDetector`'s timed rejected counts are also checked because its
result already contains them. OpenCV's untimed counts are reported explicitly as
validated counts, not fabricated timed outputs. The oracle also runs every frame;
it does not supply detection results to a worker.

### Commands

Build the ordinary standalone CPU project; the video target links the host-only
detector, utility library and standard threads:

```bash
cmake -S cpu/standalone -B build/cpu-opt -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR=/opt/opencv/lib/cmake/opencv4 -DARUCO3CUDA_BUILD_TESTS=OFF
cmake --build build/cpu-opt --target aruco3cuda_cpu_video -j2
```

Run an untimed correctness check on four selected CPUs:

```bash
taskset -c 0-3 build/cpu-opt/cpu/aruco3cuda_cpu_video \
  --frames 12 --workers 4 --opencv-threads 1 --warmup 0 \
  --route both --layout both --validate-only --save-frames
```

Compare layouts on the same four performance cores, after generating the input
corpus using the [ordinary CPU benchmark](cpu_detector.md):

```bash
taskset -c 5-8 build/cpu-opt/cpu/aruco3cuda_cpu_video \
  --input-dir data/cpu-experiment --filter cpu_1280x720_ \
  --frames 240 --workers 4 --opencv-threads 1 --warmup 18 \
  --route both --layout both --tuning default --save-frames
```

Repeat in independent processes and alternate `--reverse-order`. It reverses both
route and layout order without changing the input sequence or settings. For one
route/layout, use `--route opencv|cpu` and `--layout intra|workers`. CPU tuning can
select `default`, `previous`, `control`, `spatial`, `sparse`, `contours`, `cached`,
`lookup`, or `round2`; it has no effect on the OpenCV implementation. `default`
keeps current `CpuOptions` defaults. `previous` explicitly enables the earlier six
changes (workspace/pyramid reuse, packed matching, fused cells, shared thresholds,
parallel windows and parallel decode), with all later changes disabled. The five
named later changes each start from that same previous configuration and enable
only their own switch; `round2` enables all five together. The control disables
both sets of switches. Every tuning choice remains subject to the same quality
checks.

### Independent-process trials and metadata

The following self-contained wrapper runs exactly three processes, reverses
route/layout order on the second process, and preserves raw frame records. It
refuses to reuse an existing destination. Execute it from the repository root in
the build environment, after reserving the selected CPUs for measurement. Change
the CPU list and destination for another machine or experiment.

```bash
python3 - <<'PY'
import datetime, hashlib, json, pathlib, platform, subprocess

binary = pathlib.Path("build/cpu-opt/cpu/aruco3cuda_cpu_video").resolve()
destination = pathlib.Path("results/cpu-video-steady")
destination.mkdir(parents=True, exist_ok=False)
cpus = [5, 6, 7, 8]
command = ["taskset", "-c", ",".join(map(str, cpus)), str(binary),
           "--input-dir", "data/cpu-experiment", "--filter", "cpu_1280x720_",
           "--frames", "240", "--workers", "4", "--opencv-threads", "1",
           "--warmup", "18", "--route", "both", "--layout", "both",
           "--tuning", "default", "--save-frames"]

def read(path):
    try:
        return pathlib.Path(path).read_text()
    except OSError:
        return None

source_paths = subprocess.check_output(
    ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
    text=True).splitlines()
metadata = {
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "command": command,
    "order": ["forward", "reverse", "forward"],
    "host": platform.platform(),
    "cpuinfo": read("/proc/cpuinfo"),
    "aslr": read("/proc/sys/kernel/randomize_va_space"),
    "load_at_start": read("/proc/loadavg"),
    "cmake_cache": read("build/cpu-opt/CMakeCache.txt"),
    "compile_commands": read("build/cpu-opt/compile_commands.json"),
    "opencv_provenance": read("/opt/opencv/share/aruco3cuda/opencv-provenance.json"),
    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
    "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
    "source_sha256": {
        path: hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
        for path in source_paths if pathlib.Path(path).is_file()
        and pathlib.Path(path).suffix in {".cpp", ".hpp", ".py", ".txt"}},
    "cpu_policy": {
        str(cpu): {name: read(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/{name}")
                   for name in ("scaling_governor", "scaling_min_freq",
                                "scaling_max_freq", "scaling_cur_freq")}
        for cpu in cpus},
    "scope": "steady-state detector service and finite saturated-batch throughput",
    "host_policy": "Retained; this wrapper changes no clock or power settings"
}
(destination / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
for trial in range(1, 4):
    assert hashlib.sha256(binary.read_bytes()).hexdigest() == metadata["binary_sha256"]
    assert all(hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest() == digest
               for path, digest in metadata["source_sha256"].items())
    trial_command = command + (["--reverse-order"] if trial % 2 == 0 else [])
    with (destination / f"run-{trial}.jsonl").open("x") as output, \
         (destination / f"run-{trial}.stderr").open("x") as errors:
        subprocess.run(trial_command, stdout=output, stderr=errors, check=True)
PY
```

Check that all runs have the same sequence hash, affinity, settings and source
fingerprints before aggregating. Summarize service percentiles and throughput
separately by route/layout using the median of the three process-level values;
do not pool latency samples from different settings or substitute a frame's
queue time for service latency. Per-frame records remain available to compare
the same scene class across layouts.

### Failure handling

The preparation and phase barriers observe a cancellation flag. The first worker
exception is propagated to the caller after all workers are stopped/joined.
Thread creation errors after a partial pool also cancel and join already-created
threads. Cancellation is changed under the condition variable's mutex so a
waiting worker cannot miss shutdown notification.

For lifecycle regression checks, `--test-worker-failure setup|frame|spawn` is
accepted only with `--validate-only`. Each intentionally fails with a nonzero
exit status; `spawn` requires worker layout and at least two workers. An external
timeout can detect a regression that leaves a barrier waiting:

```bash
timeout 10s taskset -c 0-1 build/cpu-opt/cpu/aruco3cuda_cpu_video \
  --workers 2 --frames 6 --warmup 0 --layout workers --route cpu \
  --validate-only --test-worker-failure setup
```

Replace `setup` with `frame` or `spawn` to exercise the other failure paths.
These commands are expected to exit with an error, not produce a valid
measurement. `--validate-only` emits null timing fields and performs no service
or throughput measurement.

### Implementation verification

The new CLI passed twelve focused correctness/lifecycle checks: both layouts and
routes, reverse ordering, ArUco3 disabled, round2 tuning, changing dimensions,
setup/frame/partial-spawn failures, invalid worker counts, empty input selection,
duplicate decoded inputs and invalid CPU affinity/budget. A focused ASan/UBSan
build of the CLI also passed both routes/layouts with leak detection enabled;
that build linked the current Release detector library and is not a claim that
the detector was reinstrumented by this check.

A twelve-frame timed smoke test on CPUs 6-8 with three workers versus one
three-thread worker exercised all four route/layout combinations. All 48 indexed
timed results passed exact validation, and service/throughput fields were finite
and positive. This short run validates the measurement path; it is not evidence
of a steady-state scheduling speedup.

### Provenance

This is repository-authored scheduling and measurement code under Apache-2.0,
based on the local `build/cpu-frame-probe.cpp` experiment. It calls public OpenCV
APIs and the repository's CPU detector. No external implementation or GPLv3 code
was copied. OpenCV and the detector retain the provenance documented in
[the CPU design](cpu_detector.md) and [the code provenance record](../docs/code-provenance.md).

## Goals

- Compare worker scheduling and intra-frame parallelism on identical CPU budgets
  and sequences, retaining both service tails and throughput.
- Keep every frame's detection quality checked independently of timing.
- Preserve raw frame order and propagate failures without a stuck worker pool.

## Open questions

- Which input mixtures and CPU architectures benefit from independent workers.
- How service tails translate to application deadlines under a real arrival
  process, output ordering and queue policy; none is modeled here.
- How much finite-batch tail imbalance affects very short runs.
