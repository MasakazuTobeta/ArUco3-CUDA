# High-ID dictionary lookup corpus

## Purpose

Reproduce the dictionary-specific diagnostic used to evaluate the immutable exact-code
lookup table. IDs near the end of a large dictionary expose the cost of an ascending
Hamming scan. These fixtures complement the ordinary timing corpus; their lookup speedup
must not be presented as the average result for arbitrary marker IDs or images.

## Scope

`generate_high_id_corpus.cpp` is an original, standalone fixture generator. It calls public
OpenCV predefined-Dictionary rendering APIs and includes no external images or copied
implementation. The archive target is excluded from production builds and needs OpenCV
4.14.0, C++17 and the repository's SHA-256/JSON utilities. It executes no CUDA operations.

The dictionary is the actual OpenCV `DICT_6X6_1000`, not a generated custom dictionary.
The intended reference build is the repository's pinned OpenCV 4.14.0 commit
`0654a42e19215ef25b1d367d822f3c630447e7c7`. The manifest records the version and a hash of
actual OpenCV build information rather than assuming every 4.14.0 build is identical.

## Current state

The generator produces four grayscale PNGs with a 4x4 marker grid. IDs are 999 through 984
in row-major order. Marker borders are one cell, rotation is zero, the background is gray
235, and each marker is centered in its grid cell.

| File | Image dimensions | Marker side | Noise sigma |
| --- | --- | --- | --- |
| `highid_1280x720_clean.png` | 1280x720 | 128 px | 0 |
| `highid_1280x720_noise.png` | 1280x720 | 128 px | 18 gray levels |
| `highid_1920x1080_clean.png` | 1920x1080 | 180 px | 0 |
| `highid_3840x2160_clean.png` | 3840x2160 | 300 px | 0 |

The noisy image adds `cv::RNG::NORMAL` noise with seed 45168 in signed 16-bit storage,
then converts back to unsigned bytes with saturation. OpenCV is restricted to one thread
while generating. Every PNG is decoded again and checked byte for byte against rendered
pixels before publication.

`manifest.json` records generator source identity, compiler/OpenCV identity, dictionary
byte hash, all rendering settings, per-image PNG and pixel SHA-256 hashes, and marker IDs
and outer-corner pixel centers. `manifest.sha256` hashes the completed manifest. Timestamps
and output paths are omitted so unchanged builds produce identical manifests in different
directories.

### Reproduction

Run inside the repository's pinned development environment, from the repository root:

```bash
cmake -S cpu/experiments -B build/cpu-highid-generator \
  -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=/opt/opencv/lib/cmake/opencv4
cmake --build build/cpu-highid-generator --target aruco3cpu_generate_high_id_corpus -j2
build/cpu-highid-generator/aruco3cpu_generate_high_id_corpus \
  --output-dir data/cpu-high-id-1000
(cd data/cpu-high-id-1000 && sha256sum -c manifest.sha256)
```

The destination must be new or empty. A nonempty directory, regular file, or directory
symlink is refused. Repeating the command preserves every existing byte and returns an
error; choose a new destination for another generation. Files are first written and
verified in a unique sibling staging directory, then published together through a
same-filesystem rename. Failed generation removes that staging directory. Concurrent
producers cannot replace a completed output.

The archive target embeds the source SHA-256 at configure time and reconfigures when the
source changes. Verify all PNGs listed by the manifest with standard Python:

```bash
python3 - data/cpu-high-id-1000 <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
manifest = json.loads((root / "manifest.json").read_text())
for scene in manifest["scenes"]:
    actual = hashlib.sha256((root / scene["file"]).read_bytes()).hexdigest()
    assert actual == scene["png_sha256"], scene["file"]
print("All four PNG hashes match.")
PY
```

Use the matching dictionary for a paired detection-only diagnostic. Build the CPU benchmark
as documented in the [CPU experiment design](../../docs/design/cpu-optimization.md).

```bash
taskset -c 5 build/cpu-opt/cpu/aruco3cuda_cpu_benchmark \
  --input-dir data/cpu-high-id-1000 --dictionary DICT_6X6_1000 \
  --variants opencv,previous,lookup --threads 1 --warmup 10 --iterations 40 \
  --throughput-frames 0
```

CPU 5 is a performance core on the measured DGX Spark; use the appropriate core on other
hardware. Add `--profile` only for stage diagnosis. `previous` is the earlier experimental
CPU route, while `opencv` is the official baseline. The benchmark checks each result
against actual OpenCV before accepting its timing.

### Recorded identity

The archived generator reproduced all four original diagnostic PNG files byte for byte
on the measured aarch64 environment (GCC 13.3.0 and the pinned OpenCV build). The source
SHA-256 was `bceb95ec630ed9bf3246baf9487d8c08c611e297b69e3c1741b1378f52cddf3d`.
The generated manifest SHA-256 was
`97c359ee827d10e035642592fdafebf096025ddcb285abbdc2578b3a8a9b082a`.
The original local copy is the diagnostic manifest (local `../../docs/measurements/2026-11-17-cpu-steady-diagnostics/high-id-corpus-manifest.json`).

CLI validation covered malformed arguments, nonempty/file/symlink destinations, an
unwritable parent, new and existing-empty directories, trailing separators, repeated
invocation, and two concurrent producers. Exactly one concurrent producer succeeded;
manifest/image hashes matched, existing sentinel files were unchanged, and no staging
directories remained.

## Goals

- Keep focused high-ID results reproducible and separate from representative aggregate results.
- Tie timing rows to exact input files and the generator/library identities.
- Reject accidental overwrites and incomplete corpus publication.

## Open questions

- Noise pixels and PNG encoding may differ with another OpenCV/compiler/architecture build.
  Record that build's manifest rather than assuming the recorded hashes will carry over.
- These are synthetic, fully visible markers. They do not establish real-image recall,
  precision, occlusion robustness or the speedup for typical marker-ID distributions.
