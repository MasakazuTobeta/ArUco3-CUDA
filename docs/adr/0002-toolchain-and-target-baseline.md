# ADR-0002: Fix the build toolchain and target environment baseline

- Status: Accepted
- Date: 2026-08-27

## Purpose

Before implementation starts, fix the CUDA architectures, the minimum toolchain versions, and the OpenCV version used as the CPU baseline, so that the work items in the [implementation plan](../implementation-plan.md) proceed on the same assumptions.

## Scope

This covers CMake, the CUDA Toolkit, the host compiler, the target CUDA architectures, and the OpenCV version of the CPU baseline implementation. Policy on adding dependent libraries is out of scope.

## Background

### Measured values on the development machine

These values were measured on the development machine on 2026-08-27.

| Item | Value |
| --- | --- |
| GPU | NVIDIA GB10 |
| Compute Capability | 12.1 |
| SM count | 48 |
| L2 cache | 24576 KB |
| `integrated` | 1 |
| `managedMemory` / `concurrentManagedAccess` | 1 / 1 |
| CUDA Toolkit | 13.0 (V13.0.88) |
| Driver | 580.95.05 |
| host compiler | gcc 13.3.0 |
| CMake | 3.28.3 |
| OS / arch | Ubuntu 24.04.3 LTS / aarch64 |
| logical CPU core count | 20 |
| system memory | 119 GB |

`nvcc --list-gpu-arch` lists `compute_75` through `compute_121`, including both `compute_87` and `compute_121`.

### Discrepancy with the documentation

The [README](../../README.md) and the [architecture](../architecture.md) document describe DGX Spark GB10 as Compute Capability 12.0, `sm_120`. The actual machine reports 12.1, so those descriptions need to be corrected.

### Measured values on the Jetson Orin machine

These values were measured on the actual machine on 2026-08-27.

| Item | Value |
| --- | --- |
| board | Jetson AGX Orin Developer Kit |
| L4T | R35.4.1 (JetPack 5.1.2) |
| OS / glibc | Ubuntu 20.04.6 LTS / 2.31 |
| CUDA Toolkit | 11.4 (V11.4.315) |
| maximum architecture supported by nvcc | `compute_87` |
| host compiler | gcc 9.4.0 |
| power mode | MAXN (0) |
| maximum GPU clock | 1300 MHz |
| logical CPU core count | 12 |
| system memory | 61 GB |
| `nvidia-smi` | not present |

The nvcc in CUDA 11.4 supports only up to `compute_87`. The default `CMAKE_CUDA_ARCHITECTURES` value `87;121` cannot be used on Jetson, so the `jetson-orin` preset is used there.

### Prerequisites not installed

OpenCV, `clang-format`, and `ninja` are not installed on the development machine. `compute-sanitizer` is present in `/usr/local/cuda/bin`.

### OpenCV release status

For OpenCV, the latest in the 4.x line is 4.14.0, and 5.0.0 has been released as a separate line. The open question from [ADR-0001](0001-independent-implementation.md), "whether to target OpenCV 4.x or 5.x first," remains undecided.

## Decision

1. Target the CUDA architectures `sm_87`, `sm_120`, and `sm_121`. Specify `87;120;121` as the default for `CMAKE_CUDA_ARCHITECTURES`, overridable as needed. `sm_120` was added on 2026-08-28. See [the 2026-08-28 update](#2026-08-28-update-adding-sm_120-to-the-target-architectures) for the background. **`sm_110` was added on 2026-08-31**; see [the 2026-08-31 update](#2026-08-31-update-adding-sm_110-and-the-jetson-agx-thor).
2. Target only the Compute Capabilities that the target machines report. Do not rely on JIT.
3. Use C++17 as the C++ standard, and C++17 for the CUDA language standard as well.
4. Set the minimum CMake version to 3.24. It is the first version in which the handling of `CUDA_ARCHITECTURES` is stable, and both the development machine and the JetPack 6 line satisfy it.
5. Fix the CPU baseline implementation to OpenCV 4.14.0. Record the tag and build options in a script and embed them in the measurement results.
6. Target OpenCV 4.x first, and defer following 5.x to a later decision.
7. The generated artifacts may use exceptions on the host side, but the core's public API returns `Status`. Details follow the [public API draft](../design/public-api.md).
8. Pin the toolchain in a development container rather than by installing directly on the host. Pin the CUDA Toolkit in the image as well, and install only the necessary packages. Keep the mode that bind-mounts from the host for local experimentation, but do not use it for runs that involve measurement. Details follow the [Docker environment design](../design/docker-environment.md).
9. Set the minimum CUDA Toolkit version to 11.4. We confirmed on the actual machine that the target Jetson AGX Orin runs JetPack 5.1.2 (L4T R35.4.1) and that the bundled CUDA is 11.4. Keep the common path within what this version can compile.
10. Use `nvcr.io/nvidia/l4t-cuda:11.4.19-devel` as the base image for Jetson's pinned mode. It contains only CUDA and is smaller than `l4t-jetpack`. **This was changed on 2026-08-31**; see [the 2026-08-31 update](#2026-08-31-update-building-the-jetson-image-from-ubuntu-and-the-l4t-apt-repository).

## Rationale

- Matching the Compute Capability that the actual machine reports removes the need to wait on JIT and to separate out machine-to-machine differences.
- Targeting only the Compute Capabilities that the target machines actually report expresses, in the build configuration, the [architecture](../architecture.md) policy of "a common algorithm with machine-specific optimizations kept separate."
- OpenCV 4.14.0 is the latest in the 4.x line that includes `useAruco3Detection`, and it includes the `predefined_dictionaries.hpp` that the [Dictionary policy](../dictionaries.md) treats as authoritative.
- Fixing the minimum CUDA Toolkit version to the development machine's 13.0 would make it depend on the JetPack on the Jetson side, so it is not decided here.

## Consequences

### Benefits

- No discussion of build settings arises when implementation starts.
- It becomes easier to isolate cases where environmental differences leak into the measurement results.

### Drawbacks

- The minimum version cannot be fixed until the CUDA Toolkit version on the Jetson Orin side is confirmed.
- It closes off, for now, the option of targeting OpenCV 5.x first.

## Open questions

- When to update Jetson Orin to JetPack 6.x. Updating would bring CUDA 12.6 and narrow the gap with DGX Spark's CUDA 13.0.
- Whether to cross-compile for Jetson Orin or build on the machine itself.
- Whether to align the `clang-format` style settings with OpenCV or define our own. As an interim measure, anticipating a contribution to OpenCV, we have placed a `.clang-format` based on Google style with indent 4 and column limit 100. This ADR will be updated once decided.

## 2026-08-28 update: adding sm_120 to the target architectures

### Background

We added a third machine to the evaluation targets: a ZOTAC GAMING GeForce RTX 5070 Ti 16GB (GB203) installed in an x86_64 workstation.

Both existing machines have integrated GPUs, where host and device share the same physical memory. The [evaluation plan](../evaluation-plan.md) and the [benchmark report](../benchmark-report.md) explicitly state as a limitation that "these are results for integrated GPUs and cannot be generalized to discrete GPUs," and we had no machine with which to close that limitation by measurement. Because transfer costs matter, differences between memory kinds (`M-Pageable`, `M-Pinned`, `M-Managed`, `M-Device`) also become meaningful for the first time on this machine.

| Item | Value |
| --- | --- |
| GPU | ZOTAC GAMING GeForce RTX 5070 Ti 16GB GDDR7 (GB203) |
| device memory | 16303 MiB |
| Compute Capability | 12.0 (confirmed on the machine) |
| CPU | Intel Core Ultra 7 265 (20 core) |
| system memory | 62 GB |
| OS / arch | Ubuntu 22.04.5 LTS / x86_64 |
| CUDA Toolkit (host) | 13.2 |
| NVIDIA driver | 610.43.02 (`nvidia-driver-610-open`) |

### Change to the decision

We widen the target CUDA architectures from the two `sm_87` and `sm_121` to the three `sm_87`, `sm_120`, and `sm_121`. The `portability` preset default becomes `87;120;121`, and an `rtx-blackwell` preset is added.

We had originally decided not to make `sm_120` a default target. The reason was that, because GB10 reports 12.1, a binary built for `sm_120` would rely on JIT. That reason still holds for GB10; what changed is the premise, now that GB203 — a real machine reporting 12.0 — has been added. The direction of the decision has not changed. This is simply the result of applying the same principle, "target only the Compute Capabilities that the target machines report."

### Container configuration

The base image and CUDA packages for the `rtx-blackwell` profile are identical to `dgx-spark` (ubuntu:24.04, CUDA 13.0). With the container side aligned, the difference in measurements between the two is only the hardware difference. The fact that the host's CUDA is 13.2 has no effect in pinned mode.

### Confirming the Compute Capability

In the first version of this ADR we described DGX Spark GB10 as 12.0 and had to correct it when the actual machine reported 12.1. To avoid the same mistake, we settled this from measured values rather than inferring from the product specifications.

```
$ nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv
name, compute_cap, memory.total [MiB], driver_version
NVIDIA GeForce RTX 5070 Ti, 12.0, 16303 MiB, 610.43.02
```

It was 12.0, as expected. We target `sm_120`.

### Handling Secure Boot

Secure Boot is enabled on the target machine. A kernel module built with DKMS is unsigned and will not load after a reboot. Using the prebuilt module signed by Canonical (`linux-modules-nvidia-610-open-generic-hwe-22.04`) allows installation without MOK enrollment.

```
$ modinfo nvidia | grep signer
signer:         Canonical Ltd. Kernel Module Signing
```

This package installs the module for the latest HWE kernel rather than the running kernel, so after installation you must reboot and start on the new kernel.

## 2026-08-31 update: building the Jetson image from ubuntu and the L4T apt repository

### Background

Item 10 made Jetson the only profile that pulls its base image from NGC. The other
two start from `ubuntu:24.04` and install the CUDA packages this project uses from
NVIDIA's apt repository. The asymmetry was accepted because the
[Docker environment design](../design/docker-environment.md) recorded that on
Jetson there was no way to select individual packages from an apt repository.

That was wrong. The CUDA build for Tegra is published at
`repo.download.nvidia.com/jetson`, a different host from the
`developer.download.nvidia.com` the other profiles use, and every package this
project needs is there individually. Checked on the machine:

| Package | Version in `jetson/common r35.4` | Installed size |
| --- | --- | --- |
| `cuda-nvcc-11-4` | 11.4.315-1 | 98.1 MB |
| `cuda-nvdisasm-11-4` | 11.4.298-1 | 31.8 MB |
| `cuda-sanitizer-11-4` | 11.4.298-1 | 27.6 MB |
| `cuda-cccl-11-4` | 11.4.298-1 | 12.1 MB |
| `cuda-cudart-dev-11-4` | 11.4.298-1 | 5.0 MB |
| `cuda-nvtx-11-4`, `cuda-cuobjdump-11-4`, `cuda-profiler-api-11-4`, and dependencies | | 1.8 MB |
| **Total** | | **176.4 MB** |

Three things had to hold before this was worth doing, and all three were checked on
the machine.

- **The toolkit is the same one.** `nvcc` from the apt package and `nvcc` in
  `l4t-cuda:11.4.19-devel` are both `V11.4.315`, build
  `cuda_11.4.r11.4/compiler.31964100_0`, and `libcudart` is `11.4.298` on both
  sides. This is a packaging change, not a version change, so the benchmark
  results stay comparable.
- **The driver is not dragged in.** The dependency closure is
  `cuda-nvcc-11-4` → `cuda-cudart-dev-11-4` → `cuda-cudart-11-4`,
  `cuda-cccl-11-4`, `cuda-driver-dev-11-4`, and from there only
  `cuda-toolkit-*-config-common`. `nvidia-l4t-cuda`, the L4T driver runtime, never
  appears. The driver keeps coming from the NVIDIA Container Toolkit, which
  injects 219 libraries and 37 symlinks listed in the host's `l4t.csv`, including
  `libcuda.so.1`.
- **`ubuntu:20.04` is what was already underneath.** `l4t-cuda:11.4.19-devel` is
  built on Ubuntu 20.04.6, so `install-toolchain.sh` and `build-opencv.sh` were
  already running on that userspace, including the path that fetches CMake from
  Kitware because 20.04 ships 3.16.

### Change to the decision

Item 10 is replaced. The Jetson profile starts from `ubuntu:20.04` and installs
the CUDA packages from `repo.download.nvidia.com/jetson/common`, selected by
`ARUCO3_CUDA_REPO_FLAVOR=jetson`. The suite comes from `ARUCO3_JETSON_L4T_SUITE`
and defaults to `r35.4`.

`install-cuda-toolkit.sh` gains the flavour switch because the two repositories
differ in more than a URL: the CUDA repository ships a `cuda-keyring` package,
while the L4T repository publishes an armored key that has to be dearmored into
its own keyring and bound to the source with `signed-by`. Only the `common`
component is added; `t234` holds the board support package and the L4T driver,
which must not end up in this image.

### Re-measurement

The claim that this changes the packaging and not the compiler was checked by
running the sweep again on the new image and comparing it against
[2026-08-29](../measurements/2026-08-29-jetson-orin-sweep.jsonl). The result is
[2026-08-31](../measurements/2026-08-31-jetson-orin-sweep-after-image-change.jsonl),
taken with the procedure in the [benchmark report](../benchmark-report.md): the
same 28 scenes, 3 routes, 3 independent runs, 30 warm-up and 200 measured
iterations, pinned to CPU 0, MAXN. All 28 corpus images hash identically to the
ones measured in August, so the inputs are the same bytes and not merely the same
generator settings.

The environment the sweep ran in is recorded separately, in
[2026-08-31-jetson-orin-environment.json](../measurements/2026-08-31-jetson-orin-environment.json),
captured from the same image (`sha256:99e3b92dba23e8050859ac55c49c7588d809e4438fe62aeff9fe3b79045655b5`).
It carries the CUDA provenance, so the repository and all eight package versions
are on record with the numbers. The sweep file itself cannot show that: the
benchmark harness records `cuda_toolkit` only to the minor version, and the
`l4t-cuda` image it replaced reports the same `11.4` on the same Ubuntu 20.04, so
the two are indistinguishable from the sweep file alone.

The two records are complementary rather than redundant. The sweep file has the
GPU identity, which the environment record leaves empty on Jetson because
`record-environment.sh` reads it from `nvidia-smi` and L4T has none.

Ratios are new median-of-three p50 over old median-of-three p50.

| Route | Scenes | Minimum | Median | Maximum |
| --- | --- | --- | --- | --- |
| CPU | 28 | 0.995 | 1.000 | 1.006 |
| Hybrid | 28 | 0.958 | 0.997 | 1.081 |
| CUDA-Resident | 28 | 0.994 | 1.001 | 1.375 |
| All | 84 | 0.958 | **1.000** | 1.375 |

The CPU route is the control. It runs OpenCV and never touches anything `nvcc`
compiled, so its spread of 0.5% is the measurement noise floor for this machine
on this day, and it says the two sessions are comparable at all.

The one ratio far from 1 is `CUDA-Resident` on `clean_640x480_n0_s16.png`, the
smallest and emptiest scene, at 0.483 ms against 0.665 ms. It is not a
regression. The scene is bimodal on this machine, and the August data already
contains both modes:

| | run 1 | run 2 | run 3 |
| --- | --- | --- | --- |
| 2026-08-29 p50 | 0.483 | 0.664 | 0.480 |
| 2026-08-31 p50 | 0.626 | 0.669 | 0.665 |
| `min_ms`, both dates | 0.470 to 0.473 across all six runs | | |

The fastest iteration is the same to three decimal places in every run on both
dates. What differs is how often a run settles into the slower mode, and the
median of three picked the fast mode in August and the slow mode now. This is the
same instability the benchmark report already records for the Jetson GPU stage.

### Consequences

- No profile depends on NGC any more. All three are `ubuntu` plus an NVIDIA apt
  repository.
- The recorded provenance names the packages and their exact versions. Under the
  NGC base image it recorded `"packages": []`, so the image tag was the only
  evidence of which toolkit produced a measurement.
- **The coupling to the L4T version remains.** The suite is pinned to `r35.4`, so
  a move to JetPack 6 changes the suite, the package versions, and the base image
  to `ubuntu:22.04`. That is the same coupling the base image tag had; it is not
  loosened, only moved.
- `mounted` mode is unaffected. It was measured at 540 MB against the 5.0 GB of
  the NGC image, and it was considered as the way to drop NGC. It was rejected
  because the image then stops being self-contained: reproducing a measurement
  later, or on a machine whose host toolkit has moved, needs the toolkit to
  travel with the image.

## 2026-08-31 update: adding sm_110 and the Jetson AGX Thor

### Background

A fourth machine joins the evaluation targets: a Jetson AGX Thor Developer Kit. It is
the second Tegra machine and shares nothing with the first beyond that: L4T r38.4
rather than r35.4, so Ubuntu 24.04 rather than 20.04, CUDA 13.0 rather than 11.4, and
Compute Capability 11.0 rather than 8.7.

| Item | Value |
| --- | --- |
| GPU | NVIDIA Thor (integrated) |
| Compute Capability | 11.0 (confirmed on the machine) |
| CPU / system memory | 14 cores / 122.8 GB |
| OS / arch | Ubuntu 24.04.3 LTS / aarch64, kernel 6.8.12-tegra |
| L4T | R38.4.0 |
| NVIDIA driver | 580.00 |
| CUDA Toolkit on the host | none installed |

There is no CUDA Toolkit on this host, so `mounted` mode is unavailable here and the
profile has to install the packages. The apt path added on 2026-08-31 for the Orin
covers it unchanged: the same `flavour=jetson` repository serves r38.4, and the
package names are the `13-0` set the DGX Spark and RTX Blackwell profiles already use.

### Change to the decision

The target CUDA architectures widen from three to four: `sm_87`, `sm_110`, `sm_120`,
`sm_121`. The `portability` preset default becomes `87;110;120;121`, a `jetson-thor`
preset is added, and CI compiles for all four.

`compute_110` is supported by both toolkits in play. Checked on the machines: the
L4T r38.4 nvcc 13.0.48 lists it and emits an `sm_110` cubin, and the nvcc 13.0.88 the
other machines use lists it too, so widening the preset does not break them. The
`portability` build on the DGX Spark produces cubins for all four architectures.

### Consequences

- 518 tests pass on the Thor, with no build warnings, and the four Compute Sanitizer
  tools pass.
- `portability` still cannot be built on the Jetson AGX Orin, and this does not change
  that. CUDA 11.4 stops at `sm_87` and rejects `compute_120` at configure time, as
  recorded in the [Docker environment design](../design/docker-environment.md). Adding
  `sm_110` neither helps nor worsens it.
- The container user matters on this host. Its login user is uid 2002, while the
  compose default is 1000, so `docker/.env` has to set `ARUCO3_UID` and `ARUCO3_GID`.
  Without it CMake fails with `Unable to (re)create the private pkgRedirects
  directory`, which names neither permissions nor the user. The first three machines
  all run uid 1000 and never exercised this.
- No measurements were taken on the Thor. It is a build and test target for now; the
  benchmark and accuracy figures still come from the other three.

## See also

- [ADR-0001: Develop the CUDA implementation first in an independent repository](0001-independent-implementation.md)
- [ADR-0003: Adopt plan A as the primary approach for quadrilateral candidate extraction](0003-candidate-extraction-approach.md)
- [Implementation plan](../implementation-plan.md)
- [Evaluation plan](../evaluation-plan.md)
- [Benchmark report](../benchmark-report.md)
- [Docker environment design](../design/docker-environment.md)
- [Architecture](../architecture.md)
