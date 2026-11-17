# CPU multiwindow adaptive thresholding

## Purpose

Reduce the cost of sweeping several adaptive threshold windows without changing
the thresholded pixels produced by the OpenCV CPU baseline.

## Scope

`ThresholdWorkspace` accepts an 8-bit grayscale host image and window sides in
`[3, 4097]` (including the normalized odd form of configured side 4096). It
computes `ADAPTIVE_THRESH_MEAN_C`, `THRESH_BINARY_INV`, maximum
value 255, and replicate borders. Even configured sides are rounded up to odd
sides, as in this project's detector configuration. Threshold offsets must be
finite and in `[-255, 255]`. Strided inputs and ROIs are supported. The ROI is
isolated from its parent image, matching `adaptiveThreshold`.

The implementation is host-only. CUDA kernel time is not applicable. This is an
experimental component of the CPU detector, not an installed public API.

## Current state

The default three windows share one padded stripe and one integral image.
Outputs and scratch matrices reuse their allocations when their dimensions
remain unchanged. Prefix sums use `CV_32S`. The maximum stripe height is bounded
by both `INT32_MAX / (255 * padded_width) - 2 * radius` and a configurable
`max_prefix_bytes` budget, which defaults to 2 MiB. The budget includes the
integral's zero row/column and both halo sides. This keeps every prefix sum
within signed range while limiting the prefix-owner payload.

With a positive budget, only the active stripe and its original-image halo are
padded. At the image edges, explicit replicate padding supplies the missing
halo rows; `BORDER_ISOLATED` prevents reads outside the input ROI. Stripe heights
are distributed evenly, and shorter final stripes use views into the same
maximum-size padding and prefix owners. There is no alternating allocation
between the final stripe and the next frame. Small segmentation images remain
one stripe under the default budget.

Setting `max_prefix_bytes=0` restores whole-image padding and the original
overflow-only stripes. If a positive budget cannot hold one output row and its
halo, the implementation releases retained scratch owners and uses OpenCV. This
is a prefix-payload limit, not a bound on OpenCV's workspace or total process
memory. Configured sides above
127 also use the original calls because some OpenCV box-filter paths normalize
in float, which can move a nearly-half-integer mean across a rounding boundary
for large areas.

For a window area `A` (an odd square), integer sum `S`, source pixel `p`, and
`d = floor(constant)`, the ordinary result is 255 when
`p - round(S / A) <= -d`. Since `A` is odd there is no exact half-integer mean,
and this decision equals `S >= (p + d) * A - floor(A / 2)`. The implementation
therefore performs an integer comparison directly, avoiding both division and
an intermediate mean image. The supported fast-path area is small enough to
keep both the sum and the comparison boundary in signed 32-bit range.

The row helper explicitly separates immutable input and output pointers. This
permits compiler vectorization without introducing architecture-specific source
or compiler flags. It reads the owned padded image so that writing an output
cannot alter the source pixels read by a later window. When an output overlaps
the input, the source is cloned before output headers or pixels can change.
Different output planes must not overlap one another.

When OpenCV is configured with more than one thread, stripes with at least
`parallel_min_pixels` pixels can threshold disjoint row groups in parallel. The
minimum defaults to 32,768 pixels and is an argument so the caller can tune it.
Padding and integral generation remain sequential, and the one-thread path
does not call the parallel dispatcher.

### Validation

Independent scratch probes compared every output pixel against OpenCV 4.14.0
(`0654a42e19215ef25b1d367d822f3c630447e7c7`, unmodified, CUDA disabled):

- 1,056 comparisons used random strided ROIs from 1x1 through 640x480; odd,
  even, oversized, and fallback windows; and positive, negative, integer, and
  fractional offsets.
- A constructed near-half-mean stress test compared 4,991,143,815 pixels across
  all odd window sides from 3 through 255 and seven offsets. Each image placed
  tiles whose central window sums lay immediately above or below a half-integer
  mean for all upper mean values 1 through 255. Sides above 127 exercised the
  fallback rather than the integral implementation.

Both probes had zero pixel differences. The scratch probes are not committed;
the maintained automated tests live under `test/cpu/`.

After adding stripes, a further 180 large-image comparisons were exact. These
used 3840x2160, a strided 3841x2161 ROI, and a 65536x131 wide image, with all-white
and random pixels, odd and even windows, and signed fractional offsets. The
all-white case exercises the prefix accumulator's maximum. Large odd sizes
exercise the final shorter integral view and stripe boundaries.

### Exploratory performance

A single-process exploratory microbenchmark used DGX Spark GB10 CPU 5
(Cortex-X925), OpenCV one thread, GCC 13.3.0 `-O3 -std=c++17`, no architecture
flags, 30 warmups, and 200 paired samples. Each sample thresholds the same random
input with sides 3, 13, and 23 and offset 7; it includes padding, integral
generation, and all three output planes. File loading and input generation are
outside the measured interval. The compiler-vectorized version produced:

| Input | OpenCV median, ms | Shared integral median, ms | Ratio |
| --- | --- | --- | --- |
| 427x240 | 0.448466 | 0.077905 | 0.174 |
| 640x480 | 1.323300 | 0.239441 | 0.181 |
| 1280x720 | 3.959840 | 0.799364 | 0.202 |
| 1920x1080 | 8.977260 | 1.964810 | 0.219 |

At 427x240, OpenCV/shared-integral p95 was 0.467362/0.082128 ms and p99 was
0.477106/0.084289 ms. These are exploratory measurements, not the repeated
end-to-end detector comparison. Clock and power settings were not controlled by
the probe, and other host services remained active. The full detector evaluation
must establish the practical gain and its variation.

For 427x240 and maximum window 23 the persistent scratch payload is 591,038
bytes: a 449x262 byte padded image and a 450x263 signed 32-bit integral image.
The three output payloads add 307,440 bytes. These figures exclude allocator
alignment and `cv::Mat` headers.

### Additional experiments

The followup probes used `-O3 -DNDEBUG -std=c++17`. Single-thread runs were
pinned to CPU 6; three-thread runs to CPUs 6-8. Every reported small-image variant
also passed the 1,056 comparison probe. These measurements remain exploratory;
other repository experiments and host services were active, so the maintained
end-to-end benchmark is the authoritative speed comparison.

- Adding `-mcpu=native` did not consistently improve the row helper: at 427x240
  the generic/native medians were 0.077938/0.077666 ms, while 640x480 regressed
  from 0.236339 to 0.248467 ms. Architecture flags are not required by this helper.
- Changing the serial loop nesting to process all windows within each row did
  not help: the 427x240 median rose to 0.079441 ms, and larger sizes also
  regressed. Window-major traversal is retained inside each row group.
- Parallel row groups with three threads improved 427x240 from 0.077009 to
  0.060481 ms and 1280x720 from 0.787243 to 0.597848 ms. Unconditionally
  dispatching tiny inputs regressed: 16x16 rose from 0.000784 to 0.002016 ms and
  64x48 from 0.003168 to 0.004464 ms. The configurable minimum keeps small work
  sequential; the default is conservative relative to the observed crossover.
- For 3840x2160, 10 warmups and 50 paired samples gave OpenCV/striped-integral
  median 38.3702/9.09521 ms, p95 39.9345/9.65281 ms, and p99
  45.1483/10.1472 ms. A whole-image `CV_64F` integral alternative took
  15.4433 ms median in a nearby run and was rejected. It also needs about four
  times the prefix storage of the two `CV_32S` stripes chosen for this image.

Before cache-budget tuning (`max_prefix_bytes=0`), at 3840x2160 with windows
3, 13, and 23, the derived maximum stripe height is
2,158 rows, so two 1,080-row output stripes are used. Padded input storage is
8,426,884 bytes, prefix storage 17,043,556 bytes, and the output planes add
24,883,200 bytes. These payload sizes exclude headers and allocator alignment.

### Cache budget selection

An initial bounded sweep compared output stripe caps 32/64/128/256/512 and the
overflow-only size, with whole-image and per-stripe padding. It found that
small fixed caps regressed the segmentation image while reducing full-resolution
scratch memory. A prefix-byte budget preserves the small-image fast path and
adapts to image width. The selected default was confirmed against 0 and 4 MiB
using three independent processes, ten warmups, eighty paired samples with
rotating route order, CPU 5 for one thread and CPUs 5-8 for four threads.
Input generation was outside timing; padding, prefix generation, all output
planes, and dispatch were inside. The same random strided input was reused
for all routes, with seed 813535, windows 3/13/23 and offset 7.

| Input | Threads | Legacy median of p50, ms | 2 MiB median of p50, ms | 4 MiB median of p50, ms |
| --- | --- | --- | --- | --- |
| 427x240 | 1 | 0.081901 | 0.081949 | 0.081917 |
| 427x240 | 4 | 0.049502 | 0.049246 | 0.049134 |
| 1920x1080 | 1 | 1.958346 | 1.833599 | 1.903084 |
| 1920x1080 | 4 | 1.692596 | 1.172422 | 1.263876 |
| 3840x2160 | 1 | 8.109214 | 7.516515 | 7.616129 |
| 3840x2160 | 4 | 5.769316 | 4.844581 | 5.187832 |

At 3840x2160, the 2 MiB budget permits at most 112 output rows per stripe;
twenty uniform 108-row stripes are used. Prefix payload is 2,024,212 bytes and
padding payload 502,060 bytes, totaling 2,526,272 bytes versus 25,470,440 bytes
with the legacy geometry. The output planes are unchanged. The 427x240 scratch
payload remains 591,038 bytes.

Raw exploratory sweep and paired confirmation records, including p95/p99 and
per-process values, are archived under
[cache-stripe measurements](../docs/measurements/2026-11-17-cpu-cache-stripe/README.md).
The maintained reproduction runner is
[`experiments/cache_stripe_probe.cpp`](experiments/cache_stripe_probe.cpp).
The scratch confirmation used the same stripe arithmetic before the input-alias
guard and tiny-budget owner-release checks were added; the runner now compiles
the maintained helper including those checks. Full detector measurements remain
the authority for the practical benefit. Maintained tests cover tiny budgets,
`SIZE_MAX`, repeated shapes/budgets, white/random 4K ROIs, and in-place input.

### Provenance

This is an independent implementation of the standard summed-area-table
algorithm and the integer comparison derived above. No external implementation
was copied or adapted. OpenCV's public API was run as a compatibility oracle.
The box-filter normalization behavior was checked against
[`box_filter.simd.hpp`](https://github.com/opencv/opencv/blob/0654a42e19215ef25b1d367d822f3c630447e7c7/modules/imgproc/src/box_filter.simd.hpp)
at the pinned commit. That file has a 3-clause BSD header; no source expression,
loop structure, or comments from it are incorporated here. See the repository's
[existing threshold semantics](../docs/design/detector-pipeline.md) and
[code provenance policy](../docs/code-provenance.md).

## Goals

- Keep output pixels identical to the pinned OpenCV baseline.
- Measure the detector end-to-end with and without the multiwindow option.
- Preserve fallback behavior for configurations outside the validated fast path.

## Open questions

- Whether the same end-to-end improvement holds on Jetson Orin and x86_64.
- Whether the measured row-parallel crossover holds on other CPU architectures.
- Whether integral generation becomes the limiting cost on real, full-resolution
  inputs with ArUco3 disabled.
