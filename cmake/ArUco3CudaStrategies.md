# CUDA algorithm selection

## Purpose

Expose reproducible algorithm choices and tunable limits without relying on
unrecorded compiler flags or changing strategy silently with the GPU model.

## Scope

CMake strategy selection for contour extraction, thresholding, pyramid sampling,
Otsu, and candidate-stage launch grids. Resize compatibility is selected
[separately](ArUco3CudaResizeProfile.md). Contour count, approximation, and materialization variants are explicit
sub-strategies. The separate active-edge topology prototype is not integrated.

## Current state

Fresh builds retain the previous strategies until combined quality, resource,
and performance gates pass. Each cache entry is overridable with `-D`.

| Cache entry | Values (default first) | Meaning |
| --- | --- | --- |
| `ARUCO3CUDA_CONTOUR_STRATEGY` | `COMPONENT_EXTREMA`, `ORDERED_CONTOURS` | Existing component candidates or ordered contours with close-contour retry |
| `ARUCO3CUDA_THRESHOLD_STRATEGY` | `SEPARABLE`, `FUSED_AUTO` | Per-window row/column kernels or exact tiled multiwindow kernel with supported-range fallback |
| `ARUCO3CUDA_PYRAMID_STRATEGY` | `FULL`, `DEMAND` | Full image pyramid or complete levels through the requested prefix (sparse pixels are not implemented) |
| `ARUCO3CUDA_OTSU_STRATEGY` | `REFERENCE`, `FIXED_POINT`, `DENSITY_FIXED_POINT` | Reference scan, exact fixed-point shortcut, or density-selected shortcut |
| `ARUCO3CUDA_CANDIDATE_GRID_STRATEGY` | `CAPACITY`, `PERSISTENT` | One block per capacity entry or bounded blocks looping over current device counts |
| `ARUCO3CUDA_CONTOUR_COUNT_STRATEGY` | `SERIAL`, `EDGE_AGGREGATE` | Explicit contour implementation variant |
| `ARUCO3CUDA_CONTOUR_RDP_STRATEGY` | `SERIAL`, `WARP` | Explicit contour implementation variant |
| `ARUCO3CUDA_CONTOUR_MATERIALIZE_STRATEGY` | `SERIAL`, `WARP` | Explicit contour implementation variant |
| `ARUCO3CUDA_CONTOUR_MATERIALIZE_COMPACTION_STRATEGY` | `FULL_SLOTS`, `ELIGIBLE_EDGES` | Explicit contour implementation variant |
| `ARUCO3CUDA_RESIZE_COEFFICIENT_STRATEGY` | `DIRECT`, `KLEIDI_PRECOMPUTED` | Direct coordinate arithmetic or immutable Kleidi-only layout coefficients |
| `ARUCO3CUDA_OTSU_MAX_OCCUPIED_BINS` | Integer 1–256, default 96 | Density cutoff used only by `DENSITY_FIXED_POINT` |
| `ARUCO3CUDA_CANDIDATE_GRID_BLOCKS` | Integer 1–2147483647, default 128 | Upper bound on persistent blocks, also bounded by configured capacity |

`compiled_gpu_strategies()` exposes immutable identifiers and limits from the
core-only `build_profile.cpp`. `compiled_gpu_strategy_identity()` renders a
canonical string containing all main and contour sub-strategies, topology identity, and both configured limits,
including inactive tuning values. Benchmark environment/conditions and audit
frames record it as `gpu_strategy_identity`; aggregation keeps different and
unknown identities separate. The public C configuration/result layouts do not
change.

These values map to the existing internal experimental macros. CMake rejects
unknown enums, invalid numeric bounds, and contradictory legacy defines in
C++/CUDA flags, including configuration-specific flag variables. An old
`ARUCO3CUDA_EXPERIMENT_CONTOUR_CANDIDATES` cache entry must agree with the new
contour selection or be removed. Consistent old flags remain accepted to make
controlled migration possible; they are unnecessary in maintained builds.

`KLEIDI_PRECOMPUTED` prepares reusable coordinate coefficients only for the
Kleidi resize path. Carotene and other paths keep direct arithmetic: measured
Carotene 7-bit caching regressed, while its 11-bit path had no clear worthwhile
gain. The internal all-mode coefficient preparation remains a test ablation,
not a production strategy. Coefficients contain no image pixels or detections.
The startup preparation and its synchronization contract are unchanged.

`ELIGIBLE_EDGES` compaction requires `EDGE_AGGREGATE` count and `WARP`
materialization; unsupported combinations fail configuration. The live contour
topology is reported as `ORIGINAL_SLOTS` (four pixel slots plus compact root
descriptors). Warp width 32 and the proven point bound `2WH+W+H` are algorithmic
properties, not tunable quality thresholds.

The candidate block limit is an experimental tuning value, not a claim that 128
is best for every density or device. `FIXED_POINT` is retained for ablation even
where the density-selected path is faster. No optimized preset is activated by
this plumbing; paired quality and timing must use explicitly recorded choices.

### Explicit steady-state presets

`dgx-spark-steady` and `jetson-orin-steady` inherit their validated ARM resize
profiles and select the same algorithms: ordered contours, fused thresholding,
demand pyramid, density-selected Otsu with cutoff 96, persistent candidate grid
with 512 blocks, edge aggregation, warp approximation/materialization,
eligible-edge compaction, and Kleidi-only precomputed coefficients.
`ARUCO3CPU_NATIVE=ON` makes the paired optimized CPU build match the intended
performance baseline. The [combined quality and performance record](../docs/gpu-steady-state-report.md)
retains both gains and losses. Fresh build defaults remain
legacy, and `CAPACITY` remains available as a dense-workload control.

The 512-block choice balances sparse and dense work; a stress case with 16384
active candidates showed a 10–20% decode-stage regression. This must not be
presented as a universal gain. Whole-pipeline dense-scene acceptance is measured
separately. Runtime `max_candidates`/`max_markers` are not preset build values;
benchmark them explicitly (the validation candidate budget is 16384) and retain
them in every result.

## Goals

- Preserve baseline controls and reject ambiguous configurations before building.
- Allow the same strategy and tuning values on both target GPU architectures.
- Select final measured settings only after combined quality and resource gates.

## Open questions

- Does the selected persistent512 preset meet whole-pipeline acceptance on dense scenes?
- Which independently measured contour trace/count/materialization choices belong
  in the final maintained strategy set?
- Which optimized settings should be enabled by an explicit maintained preset?
