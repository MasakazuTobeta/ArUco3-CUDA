# Close-contour retries

## Purpose

Recover markers whose largest grouped quad does not decode but a sufficiently
different contour in the same group does. CPU quality comparisons found this
condition in `failmask9.png` and five markers of `regression_2492.png` from the
pinned OpenCV fixtures.

## Scope

Device preparation and resolution around a second canonical-sampling, cell-decode
and dictionary-match pass. The detector integrates this route with experimental
contour candidates. No host readback or CPU fallback is used.

## Current state

After primary matching, one thread per failed representative walks its group in
descending Euclidean quad-perimeter order. The first comparison is against the
representative; each retained alternative becomes the next comparison reference.
An alternative is retained only when its RMS corner distance exceeds
`min_group_distance * average_module_size`, following the maintained CPU grouping
helper. Successful primary groups produce no retry candidates.

A device exclusive scan packs retained alternatives in stable rank order. Each
inherits the representative's contour point count for canonical level selection:
the CPU decode loop reuses the representative's pyramid level and scale for its
close contours. The original candidate point count is not a replacement for this
inherited sampling parameter.

The detector reuses canonical and cell-ratio storage after primary matching, but
keeps separate primary and retry match arrays. Alternatives of failed groups are
decoded in one batch; this can evaluate later alternatives after an earlier one
would have succeeded in the CPU's serial loop. Resolution chooses the earliest
successful alternative and updates only its representative's corners and match
result. Primary successes and groups whose alternatives all fail stay unchanged.

Containment is built from the original representatives before retry resolution,
matching the CPU's order of operations. Suppression uses that original tree with
the final match results. No fixed small retry count is imposed: each raw candidate
belongs to one group and can be emitted at most once, so the existing raw-candidate
budget bounds all retry storage.

The implementation is authored from this repository's CPU grouping and detection
semantics. No official ArUco GPL source is used. Tests exercise primary success,
failed-primary recovery, first-success priority, exclusion of near duplicates,
inherited sampling length, all-failed preservation, and 31 eligible alternatives.

## Goals

Match or improve CPU detection quality without introducing frame history, host
synchronization, hidden truncation, or a different pyramid scale for retries.
Measure the extra launches and failed-group work separately from detection quality.

## Open questions

The CUDA grouping relation still uses connected components rather than the CPU's
ordered group assignment. Broader cases may require addressing that independent
difference. The batched retry pass may benefit from more selective device work;
any optimization must preserve stable first-success selection and complete coverage.
