# Detector submission and storage lifetime

## Purpose

Keep reusable GPU storage valid across asynchronous submissions, stream changes,
and failed initialization without reading frame counts back to the CPU.

## Scope

`Detector::Impl` submission ordering, graph lifetime, resource replacement,
readback, and error recovery. Candidate generation and decode algorithms retain
their separate module contracts. Public failure-word semantics are described in
[the C ABI and Python design](../../docs/design/c-abi-and-python.md).

## Current state

The detector owns a disable-timing completion event. Every successful submission
records it after the last pipeline operation. A partially issued direct dispatch
also records completion before returning its error. Before another submission or
readback, the destination stream waits on the most recent record. The wait is
issued even when stream handle values compare equal: CUDA may recycle a caller's
destroyed handle. No old caller stream handle is synchronized or dereferenced.

An unchanged-layout submission stays asynchronous on the host. Layout rebuilding,
reinitialization, and destruction wait for owned completion before reusing or
freeing storage. Reinitialization waits before dictionary upload even when no
allocation grows; default-stream upload otherwise need not wait for prior work
on a nonblocking stream. Graph recapture itself does not establish this ordering.
Event wait/record API cost belongs in end-to-end measurements.

Host validation and workspace sizing happen before changing a valid existing
initialization. Once resource replacement starts, initialized/result flags are
cleared and the old graph/dictionary/result views are invalidated. Therefore a
successful persistent-arena growth followed by failed frame growth cannot leave
an initialized detector pointing at freed dictionary storage. A subsequent
successful initialize can recover the object.

If completion recording fails, the result and initialization are invalidated and
the submitting stream is drained. If that drain also fails, an untracked-work
flag forces a device-wide drain attempt before later initialization or
destruction. This exceptional recovery does not add a steady-state device sync.
Dictionary-upload error paths also drain before returning, because host dictionary
storage is borrowed only for initialize's duration.

Readback waits for the producing invocation, then preserves the caller's output
until the complete result is available. Error-only stream drains keep local
readback stack/vector storage alive when a later asynchronous copy fails. Raw
CUDA drains preserve the original checked error diagnostic. Normal readback
retains its existing synchronization boundaries.

Caller-owned input ordering remains separate. Input must stay valid and unchanged
while an invocation can read it, including after a failed partial dispatch.
An input upload queued on a different stream before detect cannot be retroactively
ordered by the detector; the caller must establish that dependency. Likewise,
downstream consumers of borrowed result buffers must finish or be explicitly
ordered before another detector submission can overwrite those buffers.

## Goals

- Preserve host-asynchronous steady submission and device-resident results.
- Support nonblocking stream changes and caller stream destruction safely.
- Reject invalid partial outputs and recover after bounded injected API failures.
- Verify normal ordering under all four sanitizer tools; isolate deliberate API
  failure injection in its own test executable without production test hooks.

## Open questions

- Event API cost is included in final paired timing; no speed claim follows from
  the correctness tests alone.
- An unrecoverable CUDA context error may prevent any drain from succeeding; the
  detector remains invalid rather than claiming successful recovery.
