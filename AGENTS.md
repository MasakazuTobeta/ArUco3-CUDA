# AGENTS Guide

Agents working in this repository should read `CONTRIBUTING.md` first. `CONTRIBUTING.md` is the authoritative source for the shared rules.

## Precedence

When rules conflict, apply them in this order.

1. Explicit instructions from the user
2. `CONTRIBUTING.md`
3. The `AGENTS.md` in the repo root or in the target directory
4. The `README.md` of the area concerned and `docs/**`

## First Read Checklist

- Read `CONTRIBUTING.md`.
- When changing documents, read `docs/AGENTS.md` and `docs/terminology.md`.
- Read the design document, ADR, and evaluation plan that correspond to what you are changing.
- For a non-trivial change, work out the purpose, dependencies, risks, verification method, and the documents that need updating.

## Working Principles

- Do not mix the current implementation with the goals.
- Do not assume that CUDA is always faster; compare CPU, CUDA, and hybrid approaches using measurement results.
- Record kernel-only time separately from time that includes input preparation, synchronization, and result retrieval.
- Use the same algorithm on DGX Spark and Jetson Orin, and keep machine-specific optimizations separate as explicit options.
- Do not leave thresholds and limits as fixed values in the source alone; structure them so they can be overridden from configuration.
- Do not copy code from external implementations without authorization. When a paper, a public specification, or a compatibly licensed implementation is used as a reference, record the source and its influence.
- Record important design decisions in an ADR or a related document.

## Documentation Rules

- Specification, design, and evaluation documents must have at least `Purpose`, `Scope`, `Current state`, `Goals`, and `Open questions`.
- When behavior, configuration, the public API, or evaluation conditions change, update the related documents in the same change.
- Place a sidecar `*.md` next to a module whose design decisions are not conveyed by Doxygen alone. See Documentation in `CONTRIBUTING.md` for where each area's records belong.
- Prefer Mermaid for overview diagrams, since diffs are easier to review.

## Completion Checklist

- Was compatibility with the relevant CPU baseline implementation confirmed?
- Were tests added for normal cases, error cases, and boundary values?
- Can CUDA errors and failures in asynchronous execution be detected?
- Are kernel-only and end-to-end measurements kept distinct?
- Was verification done on at least the target side of DGX Spark and Jetson Orin, or was the reason it was not verified recorded?
- Are the README, design, evaluation conditions, and ADRs free of contradictions?
- Are secrets, build outputs, and large datasets kept out of the commit?

## Operational Know-How

When a failure that could recur or an environment-specific caveat comes to light during work, add the cause and the correct procedure to this section.

### Initializing an empty repository through a GitHub App

- In a completely empty repository, blob creation through the Git Data API fails with `409 Git Repository is empty`.
- First create a single file such as `README.md` through the Contents API to initialize the default branch.
- After initialization, multiple files can be combined into one commit in the order blob, tree, commit, ref update.
