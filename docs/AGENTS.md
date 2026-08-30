# docs AGENTS Guide

This file holds supplementary rules for agents creating and updating `docs/**`. It assumes the `AGENTS.md` and `CONTRIBUTING.md` in the repo root.

## Language And Structure

- As a rule, write documents in English.
- Use the preferred wording in `docs/terminology.md`.
- Design, planning, and evaluation documents must have `Purpose`, `Scope`, `Current state`, `Goals`, and `Open questions`.
- Do not describe unimplemented content as a current feature.
- Do not fill in open questions with guesses; leave them as `TODO` or under `Open questions`.
- Reference other documents by relative path.

## Diagram Rules

- Prefer Mermaid for explaining data flow, responsibilities, and processing stages.
- Add a short explanation before and after each diagram.
- Do not pack too much information into a single diagram; split it up where necessary.
- When adding an external image, place it in `docs/img/` and record its source and license.

## Synchronization

- When the public API, configuration, supported hardware, or evaluation conditions change, update the related documents in the same change.
- When an ADR supersedes a decision, do not delete the old ADR; mark it `Superseded` and link to the successor ADR.
- Include the environment information needed to reproduce a benchmark result alongside it.
