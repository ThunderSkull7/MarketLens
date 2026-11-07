# ADR-0001: Selected Python version — 3.11 / 3.12 compatibility range

- **Date:** 2026-11-05
- **Status:** Accepted

## Context

The prompt requires pinning one supported Python version (3.11 or 3.12).

## Decision

Target **Python 3.11 as the minimum** (`requires-python = ">=3.11"`),
developed and CI-tested on **Python 3.12**. Ruff `target-version = "py311"`.

## Consequences

- Uses modern typing/introspection available from 3.11.
- Local machines without 3.11 may install 3.12 — the range accepts both.
- CI pins 3.12 for reproducibility (see `.github/workflows/ci.yml`).
