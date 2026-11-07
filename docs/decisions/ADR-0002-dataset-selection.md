# ADR-0002: Synthetic demonstration dataset for Batch 1

- **Date:** 2026-11-05
- **Status:** Accepted

## Context

The project needs a dataset that supports sentiment analysis, theme
discovery and analytics in later batches. Candidate public datasets
(Amazon Reviews, Yelp Open Dataset) carry licence conditions (Yelp:
CC BY-NC-SA 4.0, non-commercial) and large file sizes, and their licences
were not fully verified within this batch for automated download.

## Decision

- Use a small, **clearly-labelled synthetic dataset**
  (`data/raw/synthetic_feedback_demo.csv`, 20 rows) for development,
  testing and CI.
- Document the real-dataset candidates and their licence caveats in
  `data/README.md`; do **not** silently adopt any dataset whose licence
  cannot be verified.
- Large datasets must never be committed to the repository.

## Consequences

- All tests are self-contained: no network, no paid APIs.
- A real, licence-verified dataset must be selected before the analytics
  batches; the ingestion layer already accepts external CSVs.
