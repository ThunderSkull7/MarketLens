"""Integration tests for the complete MarketLens data cleaning and analytics layer."""

from __future__ import annotations

import hashlib
from pathlib import Path

from marketlens.cleaning import CleaningPipeline, run_pipeline
from marketlens.ingestion import ingest


def _get_file_hash(path: Path) -> str:
    """Compute sha256 checksum of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def test_e2e_cleaning_pipeline_preserves_raw_file(tmp_path):
    """Verify that running the entire cleaning pipeline never modifies raw data."""
    raw_path = Path("data/raw/synthetic_feedback_demo.csv")
    hash_before = _get_file_hash(raw_path)

    summary = run_pipeline(
        raw_path,
        include_flagged=False,
        rating_scale="1-5",
        output_dir=str(tmp_path / "processed"),
        report_dir=str(tmp_path / "reports"),
    )

    hash_after = _get_file_hash(raw_path)
    assert hash_before == hash_after, "Raw CSV was modified by pipeline execution!"

    assert Path(summary["csv_path"]).exists()
    assert Path(summary["cleaning_report_path"]).exists()
    assert Path(summary["analytics_report_path"]).exists()


def test_e2e_traceability_from_cleaned_to_ingestion():
    """Verify that every cleaned record traces back accurately to ingestion records."""
    raw_path = Path("data/raw/synthetic_feedback_demo.csv")
    ingest_result = ingest(raw_path)

    pipeline = CleaningPipeline(ingest_result)
    cleaned_df, report = pipeline.run(include_flagged=True)

    # Every retained row must have a valid _row_index corresponding to ingestion records
    for _, row in cleaned_df.iterrows():
        idx = row["_row_index"]
        matching_ingest_records = [
            r
            for r in (ingest_result.accepted + ingest_result.flagged)
            if r.row_index == idx
        ]
        assert len(matching_ingest_records) == 1
        ingest_rec = matching_ingest_records[0]
        assert row["review_id"] == ingest_rec.mapped_values.get("review_id")
        assert row["_status"] == ingest_rec.status
