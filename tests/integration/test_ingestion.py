"""Integration tests for the MarketLens ingestion pipeline."""

from __future__ import annotations

import pytest

from marketlens.exceptions import (
    InvalidCSVError,
)
from marketlens.ingestion import ingest
from marketlens.src import load_settings

# ---------------------------------------------------------------------------
# Integration test fixtures (using synthetic dataset)
# ---------------------------------------------------------------------------


def test_e2e_valid_synthetic_dataset():
    """Test end-to-end valid CSV ingestion with the synthetic dataset."""
    result = ingest("data/raw/synthetic_feedback_demo.csv")

    # Verify result structure
    assert result is not None
    assert hasattr(result, "accepted_count")
    assert hasattr(result, "rejected_count")
    assert hasattr(result, "flagged_count")
    assert hasattr(result, "total_rows")
    assert hasattr(result, "quality_report")
    assert hasattr(result, "mapping_result")
    assert hasattr(result, "warnings")
    assert hasattr(result, "errors")

    # Should have some accepted records
    assert result.accepted_count > 0, "Expected some accepted records"

    # Should have a mapping result
    assert result.mapping_result is not None


def test_e2e_valid_synthetic_dataset_summary():
    """Test ingestion summary output."""
    result = ingest("data/raw/synthetic_feedback_demo.csv")
    summary = result.summary()
    assert "synthetic_feedback_demo.csv" in summary


def test_e2e_column_mapping_explicit():
    """Test ingestion with explicit column mapping."""
    # The demo dataset already uses canonical header names; supply an
    # explicit mapping for the required columns to exercise the explicit path.
    explicit_mapping = {
        "review_id": "review_id",
        "review_text": "review_text",
        "rating": "rating",
        "created_at": "created_at",
    }
    result = ingest(
        "data/raw/synthetic_feedback_demo.csv",
        explicit_mapping=explicit_mapping,
    )
    assert result.accepted_count > 0
    assert "review_id" in result.quality_report["mapping_info"]["mapped_columns"]


def test_e2e_with_rating_scale():
    """Test ingestion with custom rating scale."""
    result = ingest(
        "data/raw/synthetic_feedback_demo.csv",
        rating_scale="1-5",
    )
    assert result.accepted_count > 0


def test_e2e_quality_report_fields():
    """Test that quality report contains expected fields."""
    result = ingest("data/raw/synthetic_feedback_demo.csv")

    report = result.quality_report
    expected_fields = [
        "total_rows",
        "missing_review_ids",
        "missing_review_text",
        "invalid_ratings",
        "invalid_dates",
        "duplicate_ids",
        "missing_value_counts",
    ]
    for field in expected_fields:
        assert field in report, f"Missing field '{field}' in quality report"


def test_e2e_mapping_info():
    """Test that mapping information is preserved."""
    result = ingest("data/raw/synthetic_feedback_demo.csv")

    mapping_info = result.mapping_info
    assert mapping_info is not None
    # Should have some mapped columns
    assert len(mapping_info.get("mapped_columns", {})) > 0


# ---------------------------------------------------------------------------
# Record classification tests
# ---------------------------------------------------------------------------


def test_e2e_record_classification():
    """Test that records are classified into accepted/rejected/flagged."""
    result = ingest("data/raw/synthetic_feedback_demo.csv")

    # Total records should be accounted for
    total_accounted = (
        result.accepted_count + result.rejected_count + result.flagged_count
    )
    assert total_accounted == result.total_rows, (
        f"Total accounted ({total_accounted}) != total rows ({result.total_rows})"
    )


def test_e2e_at_least_some_accepted():
    """Test that at least some records are accepted with valid data."""
    result = ingest("data/raw/synthetic_feedback_demo.csv")
    assert result.accepted_count >= 1


# ---------------------------------------------------------------------------
# Raw input preservation tests
# ---------------------------------------------------------------------------


def test_e2e_raw_input_preservation():
    """Test that raw input file is preserved."""
    from pathlib import Path

    # Get original file modification time
    original_path = Path("data/raw/synthetic_feedback_demo.csv")
    original_mtime = original_path.stat().st_mtime

    # Run ingestion
    ingest("data/raw/synthetic_feedback_demo.csv")

    # File should still exist with same mtime (ingestion shouldn't modify it)
    assert original_path.exists()
    # The file mtime should not have changed due to ingestion
    new_mtime = original_path.stat().st_mtime
    # Allow small time difference
    assert abs(new_mtime - original_mtime) < 1.0


# ---------------------------------------------------------------------------
# Ingestion metadata tests
# ---------------------------------------------------------------------------


def test_e2e_metadata_present():
    """Test that ingestion metadata is correct."""
    result = ingest("data/raw/synthetic_feedback_demo.csv")

    # Check raw filename
    assert result.raw_filename == "synthetic_feedback_demo.csv"

    # Check quality report has expected metadata
    assert "warnings" in result.quality_report
    assert "errors" in result.quality_report


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


def test_e2e_missing_file():
    """Test ingestion with a missing file."""
    try:
        ingest("nonexistent_file.csv")
        pytest.fail("Expected InvalidCSVError to be raised")
    except InvalidCSVError:
        pass  # Expected


def test_e2e_with_max_rows():
    """Test ingestion with max_rows parameter."""
    result = ingest(
        "data/raw/synthetic_feedback_demo.csv",
        max_rows=5,
    )
    assert result.total_rows == 5  # Should limit to 5 rows


def test_e2e_settings_integration():
    """Test ingestion uses settings correctly."""
    settings = load_settings()
    # Settings should be loadable without error
    assert settings is not None
    assert hasattr(settings, "max_upload_size_mb")
    assert hasattr(settings, "logging_level")
