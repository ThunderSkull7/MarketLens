"""Unit tests for the MarketLens data cleaning pipeline."""

from __future__ import annotations

import os
from datetime import date, datetime

import pandas as pd
import pytest

from marketlens.cleaning import (
    CleaningPipeline,
    detect_duplicates,
    detect_duplicates_by_id,
    get_accepted_data,
    load_cleaned_data,
    load_cleaning_report,
    normalize_rating,
    normalize_text,
    save_cleaned_data,
    save_cleaning_report,
    standardize_date,
)
from marketlens.ingestion import IngestionRecord, IngestionResult, ingest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_result():
    """Create an IngestionResult from the committed synthetic feedback dataset."""
    return ingest("data/raw/synthetic_feedback_demo.csv")


@pytest.fixture
def custom_ingestion_result():
    """Create a mock IngestionResult with accepted, flagged, and rejected records."""
    accepted_rec = IngestionRecord(
        row_index=0,
        row_data={
            "review_id": "R100",
            "review_text": "  Great battery life!  \n\nWorks flawlessly.  ",
            "rating": 5.0,
            "created_at": "2024-03-01",
            "product_or_service": "Laptop",
            "source": "Amazon",
            "language": "EN",
            "custom_metadata": "unmapped_value_123",
        },
        status="accepted",
        reasons=[],
        mapped_values={
            "review_id": "R100",
            "review_text": "  Great battery life!  \n\nWorks flawlessly.  ",
            "rating": 5.0,
            "created_at": "2024-03-01",
            "product_or_service": "Laptop",
            "source": "Amazon",
            "language": "EN",
            "verified_label": "verified",
        },
    )

    flagged_rec = IngestionRecord(
        row_index=1,
        row_data={
            "review_id": "R100",  # Duplicate ID
            "review_text": "OK",
            "rating": 6.5,  # Out of range
            "created_at": "invalid-date",
            "product_or_service": "Laptop",
            "source": "Amazon",
            "language": "en",
        },
        status="flagged",
        reasons=["Duplicate review ID", "Rating outside expected range"],
        mapped_values={
            "review_id": "R100",
            "review_text": "OK",
            "rating": 6.5,
            "created_at": "invalid-date",
            "product_or_service": "Laptop",
            "source": "Amazon",
            "language": "en",
            "verified_label": None,
        },
    )

    rejected_rec = IngestionRecord(
        row_index=2,
        row_data={
            "review_id": "",
            "review_text": "",
            "rating": None,
            "created_at": None,
        },
        status="rejected",
        reasons=["Missing review ID", "Missing review text"],
        mapped_values={
            "review_id": "",
            "review_text": "",
            "rating": None,
            "created_at": None,
        },
    )

    quality_rep = {
        "total_rows": 3,
        "missing_review_ids": 1,
        "missing_review_text": 1,
        "invalid_ratings": 1,
        "invalid_dates": 1,
        "duplicate_ids": 1,
        "missing_value_counts": {},
        "row_accounting": {
            "total_source_rows": 3,
            "rows_parsed": 3,
            "rows_omitted_limit": 0,
            "rows_rejected": 1,
            "rows_accepted": 1,
            "rows_flagged": 1,
        },
    }

    return IngestionResult(
        accepted=[accepted_rec],
        rejected=[rejected_rec],
        flagged=[flagged_rec],
        df_validated=pd.DataFrame(),
        mapping_result=None,
        quality_report=quality_rep,
        warnings=["Duplicate ID detected"],
        errors=[],
        raw_filename="mock_feedback.csv",
    )


# ---------------------------------------------------------------------------
# Text normalisation tests
# ---------------------------------------------------------------------------


def test_normalize_text_basic():
    """Test standard whitespace trimming and blank handling."""
    assert normalize_text("  Hello   World  ") == "Hello World"
    assert normalize_text("\n\nHello\n\n") == "Hello"
    assert normalize_text(None) == ""
    assert normalize_text("") == ""
    assert normalize_text("   ") == ""
    assert normalize_text(float("nan")) == ""


def test_normalize_text_collapse_whitespace():
    """Test whitespace collapsing on lines."""
    assert normalize_text("Hello\t\tWorld", collapse_whitespace=True) == "Hello World"
    assert (
        normalize_text("Hello\t\tWorld", collapse_whitespace=False) == "Hello\t\tWorld"
    )


def test_normalize_text_line_breaks():
    """Test newline reduction."""
    assert normalize_text("A\n\n\nB") == "A\nB"
    assert normalize_text("A\r\n\r\nB") == "A\nB"


def test_normalize_text_preserves_punctuation_and_emojis():
    """Ensure punctuation, urls, and emojis are retained without destruction."""
    raw = "  Great product! Does it work? Yes: 100% 👍 Visit https://example.com  "
    cleaned = normalize_text(raw)
    assert "!" in cleaned
    assert "?" in cleaned
    assert "👍" in cleaned
    assert "https://example.com" in cleaned


def test_normalize_text_max_line_length():
    """Test max line length truncation with ellipsis."""
    long_line = "A" * 100
    res = normalize_text(long_line, max_line_length=10)
    assert len(res) <= 13
    assert res.endswith("...")


# ---------------------------------------------------------------------------
# Date standardisation tests
# ---------------------------------------------------------------------------


def test_standardize_date_valid():
    """Test parsing of various date representations."""
    assert standardize_date("2024-01-15") == date(2024, 1, 15)
    assert standardize_date("2024/01/15") == date(2024, 1, 15)
    assert standardize_date("01/15/2024") == date(2024, 1, 15)
    assert standardize_date("15-01-2024") == date(2024, 1, 15)
    assert standardize_date("2024-01-15 14:30:00") == date(2024, 1, 15)
    assert standardize_date(datetime(2024, 1, 15, 10, 0)) == date(2024, 1, 15)
    assert standardize_date(date(2024, 1, 15)) == date(2024, 1, 15)
    assert standardize_date(pd.Timestamp("2024-01-15")) == date(2024, 1, 15)


def test_standardize_date_invalid_and_missing():
    """Ensure invalid and missing dates return None without inventing values."""
    assert standardize_date(None) is None
    assert standardize_date("") is None
    assert standardize_date("not-a-date") is None
    assert standardize_date(float("nan")) is None
    assert standardize_date("NaT") is None


# ---------------------------------------------------------------------------
# Rating normalisation tests
# ---------------------------------------------------------------------------


def test_normalize_rating_scales():
    """Test normalization against diverse rating scales."""
    # 1-5 scale (within bounds)
    assert normalize_rating(3, scale="1-5") == 3.0
    assert normalize_rating("4.5", scale="1-5") == 4.5

    # Out of bounds default (clamp=False) returns None to prevent data corruption
    assert normalize_rating(0, scale="1-5") is None
    assert normalize_rating(6, scale="1-5") is None
    assert normalize_rating(-10, scale="1-5") is None
    assert normalize_rating(999, scale="1-5") is None

    # Explicit clamp=True behavior
    assert normalize_rating(0, scale="1-5", clamp=True) == 1.0
    assert normalize_rating(6, scale="1-5", clamp=True) == 5.0

    # 1-10 scale
    assert normalize_rating(8, scale="1-10") == 8.0
    assert normalize_rating(12, scale="1-10") is None
    assert normalize_rating(12, scale="1-10", clamp=True) == 10.0

    # 1-3 scale
    assert normalize_rating(2, scale="1-3") == 2.0
    assert normalize_rating(5, scale="1-3") is None
    assert normalize_rating(5, scale="1-3", clamp=True) == 3.0

    # thumbs scale (0-1)
    assert normalize_rating(1, scale="thumbs") == 1.0
    assert normalize_rating(0, scale="thumbs") == 0.0
    assert normalize_rating(2, scale="thumbs") is None


def test_normalize_rating_invalid_boundary_regression():
    """Regression test: Invalid ratings must not silently become valid ratings."""
    # Test that corrupted ratings do not become valid positive or negative ratings
    assert normalize_rating(-5, scale="1-5") is None
    assert normalize_rating(100, scale="1-5") is None
    assert normalize_rating(0.0, scale="1-5") is None


def test_normalize_rating_unparseable():
    """Test unparseable ratings return None."""
    assert normalize_rating(None) is None
    assert normalize_rating("excellent") is None
    assert normalize_rating(float("nan")) is None


# ---------------------------------------------------------------------------
# Duplicate detection tests
# ---------------------------------------------------------------------------


def test_detect_duplicates_by_id():
    """Test duplicate detection by ID only."""
    df = pd.DataFrame(
        {
            "review_id": ["R001", "R002", "R001", "R003"],
            "review_text": ["Good", "Okay", "Good", "Bad"],
        }
    )
    res_df, dup_count, groups = detect_duplicates_by_id(df, id_match_only=True)
    assert dup_count == 1
    assert bool(res_df.loc[0, "is_duplicate"]) is False
    assert bool(res_df.loc[2, "is_duplicate"]) is True
    assert len(groups) == 1
    assert groups[0][0] == "R001"
    assert groups[0][1] == [0, 2]


def test_detect_duplicates_by_id_and_text():
    """Test duplicate detection distinguishing ID match from ID+text match."""
    df = pd.DataFrame(
        {
            "review_id": ["R001", "R001", "R001"],
            "review_text": ["First text", "Different text", "First text"],
        }
    )
    res_df, dup_count, groups = detect_duplicates(df, id_match_only=False)
    # Row 0 and Row 2 have both same ID and text
    assert dup_count == 1
    assert bool(res_df.loc[0, "is_duplicate"]) is False
    assert bool(res_df.loc[1, "is_duplicate"]) is False
    assert bool(res_df.loc[2, "is_duplicate"]) is True


def test_detect_duplicates_preserves_all_records():
    """Verify duplicate detection marks rows without silently deleting them."""
    df = pd.DataFrame(
        {
            "review_id": ["R001", "R001"],
            "review_text": ["Text A", "Text A"],
        }
    )
    res_df, dup_count, _ = detect_duplicates(df)
    assert len(res_df) == 2  # No rows removed
    assert res_df["is_duplicate"].sum() == 1


# ---------------------------------------------------------------------------
# CleaningPipeline tests
# ---------------------------------------------------------------------------


def test_cleaning_pipeline_accepted_only(custom_ingestion_result):
    """Test pipeline excluding flagged records by default."""
    pipeline = CleaningPipeline(custom_ingestion_result)
    cleaned_df, report = pipeline.run(include_flagged=False)

    assert len(cleaned_df) == 1
    assert cleaned_df.iloc[0]["review_id"] == "R100"
    assert cleaned_df.iloc[0]["_status"] == "accepted"
    # Unmapped source columns preserved
    assert "custom_metadata" in cleaned_df.columns
    assert cleaned_df.iloc[0]["custom_metadata"] == "unmapped_value_123"

    # Report verification
    assert report["report"] == "cleaning_report"
    assert report["input_row_count"] == 3
    assert report["accepted_count"] == 1
    assert report["rejected_count"] == 1
    assert report["flagged_count"] == 1
    assert report["rows_retained"] == 1
    assert report["rows_excluded"] == 2
    assert "text_normalisation" in report["cleaning_operations"]


def test_cleaning_pipeline_include_flagged(custom_ingestion_result):
    """Test pipeline with optional flagged-record inclusion."""
    pipeline = CleaningPipeline(custom_ingestion_result)
    cleaned_df, report = pipeline.run(include_flagged=True)

    # 1 accepted + 1 flagged = 2 retained; rejected is still excluded
    assert len(cleaned_df) == 2
    assert report["rows_retained"] == 2
    assert report["rows_excluded"] == 1
    assert set(cleaned_df["_status"]) == {"accepted", "flagged"}


def test_cleaning_pipeline_rejected_inspection(custom_ingestion_result):
    """Ensure rejected records remain available for inspection."""
    pipeline = CleaningPipeline(custom_ingestion_result)
    rejected = pipeline.get_rejected_records()
    assert len(rejected) == 1
    assert rejected[0].status == "rejected"
    assert "Missing review ID" in rejected[0].reasons


def test_cleaning_pipeline_empty():
    """Test pipeline handling an empty IngestionResult cleanly."""
    empty_result = IngestionResult(
        accepted=[],
        rejected=[],
        flagged=[],
        df_validated=pd.DataFrame(),
        mapping_result=None,
        quality_report={},
        warnings=[],
        errors=[],
        raw_filename="empty.csv",
    )
    pipeline = CleaningPipeline(empty_result)
    cleaned_df, report = pipeline.run()
    assert cleaned_df.empty
    assert report["rows_retained"] == 0
    assert report["input_row_count"] == 0


def test_get_accepted_data_helper(synthetic_result):
    """Test convenience wrapper get_accepted_data."""
    cleaned_df, report = get_accepted_data(synthetic_result)
    assert len(cleaned_df) == len(synthetic_result.accepted)
    assert "report" in report
    assert report["input_row_count"] == len(synthetic_result.accepted)


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------


def test_save_and_load_cleaned_data(synthetic_result, tmp_path):
    """Test saving and loading cleaned dataset."""
    cleaned_df, _ = get_accepted_data(synthetic_result)
    saved_path = save_cleaned_data(
        cleaned_df,
        source_file="synthetic_feedback_demo.csv",
        output_dir=str(tmp_path),
        filename="test_cleaned.csv",
    )
    assert os.path.exists(saved_path)

    loaded_df = load_cleaned_data(saved_path)
    assert len(loaded_df) == len(cleaned_df)
    assert "review_id" in loaded_df.columns


def test_save_and_load_cleaning_report(synthetic_result, tmp_path):
    """Test saving and loading cleaning report JSON."""
    _, report = get_accepted_data(synthetic_result)
    saved_path = save_cleaning_report(
        report,
        output_dir=str(tmp_path),
        filename="test_report.json",
    )
    assert os.path.exists(saved_path)

    loaded_rep = load_cleaning_report(saved_path)
    assert loaded_rep["report"] == "cleaning_report"
    assert loaded_rep["input_row_count"] == report["input_row_count"]
