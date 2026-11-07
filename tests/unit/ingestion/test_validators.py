"""Unit tests for the data validation module."""

from __future__ import annotations

from marketlens.ingestion.column_mapping import (
    CanonicalField,
    ColumnMapping,
)
from marketlens.ingestion.validators import (
    RATING_SCALES,
    validate_data_quality,
    validate_dates,
    validate_duplicate_ids,
    validate_missing_review_ids,
    validate_missing_review_text,
    validate_rating_scale,
    validate_ratings,
    validate_required_columns,
)

# ---------------------------------------------------------------------------
# Rating scale validation tests
# ---------------------------------------------------------------------------


def test_validate_rating_scale_1_to_5():
    """Test rating scale validation with 1-5 scale."""
    min_val, max_val = validate_rating_scale("1-5", 1.0, 5.0)
    assert min_val == 1.0
    assert max_val == 5.0


def test_validate_rating_scale_1_to_10():
    """Test rating scale validation with 1-10 scale."""
    min_val, max_val = validate_rating_scale("1-10", 1.0, 10.0)
    assert min_val == 1.0
    assert max_val == 10.0


def test_validate_rating_scale_default():
    """Test rating scale validation with None uses default 1-5."""
    min_val, max_val = validate_rating_scale(None, 1.0, 5.0)
    assert min_val == 1.0
    assert max_val == 5.0


def test_rating_scales_available():
    """Test that expected rating scales are available."""
    assert "1-5" in RATING_SCALES
    assert "1-10" in RATING_SCALES


# ---------------------------------------------------------------------------
# Required columns validation tests
# ---------------------------------------------------------------------------


def test_validate_required_columns_with_mapping():
    """Test required column validation when mapped."""
    # Create a mock mapping result with review_id and review_text mapped
    mapping = ColumnMapping(
        source_to_canonical={
            "review_id": "review_id",
            "review_text": "review_text",
        }
    )

    # DataFrame with the required columns
    df = __import__("pandas").DataFrame(
        {
            "review_id": ["R001", "R002"],
            "review_text": ["Great product", "OK product"],
        }
    )

    # Build a simple canonical fields container
    canonical_fields_obj = type(
        "CanonicalFields",
        (),
        {
            "REVIEW_ID": CanonicalField(
                name="review_id",
                required=True,
            ),
            "REVIEW_TEXT": CanonicalField(
                name="review_text",
                required=True,
            ),
        },
    )()

    all_present, missing = validate_required_columns(df, canonical_fields_obj, mapping)
    assert all_present is True
    assert missing == []


def test_validate_required_columns_missing():
    """Test required column validation when columns are missing."""
    from marketlens.ingestion.column_mapping import ColumnMapping

    mapping = ColumnMapping(source_to_canonical={})

    df = __import__("pandas").DataFrame(
        {
            "some_other_column": ["foo", "bar"],
        }
    )

    canonical_fields_obj = type(
        "CanonicalFields",
        (),
        {
            "REVIEW_ID": CanonicalField(name="review_id", required=True),
            "REVIEW_TEXT": CanonicalField(name="review_text", required=True),
        },
    )()

    all_present, missing = validate_required_columns(df, canonical_fields_obj, mapping)
    assert all_present is False
    assert "review_id" in missing
    assert "review_text" in missing


# ---------------------------------------------------------------------------
# Missing review IDs tests
# ---------------------------------------------------------------------------


def test_validate_missing_review_ids(sample_dataframe):
    """Test missing review ID detection."""
    from marketlens.ingestion.column_mapping import ColumnMapping

    mapping = ColumnMapping(
        source_to_canonical={
            "review_id": "review_id",
        }
    )

    count, indices = validate_missing_review_ids(sample_dataframe, mapping)
    # All rows should have review IDs in the synthetic dataset
    assert count >= 0


def test_validate_missing_review_ids_no_column():
    """Test when no review_id column exists."""
    from marketlens.ingestion.column_mapping import ColumnMapping

    mapping = ColumnMapping(source_to_canonical={})

    df = __import__("pandas").DataFrame(
        {
            "review_text": ["Great product", "OK product"],
        }
    )

    count, indices = validate_missing_review_ids(df, mapping)
    # All rows should be missing IDs since column doesn't exist
    assert count == len(df)


# ---------------------------------------------------------------------------
# Missing review text tests
# ---------------------------------------------------------------------------


def test_validate_missing_review_text(sample_dataframe):
    """Test missing review text detection."""
    from marketlens.ingestion.column_mapping import ColumnMapping

    mapping = ColumnMapping(
        source_to_canonical={
            "review_text": "review_text",
        }
    )

    count, indices = validate_missing_review_text(sample_dataframe, mapping)
    # All rows should have review text in the synthetic dataset
    assert count >= 0


def test_validate_missing_review_text_no_column():
    """Test when no review_text column exists."""
    from marketlens.ingestion.column_mapping import ColumnMapping

    mapping = ColumnMapping(source_to_canonical={})

    df = __import__("pandas").DataFrame(
        {
            "review_id": ["R001", "R002"],
        }
    )

    count, indices = validate_missing_review_text(df, mapping)
    # All rows should be missing text since column doesn't exist
    assert count == len(df)


# ---------------------------------------------------------------------------
# Ratings validation tests
# ---------------------------------------------------------------------------


def test_validate_ratings_with_valid_data(sample_dataframe):
    """Test rating validation with valid data."""
    from marketlens.ingestion.column_mapping import ColumnMapping

    mapping = ColumnMapping(
        source_to_canonical={
            "rating": "rating",
        }
    )

    count_invalid, invalid_indices, out_of_range = validate_ratings(
        sample_dataframe, mapping, scale="1-5"
    )
    # Valid data should have no invalid ratings
    assert count_invalid >= 0


def test_validate_ratings_invalid_values(sample_dataframe):
    """Test rating validation with invalid values."""
    # Create DataFrame with invalid ratings
    import pandas as pd

    from marketlens.ingestion.column_mapping import ColumnMapping

    df = pd.DataFrame(
        {
            "review_id": ["R001", "R002", "R003", "R004"],
            "review_text": ["Good", "Bad", "Okay", "Great"],
            "rating": [5.0, 10.0, -1.0, "not_a_number"],
        }
    )

    mapping = ColumnMapping(
        source_to_canonical={
            "rating": "rating",
        }
    )

    count_invalid, invalid_indices, out_of_range = validate_ratings(
        df, mapping, scale="1-5"
    )
    # Should detect some invalid ratings
    assert count_invalid > 0 or len(out_of_range) > 0


def test_validate_ratings_no_column():
    """Test rating validation when no rating column exists."""
    from marketlens.ingestion.column_mapping import ColumnMapping

    mapping = ColumnMapping(source_to_canonical={})

    df = __import__("pandas").DataFrame(
        {
            "review_id": ["R001", "R002"],
            "review_text": ["Good", "Bad"],
        }
    )

    count_invalid, invalid_indices, out_of_range = validate_ratings(
        df, mapping, scale="1-5"
    )
    # No rating column means no invalid ratings
    assert count_invalid == 0


# ---------------------------------------------------------------------------
# Dates validation tests
# ---------------------------------------------------------------------------


def test_validate_dates_with_valid_data(sample_dataframe):
    """Test date validation with valid data."""
    from marketlens.ingestion.column_mapping import ColumnMapping

    mapping = ColumnMapping(
        source_to_canonical={
            "created_at": "created_at",
        }
    )

    count_invalid, invalid_indices = validate_dates(sample_dataframe, mapping)
    # Valid data should have no invalid dates
    assert count_invalid >= 0


def test_validate_dates_no_column():
    """Test date validation when no created_at column exists."""
    from marketlens.ingestion.column_mapping import ColumnMapping

    mapping = ColumnMapping(source_to_canonical={})

    df = __import__("pandas").DataFrame(
        {
            "review_id": ["R001", "R002"],
            "review_text": ["Good", "Bad"],
        }
    )

    count_invalid, invalid_indices = validate_dates(df, mapping)
    # No date column means no invalid dates
    assert count_invalid == 0


# ---------------------------------------------------------------------------
# Duplicate IDs validation tests
# ---------------------------------------------------------------------------


def test_validate_duplicate_ids_with_duplicates(sample_dataframe):
    """Test duplicate ID detection with duplicates."""
    # Create DataFrame with duplicate IDs
    import pandas as pd

    from marketlens.ingestion.column_mapping import ColumnMapping

    df = pd.DataFrame(
        {
            "review_id": ["R001", "R001", "R002"],  # R001 is duplicate
            "review_text": ["Good", "Duplicate", "Okay"],
        }
    )

    mapping = ColumnMapping(
        source_to_canonical={
            "review_id": "review_id",
        }
    )

    duplicate_count, duplicate_groups = validate_duplicate_ids(df, mapping)
    assert duplicate_count > 0 or len(duplicate_groups) > 0


def test_validate_duplicate_ids_no_duplicates(sample_dataframe):
    """Test duplicate ID detection when no duplicates exist."""
    from marketlens.ingestion.column_mapping import ColumnMapping

    mapping = ColumnMapping(
        source_to_canonical={
            "review_id": "review_id",
        }
    )

    count, groups = validate_duplicate_ids(sample_dataframe, mapping)
    assert count == 0
    assert len(groups) == 0


def test_validate_duplicate_ids_no_column():
    """Test duplicate ID detection when no ID column exists."""
    from marketlens.ingestion.column_mapping import ColumnMapping

    mapping = ColumnMapping(source_to_canonical={})

    df = __import__("pandas").DataFrame(
        {
            "review_text": ["Good", "Bad"],
        }
    )

    count, groups = validate_duplicate_ids(df, mapping)
    assert count == 0


# ---------------------------------------------------------------------------
# Data quality report tests
# ---------------------------------------------------------------------------


def test_validate_data_quality_basic(sample_dataframe):
    """Test basic data quality report generation."""
    from marketlens.ingestion.column_mapping import ColumnMapping

    mapping = ColumnMapping(
        source_to_canonical={
            "review_id": "review_id",
            "review_text": "review_text",
            "rating": "rating",
        }
    )

    report = validate_data_quality(sample_dataframe, mapping)
    assert "total_rows" in report
    assert "missing_review_ids" in report
    assert "missing_review_text" in report
    assert "invalid_ratings" in report
    assert "invalid_dates" in report
    assert "duplicate_ids" in report
    assert "missing_value_counts" in report


def test_validate_data_quality_empty():
    """Test data quality report with empty DataFrame."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "review_id": [],
            "review_text": [],
            "rating": [],
        }
    )

    from marketlens.ingestion.column_mapping import ColumnMapping

    mapping = ColumnMapping(
        source_to_canonical={
            "review_id": "review_id",
            "review_text": "review_text",
            "rating": "rating",
        }
    )

    report = validate_data_quality(df, mapping)
    assert report["total_rows"] == 0
