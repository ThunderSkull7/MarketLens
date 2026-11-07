"""Pytest configuration and fixtures for MarketLens."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from marketlens.ingestion.column_mapping import (
    auto_map_columns,
    normalise_headers,
)
from marketlens.ingestion.csv_loader import load_csv
from marketlens.src import load_settings

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_dataset_path():
    """Path to the synthetic demonstration dataset."""
    return Path("data/raw/synthetic_feedback_demo.csv")


@pytest.fixture
def sample_dataframe(synthetic_dataset_path):
    """DataFrame loaded from the synthetic dataset."""
    return load_csv(synthetic_dataset_path)


@pytest.fixture
def mapped_result(sample_dataframe):
    """Column mapping result from the sample data."""
    headers = normalise_headers(sample_dataframe.columns.tolist())
    return auto_map_columns(headers, strict=True)


@pytest.fixture
def settings():
    """Project settings instance."""
    return load_settings()


@pytest.fixture
def temp_csv_path():
    """Create a temporary CSV file for testing."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    ) as f:
        f.write("review_id,review_text,rating\n")
        f.write("R001,Great product,5.0\n")
        f.write("R002,OK product,3.0\n")
        f.write("R003,Poor quality,2.0\n")
        path = f.name
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
def temp_csv_bom():
    """Create a temporary CSV file with UTF-8 BOM."""
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as f:
        # Write UTF-8 BOM
        f.write(b"\xef\xbb\xbf")
        # Write CSV content
        f.write("review_id,review_text,rating\n".encode("utf-8"))
        f.write("R001,Great product,5.0\n".encode("utf-8"))
        f.write("R002,OK product,3.0\n".encode("utf-8"))
        path = f.name
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
def temp_csv_with_duplicates():
    """Create a temporary CSV with duplicate normalised headers."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    ) as f:
        # Normalises to: review_id, review_text, rating
        f.write("Review ID,Review Text,Rating\n")
        f.write("R001,Great product,5.0\n")
        f.write("R002,OK product,3.0\n")
        path = f.name
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
def temp_csv_missing_required():
    """Create a temporary CSV missing required columns."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    ) as f:
        f.write("some_other_column,other_value\n")
        f.write("foo,bar\n")
        path = f.name
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
def temp_csv_invalid_ratings():
    """Create a temporary CSV with invalid ratings."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    ) as f:
        f.write("review_id,review_text,rating\n")
        f.write("R001,Great product,5.0\n")
        f.write("R002,OK product,10.0\n")  # Outside 1-5 range
        f.write("R003,Poor quality,not_a_number\n")  # Non-numeric
        f.write("R004,Okay product,3.0\n")
        path = f.name
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
def temp_csv_duplicate_ids():
    """Create a temporary CSV with duplicate review IDs."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    ) as f:
        f.write("review_id,review_text,rating\n")
        f.write("R001,Great product,5.0\n")
        f.write("R001,Duplicate review,3.0\n")  # Duplicate ID
        f.write("R002,OK product,3.0\n")
        path = f.name
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
def temp_csv_empty():
    """Create a temporary empty CSV."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    ) as f:
        f.write("\n")  # Just a header or empty
        path = f.name
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def assert_ingestion_result_has(
    result: object,
    expected_accepted: int | None = None,
    expected_rejected: int | None = None,
    expected_flagged: int | None = None,
    expected_total: int | None = None,
) -> None:
    """Assert common ingestion result properties."""
    if expected_accepted is not None:
        assert result.accepted_count == expected_accepted, (
            f"Expected {expected_accepted} accepted, got {result.accepted_count}"
        )
    if expected_rejected is not None:
        assert result.rejected_count == expected_rejected, (
            f"Expected {expected_rejected} rejected, got {result.rejected_count}"
        )
    if expected_flagged is not None:
        assert result.flagged_count == expected_flagged, (
            f"Expected {expected_flagged} flagged, got {result.flagged_count}"
        )
    if expected_total is not None:
        assert result.total_rows == expected_total, (
            f"Expected {expected_total} total rows, got {result.total_rows}"
        )
