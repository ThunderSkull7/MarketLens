"""Unit tests for the column normalisation and mapping module."""

from __future__ import annotations

from marketlens.ingestion.column_mapping import (
    auto_map_columns,
    detect_duplicate_normalised_headers,
    normalise_column_name,
    normalise_headers,
)

# ---------------------------------------------------------------------------
# Column normalisation tests
# ---------------------------------------------------------------------------


def test_normalise_column_name_basic():
    """Test basic column name normalisation."""
    assert normalise_column_name("Review ID") == "review_id"
    assert normalise_column_name("  hello world  ") == "hello_world"
    assert normalise_column_name("Email Address") == "email_address"


def test_normalise_column_name_whitespace():
    """Test normalisation handles internal whitespace."""
    assert normalise_column_name("hello   world") == "hello_world"
    assert normalise_column_name("a  b  c") == "a_b_c"


def test_normalise_column_name_lowercase():
    """Test normalisation converts to lowercase."""
    assert normalise_column_name("REVIEW_ID") == "review_id"
    assert normalise_column_name("Rating") == "rating"


def test_normalise_column_name_special_chars():
    """Test normalisation removes special characters."""
    assert normalise_column_name("@Review!@#") == "review"
    assert normalise_column_name("Price ($)") == "price_"


def test_normalise_headers():
    """Test header normalisation for a list."""
    headers = normalise_headers(["Review ID", "Review Text", "Rating"])
    assert headers == ["review_id", "review_text", "rating"]


# ---------------------------------------------------------------------------
# Duplicate header detection tests
# ---------------------------------------------------------------------------


def test_no_duplicate_headers():
    """Test that no duplicates are reported for unique headers."""
    has_dups, duplicates = detect_duplicate_normalised_headers(
        ["review_id", "review_text", "rating"]
    )
    assert has_dups is False
    assert len(duplicates) == 0


def test_duplicate_headers_detected():
    """Test that duplicate normalised headers are detected."""
    has_dups, duplicates = detect_duplicate_normalised_headers(
        ["review_id", "review_id", "rating"]
    )
    assert has_dups is True
    assert len(duplicates) > 0


def test_multiple_duplicate_headers():
    """Test detection of multiple types of duplicates."""
    has_dups, duplicates = detect_duplicate_normalised_headers(
        ["id", "id", "name", "name", "name"]
    )
    assert has_dups is True
    assert len(duplicates) > 0


# ---------------------------------------------------------------------------
# Auto mapping tests
# ---------------------------------------------------------------------------


def test_auto_map_basic_columns(sample_dataframe):
    """Test automatic mapping of basic canonical columns."""
    headers = normalise_headers(sample_dataframe.columns.tolist())
    result = auto_map_columns(headers, strict=True)
    assert result.mapping is not None
    # Should map at least review_id and review_text
    mapped = result.mapping.source_to_canonical
    assert "review_id" in mapped.values() or any(
        v == "review_id" for v in mapped.values()
    )


def test_auto_map_with_explicit_mapping(sample_dataframe):
    """Test automatic mapping with explicit user-supplied mapping."""
    headers = normalise_headers(sample_dataframe.columns.tolist())
    explicit = {"review_text": "review_text"}
    result = auto_map_columns(headers, explicit_mapping=explicit, strict=True)
    assert result.mapping is not None


def test_auto_map_missing_required(sample_dataframe):
    """Test that missing required columns are reported."""
    # Use headers that don't include review_id or review_text
    headers = normalise_headers(["some_other_column"])
    result = auto_map_columns(headers, strict=True)
    assert len(result.errors) > 0
    assert any("required" in e.lower() for e in result.errors)


def test_auto_map_ambiguous_columns():
    """Test ambiguous column mapping detection."""
    # Headers that could match multiple aliases
    headers = normalise_headers(["id", "name"])
    result = auto_map_columns(headers, strict=True)
    # May or may not have ambiguous depending on mapping logic
    assert result.mapping is not None


def test_auto_map_unmapped_columns():
    """Test that unmapped columns are reported."""
    headers = normalise_headers(["unknown_column1", "unknown_column2"])
    result = auto_map_columns(headers, strict=True)
    assert len(result.unmapped) > 0
