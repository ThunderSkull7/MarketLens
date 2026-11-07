"""Unit tests for the CSV loading module."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from marketlens.exceptions import (
    InvalidCSVError,
    UnsupportedFileTypeError,
)
from marketlens.ingestion.csv_loader import (
    detect_encoding,
    load_csv,
    load_csv_with_encoding_fallback,
    validate_file_type,
)

# ---------------------------------------------------------------------------
# CSV loading tests
# ---------------------------------------------------------------------------


def test_valid_csv_loading(sample_dataframe):
    """Test that a valid CSV can be loaded successfully."""
    assert sample_dataframe is not None
    assert len(sample_dataframe) > 0


def test_csv_with_bom(temp_csv_bom):
    """Test UTF-8 BOM handling."""
    df = load_csv(temp_csv_bom)
    assert len(df) == 2  # 2 data rows (fixture writes BOM + 2 data rows)


def test_csv_with_bom_detect_encoding(temp_csv_bom):
    """Test encoding detection with BOM."""
    encoding, has_bom = detect_encoding(Path(temp_csv_bom))
    assert has_bom is True
    assert encoding == "utf-8-sig"


def test_csv_without_bom(temp_csv_path):
    """Test loading a CSV without BOM."""
    df = load_csv(temp_csv_path)
    assert len(df) == 3


def test_csv_encoding_fallback(temp_csv_path):
    """Test encoding fallback mechanism."""
    df, info = load_csv_with_encoding_fallback(temp_csv_path)
    assert len(df) > 0
    assert "encoding" in info


def test_unsupported_file_type():
    """Test that unsupported file types raise an error."""
    with pytest.raises(UnsupportedFileTypeError):
        validate_file_type("test.txt")


def test_unsupported_file_type_json():
    """Test that JSON file types are rejected."""
    with pytest.raises(UnsupportedFileTypeError):
        validate_file_type("test.json")


def test_missing_file():
    """Test that missing files raise an error."""
    try:
        load_csv(Path("nonexistent_file.csv"))
        pytest.fail("Expected InvalidCSVError to be raised")
    except InvalidCSVError:
        pass  # Expected


# ---------------------------------------------------------------------------
# Encoding fallback tests
# ---------------------------------------------------------------------------


def test_encoding_fallback_with_latin1():
    """Test encoding fallback with a Latin-1 encoded file."""
    import tempfile as _tempfile

    # Create a file with Latin-1 encoding
    with _tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as f:
        content = "review_id,review_text,rating\nR001,g\xf6tes,5.0\n".encode("latin-1")
        f.write(content)
        path = f.name
    try:
        df, info = load_csv_with_encoding_fallback(Path(path))
        # Should succeed with latin-1 fallback or at least not crash
        assert df is not None
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
