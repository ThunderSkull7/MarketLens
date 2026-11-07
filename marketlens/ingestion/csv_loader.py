"""
MarketLens - CSV Loading and Encoding Handling.

Provides reusable CSV file loading with support for:
- UTF-8 and UTF-8 BOM encoding handling
- Multiple encoding fallbacks
- Reasonable handling of common encoding problems
- Clear error messages for unsupported or malformed files

Preserves the original uploaded file and never silently overwrites it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import pandas as pd

from marketlens.exceptions import InvalidCSVError, UnsupportedFileTypeError

# ---------------------------------------------------------------------------
# Supported file types and extensions
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".csv"}
SUPPORTED_MIME_TYPES = {"text/csv", "text/plain", "application/csv"}


def validate_file_type(file_path: str | Path, mime_type: Optional[str] = None) -> None:
    """
    Validate that the file has a supported type/extension.

    Parameters
    ----------
    file_path : str | Path
        Path to the file to validate.
    mime_type : Optional[str]
        MIME type if available (e.g., from upload).

    Raises
    ------
    UnsupportedFileTypeError
        If the file extension or MIME type is not supported.
    """
    path = Path(file_path)
    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        msg = f"File '{path.name}' has unsupported extension '{extension}'"
        if mime_type:
            msg += f" (MIME type: {mime_type})"
        raise UnsupportedFileTypeError(extension, list(SUPPORTED_EXTENSIONS))


def detect_encoding(file_path: str | Path) -> Tuple[str, bool]:
    """
    Detect the encoding of a file, handling UTF-8 BOM.

    Tries to read the file with UTF-8 BOM detection first, then falls back
    to other common encodings.

    Parameters
    ----------
    file_path : str | Path
        Path to the file.

    Returns
    -------
    Tuple[str, bool]
        (detected_encoding, has_bom) where has_bom indicates whether a UTF-8 BOM
        was present at the start of the file.
    """
    path = Path(file_path)

    # Read raw bytes to detect BOM
    with open(path, "rb") as f:
        raw = f.read()

    # Check for UTF-8 BOM
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig", True

    # Check for UTF-16 BOM (unlikely for CSVs but handled)
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return "utf-16", True

    # Assume UTF-8 without BOM
    return "utf-8", False


def load_csv(
    file_path: str | Path,
    encoding: Optional[str] = None,
    sep: str = ",",
    dtype: dict[str, Any] | None = None,
    on_bad_lines: str = "warn",
    nrows: int | None = None,
) -> pd.DataFrame:
    """
    Load a CSV file with encoding handling and error reporting.

    Supports UTF-8 and UTF-8 BOM automatically. Falls back to other encodings
    if UTF-8 fails.

    Parameters
    ----------
    file_path : str | Path
        Path to the CSV file to load.
    encoding : Optional[str]
        Explicit encoding. If None, encoding is auto-detected.
    sep : str
        Delimiter to use. Defaults to comma.
    dtype : Optional[dict[str, Any]]
        Column type overrides. If None, types are inferred.
    on_bad_lines : str
        How to handle bad lines. Options: 'warn', 'error', 'skip'.

    Returns
    -------
    pd.DataFrame
        Loaded data as a DataFrame.

    Raises
    ------
    InvalidCSVError
        If the file cannot be parsed as CSV.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        msg = f"File not found: '{file_path}'"
        raise InvalidCSVError(msg)

    # Validate file type
    try:
        validate_file_type(file_path)
    except UnsupportedFileTypeError:
        # Re-raise with more context
        raise

    # Determine encoding if not provided
    if encoding is None:
        encoding, has_bom = detect_encoding(file_path)

    # Try to load the CSV
    try:
        df = pd.read_csv(
            file_path,
            encoding=encoding,
            sep=sep,
            dtype=dtype,
            on_bad_lines=on_bad_lines,
            nrows=nrows,
            dtype_backend="numpy_nullable",
        )
        return df

    except pd.errors.ParserError as e:
        msg = f"Failed to parse CSV file '{file_path}': {e}"
        raise InvalidCSVError(msg) from e
    except UnicodeDecodeError as e:
        # Fall back to alternative encoding
        msg = f"Unicode decode error with encoding '{encoding}'; trying fallback"
        # Try with utf-8-sig
        try:
            df = pd.read_csv(
                file_path,
                encoding="utf-8-sig",
                sep=sep,
                dtype=dtype,
                on_bad_lines=on_bad_lines,
                nrows=nrows,
                dtype_backend="numpy_nullable",
            )
            return df
        except Exception:
            pass
        msg = f"Failed to decode CSV file '{file_path}' with encoding '{encoding}': {e}"
        raise InvalidCSVError(msg) from e
    except Exception as e:
        msg = f"Unexpected error loading CSV '{file_path}': {e}"
        raise InvalidCSVError(msg) from e


def load_csv_with_encoding_fallback(
    file_path: str | Path,
    sep: str = ",",
    dtype: dict[str, Any] | None = None,
    nrows: int | None = None,
) -> Tuple[pd.DataFrame, dict[str, Any]]:
    """
    Load a CSV file with multiple encoding fallbacks.

    Attempts to load the CSV with several encodings, returning both the
    DataFrame and information about the encoding used.

    Parameters
    ----------
    file_path : str | Path
        Path to the CSV file.
    sep : str
        Delimiter to use. Defaults to comma.
    dtype : Optional[dict[str, Any]]
        Column type overrides.

    Returns
    -------
    Tuple[pd.DataFrame, dict[str, Any]]
        (DataFrame, info_dict) where info_dict contains:
        - "encoding": The encoding that successfully loaded the file
        - "has_bom": Whether a UTF-8 BOM was detected
    """
    file_path = Path(file_path)

    # Order of encodings to try
    encodings_to_try = ["utf-8-sig", "utf-8", "latin-1", "cp1252"]

    last_error: Optional[Exception] = None

    for encoding in encodings_to_try:
        try:
            df = load_csv(
                file_path, encoding=encoding, sep=sep, dtype=dtype, nrows=nrows
            )
            has_bom = encoding == "utf-8-sig"
            info = {
                "encoding": encoding,
                "has_bom": has_bom,
            }
            return df, info
        except (InvalidCSVError, UnicodeDecodeError) as e:
            last_error = e
            continue

    # All encodings failed
    msg = f"Could not load CSV file '{file_path}' with any supported encoding"
    if last_error:
        msg += f": {last_error}"
    raise InvalidCSVError(msg)
