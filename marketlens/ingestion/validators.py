"""
MarketLens - Data Validation.

Provides schema and data validation for ingested CSV data.

Validates:
- Required columns presence
- Empty datasets
- Missing review IDs and review text
- Invalid rating values (configurable scale)
- Invalid dates
- Duplicate review IDs
- Data quality metrics

Distinguishes between:
1. Accepted records (pass all validation)
2. Rejected records (fail mandatory validation)
3. Flagged records (quality issues requiring attention)

Does not silently remove questionable records.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import pandas as pd

from marketlens.ingestion.column_mapping import CanonicalField

# ---------------------------------------------------------------------------
# Rating scale configurations
# ---------------------------------------------------------------------------

# Common rating scale configurations documented by dataset
RATING_SCALES: dict[str, dict[str, Any]] = {
    "1-5": {
        "min": 1.0,
        "max": 5.0,
        "description": "1 to 5 star rating (most common)",
    },
    "1-10": {
        "min": 1.0,
        "max": 10.0,
        "description": "1 to 10 rating scale",
    },
    "1-3": {
        "min": 1.0,
        "max": 3.0,
        "description": "1 to 3 rating scale",
    },
    "5-star": {
        "min": 1.0,
        "max": 5.0,
        "description": "5-star rating scale",
    },
    "thumbs": {
        "min": 0.0,
        "max": 1.0,
        "description": "Thumbs up/down (0/1 or -1/1)",
    },
}


def validate_rating_scale(
    scale: str | None,
    min_val: float,
    max_val: float,
) -> Tuple[float, float]:
    """
    Validate that rating values fall within the expected scale.

    If scale is None, uses documented defaults. Otherwise, looks up the
    configured scale and validates min/max.

    Parameters
    ----------
    scale : Optional[str]
        Named rating scale (e.g., "1-5", "1-10"). If None, uses
        configurable defaults.
    min_val : float
        Observed minimum value in the data.
    max_val : float
        Observed maximum value in the data.

    Returns
    -------
    Tuple[float, float]
        (valid_min, valid_max) the acceptable range after validation.
    """
    if scale is None:
        # Default to 1-5 if not specified
        return 1.0, 5.0

    scale_lower = scale.lower().replace(" ", "")
    if scale_lower in RATING_SCALES:
        config = RATING_SCALES[scale_lower]
        return float(config["min"]), float(config["max"])

    # Unknown scale; return defaults and warn
    return 1.0, 5.0


# ---------------------------------------------------------------------------
# Validation result types
# ---------------------------------------------------------------------------


class ValidationStatus:
    """Status of a record during validation."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    FLAGGED = "flagged"


class RecordValidationResult:
    """
    Result of validating a single record.

    Attributes
    ----------
    status : str
        One of ValidationStatus.ACCEPTED, .REJECTED, .FLAGGED
    reasons : list[str]
        Human-readable reasons for the status
    field_issues : dict[str, str]
        Per-field issue descriptions if any
    """

    def __init__(
        self,
        status: str,
        reasons: list[str],
        field_issues: dict[str, str] | None = None,
    ) -> None:
        self.status = status
        self.reasons = reasons
        self.field_issues = field_issues or {}

    def is_accepted(self) -> bool:
        return self.status == ValidationStatus.ACCEPTED

    def is_rejected(self) -> bool:
        return self.status == ValidationStatus.REJECTED

    def is_flagged(self) -> bool:
        return self.status == ValidationStatus.FLAGGED


# ---------------------------------------------------------------------------
# Core validation functions
# ---------------------------------------------------------------------------


def validate_required_columns(
    df: pd.DataFrame,
    canonical_fields: Any,
    mapping_result: Any,
) -> Tuple[bool, list[str]]:
    """
    Validate that all required canonical columns are present in the mapping.

    Parameters
    ----------
    df : pd.DataFrame
        The loaded DataFrame.
    canonical_fields : Any
        Canonical fields definition.
    mapping_result : Any
        The column mapping result.

    Returns
    -------
    Tuple[bool, list[str]]
        (all_present, missing_columns)
    """
    missing: list[str] = []

    # Ensure iter_canonical_fields is available
    from marketlens.ingestion.column_mapping import iter_canonical_fields

    # Determine canonical fields to check: fall back to built-in fields
    # if none supplied.
    if canonical_fields is None:
        fields = iter_canonical_fields()
    else:
        # Collect CanonicalField values from both instance and class
        # attributes (fields may be class-level).
        collected: list[tuple[str, CanonicalField]] = []
        seen_ids: set[int] = set()
        for source_dict in (
            getattr(type(canonical_fields), "__dict__", {}),
            getattr(canonical_fields, "__dict__", {}),
        ):
            for name, value in source_dict.items():
                if isinstance(value, CanonicalField) and id(value) not in seen_ids:
                    seen_ids.add(id(value))
                    collected.append((name, value))
        if collected:
            fields = collected
        else:
            fields = iter_canonical_fields()

    # Resolve mapped canonical names: mapping_result may be a
    # ColumnMapping (has source_to_canonical) or a ColumnMappingResult
    # (has .mapping.source_to_canonical).
    if hasattr(mapping_result, "source_to_canonical"):
        mapped_canonicals = set(mapping_result.source_to_canonical.values())
    elif hasattr(mapping_result, "mapping"):
        mapped_canonicals = set(mapping_result.mapping.source_to_canonical.values())
    else:
        mapped_canonicals = set()

    # Check which required fields are mapped
    for _name, field in fields:
        if field.is_required:
            if field.name not in mapped_canonicals and field.name not in df.columns:
                missing.append(field.name)

    return len(missing) == 0, missing


def _get_source_column(mapping_result: Any, canonical: str):
    """Return the source column mapped to ``canonical``.

    Accepts either a :class:`ColumnMapping` object (exposes
    ``source_to_canonical``) or a :class:`ColumnMappingResult` (exposes
    ``mapping``).
    """
    if hasattr(mapping_result, "source_to_canonical"):
        return mapping_result.source_to_canonical.get(canonical)
    return mapping_result.get(canonical)


def validate_missing_review_ids(
    df: pd.DataFrame, mapping_result: Any
) -> Tuple[int, list[int]]:
    """
    Validate review IDs are present and not null.

    Parameters
    ----------
    df : pd.DataFrame
        The loaded DataFrame.
    mapping_result : Any
        The column mapping result (ColumnMapping object).

    Returns
    -------
    Tuple[int, list[int]]
        (count_of_missing_ids, list_of_row_indices_missing_id)
    """
    # Find the source column that maps to review_id
    # mapping_result is a ColumnMapping object with source_to_canonical dict
    source_col = _get_source_column(mapping_result, "review_id")

    if source_col is None or source_col not in df.columns:
        # No review_id column at all - all rows are missing IDs
        return len(df), list(df.index)

    ids = df[source_col]
    missing_indices: list[int] = []

    for idx in df.index:
        val = ids.loc[idx]
        # Consider NaN, None, empty string as missing
        if pd.isna(val) or val is None or (isinstance(val, str) and val.strip() == ""):
            missing_indices.append(idx)

    return len(missing_indices), missing_indices


def validate_missing_review_text(
    df: pd.DataFrame, mapping_result: Any
) -> Tuple[int, list[int]]:
    """
    Validate review text is present and not empty.

    Parameters
    ----------
    df : pd.DataFrame
        The loaded DataFrame.
    mapping_result : Any
        The column mapping result (ColumnMapping object).

    Returns
    -------
    Tuple[int, list[int]]
        (count_of_missing_text, list_of_row_indices_missing_text)
    """
    # Find the source column that maps to review_text
    # mapping_result is a ColumnMapping object with source_to_canonical dict
    source_col = _get_source_column(mapping_result, "review_text")

    if source_col is None or source_col not in df.columns:
        # No review_text column at all
        return len(df), list(df.index)

    text_col = df[source_col]
    missing_indices: list[int] = []

    for idx in df.index:
        val = text_col.loc[idx]
        # Consider NaN, None, empty string (after strip) as missing
        if pd.isna(val) or val is None:
            missing_indices.append(idx)
        elif isinstance(val, str) and val.strip() == "":
            missing_indices.append(idx)
        elif isinstance(val, (int, float)) and np.isnan(val):
            missing_indices.append(idx)

    return len(missing_indices), missing_indices


def validate_ratings(
    df: pd.DataFrame,
    mapping_result: Any,
    scale: str | None = None,
) -> Tuple[int, list[int], list[int]]:
    """
    Validate rating values are within the expected scale.

    Parameters
    ----------
    df : pd.DataFrame
        The loaded DataFrame.
    mapping_result : Any
        The column mapping result (ColumnMapping object).
    scale : Optional[str]
        Rating scale name (e.g., "1-5", "1-10"). If None, uses default
        1-5 scale.

    Returns
    -------
    Tuple[int, list[int], list[int]]
        (count_invalid, list_of_invalid_row_indices, list_of_out_of_range_indices)
    """
    # Find the source column that maps to rating
    # mapping_result is a ColumnMapping object with source_to_canonical dict
    source_col = _get_source_column(mapping_result, "rating")

    if source_col is None or source_col not in df.columns:
        # No rating column - not an error, just not present
        return 0, [], []

    # Get the rating scale boundaries
    min_allowed, max_allowed = validate_rating_scale(scale, 1.0, 5.0)

    ratings = pd.to_numeric(df[source_col], errors="coerce")
    invalid_indices: list[int] = []
    out_of_range_indices: list[int] = []

    for idx in df.index:
        val = ratings.loc[idx]
        # Check for non-numeric values (coerced to NaN)
        if pd.isna(val):
            # If the original value was non-numeric, flag it
            original_val = df[source_col].loc[idx]
            if not (isinstance(original_val, (int, float)) or original_val is None):
                # Non-numeric value where a number is expected
                invalid_indices.append(idx)
            continue

        # Check range
        if val < min_allowed or val > max_allowed:
            out_of_range_indices.append(idx)
        else:
            # Valid rating
            pass

    # Also include rows with non-numeric values as invalid (unless they were
    # already counted in out_of_range)
    # Actually, we already handled non-numeric above; just make sure we don't
    # double-count
    all_invalid = set(invalid_indices) | set(out_of_range_indices)

    return len(all_invalid), list(all_invalid), list(out_of_range_indices)


def validate_dates(
    df: pd.DataFrame,
    mapping_result: Any,
) -> Tuple[int, list[int]]:
    """
    Validate that created_at timestamps can be parsed as dates.

    Parameters
    ----------
    df : pd.DataFrame
        The loaded DataFrame.
    mapping_result : Any
        The column mapping result (ColumnMapping object).

    Returns
    -------
    Tuple[int, list[int]]
        (count_of_invalid_dates, list_of_row_indices_with_invalid_dates)
    """
    # Find the source column that maps to created_at
    source_col = _get_source_column(mapping_result, "created_at")

    if source_col is None or source_col not in df.columns:
        # No created_at column - not an error, just not present
        return 0, []

    # Try to parse dates
    date_col = pd.to_datetime(df[source_col], errors="coerce")
    invalid_mask = date_col.isna()

    # Filter out NaT that were originally valid strings that just failed parsing
    invalid_indices: list[int] = []
    for idx in df.index:
        if invalid_mask.loc[idx]:
            # Check if the original value was actually a valid date string
            # that just had a weird format - still count as invalid if NaT
            invalid_indices.append(idx)

    return len(invalid_indices), invalid_indices


def validate_duplicate_ids(
    df: pd.DataFrame, mapping_result: Any
) -> Tuple[int, list[tuple]]:
    """
    Validate there are no duplicate review IDs.

    Parameters
    ----------
    df : pd.DataFrame
        The loaded DataFrame.
    mapping_result : Any
        The column mapping result (ColumnMapping object).

    Returns
    -------
    Tuple[int, list[tuple]]
        (count_of_duplicate_ids, list_of_duplicate_groups)
        Each group is (id_value, [list_of_row_indices])
    """
    # Find the source column that maps to review_id
    source_col = _get_source_column(mapping_result, "review_id")

    if source_col is None or source_col not in df.columns:
        # No review_id column
        return 0, []

    ids = df[source_col].astype(str)

    # Find duplicates
    duplicate_mask = ids.duplicated(keep=False)
    duplicate_ids = ids[duplicate_mask]

    if len(duplicate_ids) == 0:
        return 0, []

    # Group by the duplicate ID value
    groups: dict[str, list[int]] = {}
    for idx in df.index[duplicate_mask]:
        id_val = str(ids.loc[idx])
        if id_val not in groups:
            groups[id_val] = []
        groups[id_val].append(idx)

    duplicate_groups: list[tuple] = []
    for id_val, indices in groups.items():
        duplicate_groups.append((id_val, sorted(indices)))

    return len(duplicate_groups), duplicate_groups


def validate_data_quality(
    df: pd.DataFrame,
    mapping_result: Any,
    rating_scale: str | None = None,
) -> dict[str, Any]:
    """
    Generate a comprehensive data quality report.

    Parameters
    ----------
    df : pd.DataFrame
        The loaded DataFrame.
    mapping_result : Any
        The column mapping result.
    rating_scale : Optional[str]
        Rating scale for validation.

    Returns
    -------
    dict[str, Any]
        Dictionary containing quality metrics:
        - total_rows: int
        - missing_review_ids: int
        - missing_review_text: int
        - invalid_ratings: int
        - invalid_dates: int
        - duplicate_ids: int
        - missing_value_counts: dict
        - per-column metrics
    """
    total_rows = len(df)

    # Missing review IDs
    missing_ids_count, _ = validate_missing_review_ids(df, mapping_result)

    # Missing review text
    missing_text_count, _ = validate_missing_review_text(df, mapping_result)

    # Invalid ratings
    invalid_ratings_count, _, _ = validate_ratings(df, mapping_result, rating_scale)

    # Invalid dates
    invalid_dates_count, _ = validate_dates(df, mapping_result)

    # Duplicate IDs
    duplicate_ids_count, _ = validate_duplicate_ids(df, mapping_result)

    # Missing value counts per column
    missing_value_counts: dict[str, int] = {}
    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        if missing_count > 0:
            missing_value_counts[col] = missing_count

    return {
        "total_rows": total_rows,
        "missing_review_ids": missing_ids_count,
        "missing_review_text": missing_text_count,
        "invalid_ratings": invalid_ratings_count,
        "invalid_dates": invalid_dates_count,
        "duplicate_ids": duplicate_ids_count,
        "missing_value_counts": missing_value_counts,
    }
