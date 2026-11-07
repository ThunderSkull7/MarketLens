"""
MarketLens - Ingestion Module.

Top-level ingestion pipeline orchestrating CSV loading, column normalisation,
mapping, and validation. Provides a unified interface for processing
customer feedback datasets.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from marketlens.config import settings
from marketlens.exceptions import (
    AmbiguousColumnMappingError as AmbiguousColumnMappingError,
)
from marketlens.exceptions import (
    DataValidationError as DataValidationError,
)
from marketlens.exceptions import (
    FileSizeLimitExceededError,
    InvalidCSVError,
    MissingRequiredColumnsError,
    UnsupportedFileTypeError,
)
from marketlens.ingestion.column_mapping import (
    ColumnMappingResult,
    auto_map_columns,
    detect_duplicate_normalised_headers,
    normalise_column_name,
    normalise_headers,
)
from marketlens.ingestion.csv_loader import (
    load_csv as load_csv,
)
from marketlens.ingestion.csv_loader import (
    load_csv_with_encoding_fallback,
)
from marketlens.ingestion.validators import (
    validate_data_quality,
    validate_required_columns,
)
from marketlens.ingestion.validators import (
    validate_dates as validate_dates,
)
from marketlens.ingestion.validators import (
    validate_duplicate_ids as validate_duplicate_ids,
)
from marketlens.ingestion.validators import (
    validate_missing_review_ids as validate_missing_review_ids,
)
from marketlens.ingestion.validators import (
    validate_missing_review_text as validate_missing_review_text,
)
from marketlens.ingestion.validators import (
    validate_ratings as validate_ratings,
)

__all__ = [
    "AmbiguousColumnMappingError",
    "DataValidationError",
    "IngestionRecord",
    "IngestionResult",
    "ingest",
    "load_csv",
    "validate_dates",
    "validate_duplicate_ids",
    "validate_missing_review_ids",
    "validate_missing_review_text",
    "validate_ratings",
]

logger = logging.getLogger("marketlens.ingestion")


# ---------------------------------------------------------------------------
# Ingestion result structure
# ---------------------------------------------------------------------------


class IngestionRecord:
    """
    Represents a single record through the ingestion pipeline.

    Attributes
    ----------
    row_index : int
        The original row index in the DataFrame.
    row_data : dict
        The row data as a dictionary.
    status : str
        One of "accepted", "rejected", "flagged".
    reasons : list[str]
        Reasons for the status assignment.
    mapped_values : dict
        Mapped canonical field values for this record.
    """

    def __init__(
        self,
        row_index: int,
        row_data: dict[str, Any],
        status: str,
        reasons: list[str],
        mapped_values: dict[str, Any] | None = None,
    ) -> None:
        self.row_index = row_index
        self.row_data = row_data
        self.status = status
        self.reasons = reasons
        self.mapped_values = mapped_values or {}

    def is_accepted(self) -> bool:
        return self.status == "accepted"

    def is_rejected(self) -> bool:
        return self.status == "rejected"

    def is_flagged(self) -> bool:
        return self.status == "flagged"


class IngestionResult:
    """
    The result of a complete ingestion operation.

    Attributes
    ----------
    accepted : list[IngestionRecord]
        Records that passed all validation.
    rejected : list[IngestionRecord]
        Records that failed mandatory validation.
    flagged : list[IngestionRecord]
        Records flagged for attention (quality issues).
    df_validated : pd.DataFrame
        The validated DataFrame with only accepted records.
    mapping_result : ColumnMappingResult
        The column mapping that was applied.
    quality_report : dict[str, Any]
        Structured data quality report.
    warnings : list[str]
        Warnings generated during ingestion.
    errors : list[str]
        Errors generated during ingestion.
    raw_filename : str
        The original filename that was ingested.
    """

    def __init__(
        self,
        accepted: list[IngestionRecord],
        rejected: list[IngestionRecord],
        flagged: list[IngestionRecord],
        df_validated: pd.DataFrame,
        mapping_result: ColumnMappingResult,
        quality_report: dict[str, Any],
        warnings: list[str],
        errors: list[str],
        raw_filename: str,
    ) -> None:
        self.accepted = accepted
        self.rejected = rejected
        self.flagged = flagged
        self.df_validated = df_validated
        self.mapping_result = mapping_result
        self.quality_report = quality_report
        self.warnings = warnings
        self.errors = errors
        self.raw_filename = raw_filename

    @property
    def accepted_count(self) -> int:
        return len(self.accepted)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)

    @property
    def flagged_count(self) -> int:
        return len(self.flagged)

    @property
    def total_rows(self) -> int:
        return self.quality_report.get("total_rows", 0)

    @property
    def total_source_rows(self) -> Optional[int]:
        """Total rows in the source file, when reliably measurable."""
        return self.quality_report.get("total_source_rows")

    @property
    def rows_parsed(self) -> int:
        """Rows actually parsed from the CSV."""
        return self.quality_report.get("rows_parsed", self.total_rows)

    @property
    def rows_omitted_limit(self) -> int:
        """Rows omitted due to configured max_rows limits."""
        return self.quality_report.get("rows_omitted_limit", 0)

    @property
    def row_accounting(self) -> dict[str, Any]:
        """Comprehensive row-accounting breakdown."""
        return self.quality_report.get(
            "row_accounting",
            {
                "total_source_rows": self.total_source_rows,
                "rows_parsed": self.rows_parsed,
                "rows_omitted_limit": self.rows_omitted_limit,
                "rows_rejected": self.rejected_count,
                "rows_accepted": self.accepted_count,
                "rows_flagged": self.flagged_count,
            },
        )

    @property
    def mapping_info(self) -> dict[str, Any]:
        """Column mapping metadata stored in the quality report."""
        return self.quality_report.get("mapping_info", {})

    def summary(self) -> str:
        """Return a human-readable summary of the ingestion result."""
        lines = [
            f"=== Ingestion Summary for: {self.raw_filename} ===",
            f"Total rows detected: {self.total_rows}",
        ]
        if (
            self.total_source_rows is not None
            and self.total_source_rows != self.rows_parsed
        ):
            lines.append(f"Total rows in source file: {self.total_source_rows}")
            lines.append(f"Rows parsed: {self.rows_parsed}")
            lines.append(f"Rows omitted due to limit: {self.rows_omitted_limit}")
        lines.extend(
            [
                f"Accepted: {self.accepted_count}",
                f"Rejected: {self.rejected_count}",
                f"Flagged: {self.flagged_count}",
                "",
                f"Warnings: {len(self.warnings)}",
                f"Errors: {len(self.errors)}",
            ]
        )
        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        if self.errors:
            lines.append("")
            lines.append("Errors:")
            for e in self.errors:
                lines.append(f"  - {e}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ingestion pipeline
# ---------------------------------------------------------------------------


def ingest(
    file_path: Union[str, Path],
    explicit_mapping: Optional[dict[str, str]] = None,
    rating_scale: Optional[str] = None,
    max_rows: Optional[int] = None,
    max_size_mb: Optional[float] = None,
    **kwargs: Any,
) -> IngestionResult:
    """
    Run the complete ingestion pipeline on a CSV file.

    Pipeline steps:
    1. File type and size validation
    2. CSV loading with encoding handling
    3. Column header normalisation
    4. Duplicate normalised header detection
    5. Automatic (or explicit) column mapping
    6. Data validation (required columns, missing IDs, text, ratings, dates, duplicates)
    7. Record-level classification (accepted/rejected/flagged)
    8. Quality report generation

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the CSV file to ingest.
    explicit_mapping : Optional[dict[str, str]]
        User-supplied explicit mapping {source_column: canonical_field}.
        If None, automatic mapping is attempted.
    rating_scale : Optional[str]
        Rating scale for validation (e.g., "1-5", "1-10").
        If None, defaults to 1-5.
    max_rows : Optional[int]
        Maximum number of rows to read from the CSV.
    max_size_mb : Optional[float]
        Maximum file size in megabytes. Uses config default if None.
    **kwargs :
        Additional arguments passed to pd.read_csv.

    Returns
    -------
    IngestionResult
        The complete ingestion result with accepted/rejected/flagged records
        and a quality report.

    Raises
    ------
    UnsupportedFileTypeError
        If the file has an unsupported extension.
    InvalidCSVError
        If the CSV cannot be parsed.
    MissingRequiredColumnsError
        If required canonical columns are missing.
    AmbiguousColumnMappingError
        If column mapping is ambiguous.
    FileSizeLimitExceededError
        If the file exceeds the size limit.
    """
    file_path = Path(file_path)

    warnings: list[str] = []
    errors: list[str] = []

    # --- Step 1: File type and size validation ---
    max_size = max_size_mb or settings.max_upload_size_mb

    # Check file size
    if file_path.exists():
        file_size_bytes = file_path.stat().st_size
        file_size_mb = file_size_bytes / (1024 * 1024)
        if file_size_mb > max_size:
            errors.append(
                f"File size ({file_size_mb:.2f} MB) exceeds maximum limit of "
                f"{max_size:.2f} MB"
            )
            raise FileSizeLimitExceededError(file_size_mb, max_size)

    # Validate file type
    try:
        from marketlens.ingestion.csv_loader import validate_file_type

        validate_file_type(file_path)
    except UnsupportedFileTypeError as e:
        errors.append(str(e))
        raise

    logger.info(f"Starting ingestion of: {file_path.name}")

    # --- Step 2: CSV loading ---
    try:
        df, encoding_info = load_csv_with_encoding_fallback(
            file_path, nrows=max_rows, **kwargs
        )
    except InvalidCSVError as e:
        errors.append(str(e))
        raise

    rows_parsed = len(df)
    total_source_rows: Optional[int] = None

    if max_rows is None:
        total_source_rows = rows_parsed
    elif file_path.is_file():
        try:
            enc = encoding_info.get("encoding", "utf-8")
            with open(file_path, "r", encoding=enc, errors="replace") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header is not None:
                    total_source_rows = sum(1 for _ in reader)
        except Exception:
            total_source_rows = None

    rows_omitted_limit = 0
    if total_source_rows is not None and max_rows is not None:
        rows_omitted_limit = max(0, total_source_rows - rows_parsed)

    logger.info(
        "Loaded %d rows from %s (encoding: %s, total_source_rows: %s)",
        rows_parsed,
        file_path.name,
        encoding_info.get("encoding"),
        total_source_rows,
    )

    # --- Step 3: Column header normalisation ---
    normalised_headers = normalise_headers(df.columns.tolist())

    # Check for duplicate normalised headers
    has_duplicates, duplicate_info = detect_duplicate_normalised_headers(
        normalised_headers
    )
    if has_duplicates:
        for row1, row2, header in duplicate_info[:5]:  # Show first 5
            warnings.append(
                f"Duplicate normalised header '{header}' at columns "
                f"({row1 + 1}) and ({row2 + 1})"
            )
        # Don't fail; just warn. The mapping will handle it.

    # --- Step 4: Automatic or explicit column mapping ---
    if explicit_mapping:
        # Apply explicit mapping, then auto-fill remaining
        explicit_norm = {
            normalise_column_name(k): normalise_column_name(v)
            for k, v in explicit_mapping.items()
        }
        mapping_result = auto_map_columns(
            normalised_headers,
            explicit_mapping=explicit_norm,
            strict=True,
        )
    else:
        mapping_result = auto_map_columns(normalised_headers, strict=True)

    # Check for ambiguous mappings
    if mapping_result.ambiguous:
        for col in mapping_result.ambiguous[:10]:
            warnings.append(f"Ambiguous column mapping skipped for: {col}")

    # Check for missing required columns
    required_present, missing_cols = validate_required_columns(
        df, canonical_fields=None, mapping_result=mapping_result
    )
    if not required_present:
        errors.append(f"Missing required columns: {', '.join(missing_cols)}")
        raise MissingRequiredColumnsError(missing_cols, [])  # type: ignore[arg-type]

    logger.info(
        f"Column mapping: {len(mapping_result.mapping.source_to_canonical)} "
        f"columns mapped, {len(mapping_result.unmapped)} unmapped, "
        f"{len(mapping_result.ambiguous)} ambiguous"
    )

    # --- Step 5: Record-level validation and classification ---
    # Get mapping source columns
    review_id_col = mapping_result.mapping.get("review_id")
    review_text_col = mapping_result.mapping.get("review_text")
    rating_col = mapping_result.mapping.get("rating")
    created_at_col = mapping_result.mapping.get("created_at")

    # Classify each record
    accepted: list[IngestionRecord] = []
    rejected: list[IngestionRecord] = []
    flagged: list[IngestionRecord] = []

    # Track duplicate IDs for classification
    source_id_col = mapping_result.mapping.get("review_id")
    id_values: list[str] = []
    if source_id_col and source_id_col in df.columns:
        id_values = df[source_id_col].astype(str).tolist()
    else:
        # No ID column - assign synthetic indices
        id_values = [f"ROW_{i}" for i in range(len(df))]

    # Track which IDs we've seen to detect duplicates
    seen_ids: dict[str, list[int]] = {}
    for idx, id_val in enumerate(id_values):
        if id_val not in seen_ids:
            seen_ids[id_val] = []
        seen_ids[id_val].append(idx)

    # Build a set of IDs that are duplicates
    duplicate_id_set: set[str] = set()
    for id_val, indices in seen_ids.items():
        if len(indices) > 1:
            duplicate_id_set.add(id_val)

    # Process each row
    for idx in df.index:
        row_data = df.loc[idx].to_dict()
        row_reasons: list[str] = []
        row_status = "accepted"
        mapped_values: dict[str, Any] = {}

        # --- Map canonical fields ---
        # review_id
        if review_id_col and review_id_col in row_data:
            rid = row_data[review_id_col]
            mapped_values["review_id"] = rid if pd.notna(rid) else f"MISSING_{idx}"
        else:
            mapped_values["review_id"] = f"MISSING_{idx}"
            row_reasons.append("Missing review_id column data")

        # review_text
        if review_text_col and review_text_col in row_data:
            rt = row_data[review_text_col]
            mapped_values["review_text"] = rt if pd.notna(rt) else ""
        else:
            mapped_values["review_text"] = ""
            row_reasons.append("Missing review_text column data")

        # rating
        if rating_col and rating_col in row_data:
            rating_val = row_data[rating_col]
            if pd.notna(rating_val):
                mapped_values["rating"] = float(rating_val)
            else:
                mapped_values["rating"] = None
        else:
            mapped_values["rating"] = None

        # created_at
        if created_at_col and created_at_col in row_data:
            ca_val = row_data[created_at_col]
            if pd.notna(ca_val):
                try:
                    mapped_values["created_at"] = str(pd.to_datetime(ca_val).date())
                except Exception:
                    mapped_values["created_at"] = str(ca_val)
                    row_reasons.append(f"Invalid date at row {idx}")
            else:
                mapped_values["created_at"] = None
        else:
            mapped_values["created_at"] = None

        # product_or_service
        product_col = mapping_result.mapping.get("product_or_service")
        if product_col and product_col in row_data:
            mapped_values["product_or_service"] = row_data[product_col]
        else:
            mapped_values["product_or_service"] = None

        # source
        source_col = mapping_result.mapping.get("source")
        if source_col and source_col in row_data:
            mapped_values["source"] = row_data[source_col]
        else:
            mapped_values["source"] = None

        # language
        language_col = mapping_result.mapping.get("language")
        if language_col and language_col in row_data:
            mapped_values["language"] = row_data[language_col]
        else:
            mapped_values["language"] = None

        # verified_label
        verified_col = mapping_result.mapping.get("verified_label")
        if verified_col and verified_col in row_data:
            mapped_values["verified_label"] = row_data[verified_col]
        else:
            mapped_values["verified_label"] = None

        # --- Validation checks ---

        # Check for missing review ID
        rid = mapped_values.get("review_id", "")
        if rid is None or (isinstance(rid, str) and rid.strip() == ""):
            row_status = "rejected"
            row_reasons.append("Missing review ID")

        # Check for missing review text
        rt = mapped_values.get("review_text", "")
        if rt is None or (isinstance(rt, str) and rt.strip() == ""):
            if row_status != "rejected":
                row_status = "rejected"
            row_reasons.append("Missing review text")

        # Check for duplicate IDs (flag if not already rejected)
        rid_value = mapped_values.get("review_id", "")
        if rid_value in duplicate_id_set and row_status != "rejected":
            row_status = "flagged"
            if "Duplicate review ID" not in row_reasons:
                row_reasons.append("Duplicate review ID - may affect analysis")

        # If passed all checks, accepted
        if row_status == "accepted":
            # Additional flagging for common quality issues
            # Check for very short review text
            if isinstance(rt, str) and len(rt.strip()) < 3:
                row_status = "flagged"
                row_reasons.append("Very short review text")

            # Check for extreme ratings
            rating = mapped_values.get("rating")
            if rating is not None:
                try:
                    rating_float = float(rating)
                    if rating_float < 1 or rating_float > 5:
                        if row_status != "rejected":
                            row_status = "flagged"
                            row_reasons.append("Rating outside expected 1-5 range")
                except (ValueError, TypeError):
                    pass

        # Set final status
        if row_status == "accepted":
            accepted.append(
                IngestionRecord(
                    row_index=idx,
                    row_data=row_data,
                    status="accepted",
                    reasons=row_reasons,
                    mapped_values=mapped_values,
                )
            )
        elif row_status == "rejected":
            rejected.append(
                IngestionRecord(
                    row_index=idx,
                    row_data=row_data,
                    status="rejected",
                    reasons=row_reasons,
                    mapped_values=mapped_values,
                )
            )
        else:
            # flagged
            flagged.append(
                IngestionRecord(
                    row_index=idx,
                    row_data=row_data,
                    status="flagged",
                    reasons=row_reasons,
                    mapped_values=mapped_values,
                )
            )

    logger.info(
        f"Ingestion complete: {len(accepted)} accepted, "
        f"{len(rejected)} rejected, {len(flagged)} flagged"
    )

    # --- Step 6: Quality report ---
    quality_report = validate_data_quality(df, mapping_result, rating_scale)
    quality_report["total_source_rows"] = total_source_rows
    quality_report["rows_parsed"] = rows_parsed
    quality_report["rows_omitted_limit"] = rows_omitted_limit
    quality_report["row_accounting"] = {
        "total_source_rows": total_source_rows,
        "rows_parsed": rows_parsed,
        "rows_omitted_limit": rows_omitted_limit,
        "rows_rejected": len(rejected),
        "rows_accepted": len(accepted),
        "rows_flagged": len(flagged),
    }

    # Add additional metadata
    quality_report["warnings"] = warnings
    quality_report["errors"] = errors
    quality_report["mapping_info"] = {
        "mapped_columns": mapping_result.mapping.source_to_canonical,
        "unmapped_columns": mapping_result.unmapped,
        "ambiguous_columns": mapping_result.ambiguous,
    }

    return IngestionResult(
        accepted=accepted,
        rejected=rejected,
        flagged=flagged,
        df_validated=df,  # In a full implementation, this would be filtered
        mapping_result=mapping_result,
        quality_report=quality_report,
        warnings=warnings,
        errors=errors,
        raw_filename=file_path.name,
    )
