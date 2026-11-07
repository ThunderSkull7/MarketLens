"""
MarketLens - Data Cleaning Pipeline.

Modular cleaning pipeline that accepts Batch 1 ingestion results and produces
reproducible, analysis-ready datasets. Preserves data integrity, traceability,
and classifications (accepted / rejected / flagged).
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import pandas as pd

from marketlens.analytics import (
    aggregate_by,
    dataset_overview,
    filter_by_date_range,
    filter_by_language,
    filter_by_product_service,
    filter_by_rating_range,
    filter_by_source,
    filter_by_status,
    filter_dataset,
    product_service_analysis,
    rating_analysis,
    temporal_analysis,
    text_analysis,
)
from marketlens.ingestion import IngestionRecord, IngestionResult
from marketlens.ingestion.validators import RATING_SCALES

logger = logging.getLogger("marketlens.cleaning")

__all__ = [
    "CleaningPipeline",
    "aggregate_by",
    "dataset_overview",
    "detect_duplicates",
    "detect_duplicates_by_id",
    "filter_by_date_range",
    "filter_by_language",
    "filter_by_product_service",
    "filter_by_rating_range",
    "filter_by_source",
    "filter_by_status",
    "filter_dataset",
    "get_accepted_data",
    "load_cleaned_data",
    "load_cleaning_report",
    "normalize_rating",
    "normalize_text",
    "product_service_analysis",
    "rating_analysis",
    "run_pipeline",
    "save_cleaned_data",
    "save_cleaning_report",
    "standardize_date",
    "temporal_analysis",
    "text_analysis",
]


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------


def normalize_text(
    text: Any,
    *,
    collapse_whitespace: bool = True,
    max_line_length: Optional[int] = None,
    remove_extra_newlines: bool = True,
) -> str:
    """
    Normalize review text without destroying punctuation, emojis, or meaning.

    Operations:
    1. Coerce to string (None / NaN -> "").
    2. Normalize line endings (\\r\\n and \\r -> \\n).
    3. Strip leading and trailing whitespace.
    4. Collapse runs of horizontal whitespace (spaces, tabs) per line (optional).
    5. Reduce runs of consecutive line breaks to a single \\n (optional).
    6. Truncate lines exceeding max_line_length with "..." marker (optional).
    7. Preserve punctuation, symbols, URLs, and linguistic structure.

    Parameters
    ----------
    text : Any
        Raw text value.
    collapse_whitespace : bool, default True
        If True, collapse runs of spaces and tabs to a single space.
    max_line_length : Optional[int], default None
        If set and > 0, truncate lines longer than this length.
    remove_extra_newlines : bool, default True
        If True, reduce sequences of multiple newlines to a single newline.

    Returns
    -------
    str
        Cleaned, normalized text string.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""

    s = str(text)
    # Normalize carriage returns
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.strip()

    if not s:
        return ""

    # Collapse horizontal whitespace per line
    if collapse_whitespace:
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in s.split("\n")]
        s = "\n".join(lines)

    # Normalize line breaks
    if remove_extra_newlines:
        s = re.sub(r"\n+", "\n", s)

    # Optional line truncation
    if max_line_length is not None and max_line_length > 0:
        lines = s.split("\n")
        truncated: list[str] = []
        for line in lines:
            if len(line) > max_line_length:
                marker = "..."
                cut = max(0, max_line_length - len(marker))
                truncated.append(line[:cut] + marker)
            else:
                truncated.append(line)
        s = "\n".join(truncated)

    return s


# ---------------------------------------------------------------------------
# Date standardisation
# ---------------------------------------------------------------------------


def standardize_date(
    value: Any,
    *,
    default_na: bool = True,
) -> Optional[date]:
    """
    Parse a date value into a standard datetime.date object.

    Does not invent default dates or sentinel values (e.g. 1970).
    Returns None for invalid or missing values.

    Parameters
    ----------
    value : Any
        Date representation (string, date, datetime, Timestamp, or NaN).
    default_na : bool, default True
        Preserved for signature compatibility; missing/invalid dates return None.

    Returns
    -------
    Optional[date]
        Parsed datetime.date or None.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.date()

    val_str = str(value).strip()
    if not val_str or val_str.lower() in ("nan", "nat", "none", "null"):
        return None

    # Common explicit formats first
    date_formats = (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%m/%d/%Y",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%d %b %Y",
        "%b %d, %Y",
    )
    for fmt in date_formats:
        try:
            return datetime.strptime(val_str, fmt).date()
        except ValueError:
            continue

    # Fallback to pandas flexible datetime parser
    try:
        ts = pd.to_datetime(val_str, errors="coerce")
        if pd.notna(ts):
            return ts.date()
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Rating normalisation
# ---------------------------------------------------------------------------


def normalize_rating(
    value: Any,
    *,
    scale: str = "1-5",
    clamp: bool = False,
) -> Optional[float]:
    """
    Normalize a rating value to a float according to the configured rating scale.

    Data-integrity policy: By default (clamp=False), values outside the scale bounds
    are NOT clamped into valid ratings, avoiding silent data corruption where invalid
    ratings (e.g. 0.0 or 10.0 on a 1-5 scale) become valid ratings or sentiment labels.
    Out-of-bounds ratings return None (or preserve invalid status).
    When clamp=True, values outside bounds are clamped to min_val or max_val.

    Parameters
    ----------
    value : Any
        Raw rating value.
    scale : str, default "1-5"
        Rating scale name ("1-5", "1-10", "1-3", "5-star", "thumbs").
    clamp : bool, default False
        If True, numeric values outside the scale bounds are clamped to min/max.
        If False (default), out-of-scale values return None.

    Returns
    -------
    Optional[float]
        Normalized float rating within scale bounds, or None.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    scale_cfg = RATING_SCALES.get(
        scale.lower().replace(" ", ""),
        {"min": 1.0, "max": 5.0},
    )
    min_val = float(scale_cfg["min"])
    max_val = float(scale_cfg["max"])

    try:
        num = float(value)
        if pd.isna(num):
            return None
    except (TypeError, ValueError):
        return None

    if min_val <= num <= max_val:
        return num

    if clamp:
        return max(min_val, min(max_val, num))

    return None


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------


def detect_duplicates_by_id(
    df: pd.DataFrame,
    id_column: str = "review_id",
    text_column: str = "review_text",
    id_match_only: bool = True,
) -> Tuple[pd.DataFrame, int, list[tuple]]:
    """
    Detect duplicate feedback records while preserving original rows.

    Distinguishes detection from deletion: records are flagged in an `is_duplicate`
    boolean column rather than being discarded.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    id_column : str, default "review_id"
        Identifier column name.
    text_column : str, default "review_text"
        Review text column name.
    id_match_only : bool, default True
        If True, match by ID alone. If False, match by both ID and text.

    Returns
    -------
    Tuple[pd.DataFrame, int, list[tuple]]
        - DataFrame copy with `is_duplicate` column.
        - Total count of duplicate rows (subsequent occurrences).
        - Duplicate groups list of (key, [indices]).
    """
    df_out = df.copy()

    if id_column not in df_out.columns:
        df_out["is_duplicate"] = False
        return df_out, 0, []

    if id_match_only or text_column not in df_out.columns:
        # Match by ID only
        key_series = df_out[id_column].astype(str)
    else:
        # Match by ID and text
        key_series = (
            df_out[id_column].astype(str) + "|||" + df_out[text_column].astype(str)
        )

    dup_mask = key_series.duplicated(keep="first")
    df_out["is_duplicate"] = dup_mask

    # Group all duplicates (including first occurrence) to build groups
    all_dup_mask = key_series.duplicated(keep=False)
    groups_dict: dict[str, list[int]] = {}
    for idx in df_out.index[all_dup_mask]:
        k = str(key_series.loc[idx])
        groups_dict.setdefault(k, []).append(idx)

    dup_groups = [(k, sorted(idxs)) for k, idxs in groups_dict.items()]
    dup_count = int(dup_mask.sum())

    return df_out, dup_count, dup_groups


# Alias for backward compatibility
detect_duplicates = detect_duplicates_by_id


# ---------------------------------------------------------------------------
# Cleaning pipeline orchestration
# ---------------------------------------------------------------------------


class CleaningPipeline:
    """
    Orchestrates the data cleaning pipeline on an IngestionResult.

    Preserves data traceability, unmapped source columns, and access to
    rejected and flagged records. Never modifies raw source data.

    Example
    -------
    >>> from marketlens.ingestion import ingest
    >>> from marketlens.cleaning import CleaningPipeline
    >>> result = ingest("data/raw/synthetic_feedback_demo.csv")
    >>> pipeline = CleaningPipeline(result)
    >>> cleaned_df, report = pipeline.run(include_flagged=False)
    """

    def __init__(self, result: IngestionResult) -> None:
        self.result = result

    @property
    def accepted_records(self) -> list[IngestionRecord]:
        """Access records classified as accepted."""
        return self.result.accepted

    @property
    def rejected_records(self) -> list[IngestionRecord]:
        """Access records classified as rejected for inspection."""
        return self.result.rejected

    @property
    def flagged_records(self) -> list[IngestionRecord]:
        """Access records classified as flagged."""
        return self.result.flagged

    def get_accepted_records(self) -> list[IngestionRecord]:
        """Return list of accepted IngestionRecords."""
        return self.result.accepted

    def get_rejected_records(self) -> list[IngestionRecord]:
        """Return list of rejected IngestionRecords for inspection."""
        return self.result.rejected

    def get_flagged_records(self) -> list[IngestionRecord]:
        """Return list of flagged IngestionRecords."""
        return self.result.flagged

    def _build_work_df(self) -> pd.DataFrame:
        """
        Build working DataFrame preserving mapped values, unmapped source columns,
        and traceability metadata (_row_index, _status, _reasons).
        """
        records = self.result.accepted + self.result.flagged + self.result.rejected
        data: list[dict[str, Any]] = []

        for rec in records:
            row: dict[str, Any] = {}
            # Preserve unmapped source columns from row_data
            for col, val in rec.row_data.items():
                if col not in ("_row_index", "_status", "_reasons", "is_flagged"):
                    row[col] = val

            # Canonical mapped values override raw unmapped
            for k, v in rec.mapped_values.items():
                row[k] = v

            row["_row_index"] = rec.row_index
            row["_status"] = rec.status
            row["_reasons"] = list(rec.reasons)
            row["is_flagged"] = rec.is_flagged()
            data.append(row)

        if not data:
            return pd.DataFrame()

        return pd.DataFrame(data)

    def run(
        self,
        *,
        include_flagged: bool = False,
        rating_scale: str = "1-5",
        normalize_text_opts: Optional[dict[str, Any]] = None,
        text_column: str = "review_text",
        date_column: str = "created_at",
        rating_column: str = "rating",
        id_column: str = "review_id",
        should_detect_duplicates: bool = True,
        id_match_only: bool = True,
        detect_duplicates: Optional[bool] = None,
        **kwargs: Any,
    ) -> Tuple[pd.DataFrame, dict[str, Any]]:
        """
        Run the full cleaning pipeline and return the cleaned DataFrame and report.

        Parameters
        ----------
        include_flagged : bool, default False
            If True, flagged records are retained in the analysis-ready output.
            If False, only accepted records are retained.
        rating_scale : str, default "1-5"
            Rating scale for normalisation.
        normalize_text_opts : Optional[dict], default None
            Keyword arguments passed to :func:`normalize_text`.
        text_column : str, default "review_text"
            Canonical text column.
        date_column : str, default "created_at"
            Canonical timestamp column.
        rating_column : str, default "rating"
            Canonical rating column.
        id_column : str, default "review_id"
            Canonical ID column.
        should_detect_duplicates : bool, default True
            Whether to detect duplicate records.
        id_match_only : bool, default True
            Whether duplicate matching matches review_id alone.
        detect_duplicates : Optional[bool], default None
            Parameter alias for should_detect_duplicates.
        **kwargs : Any
            Additional options.

        Returns
        -------
        Tuple[pd.DataFrame, dict[str, Any]]
            Cleaned analysis-ready DataFrame and structured cleaning report.
        """
        if detect_duplicates is not None:
            should_detect_duplicates = detect_duplicates

        working_df = self._build_work_df()
        if working_df.empty:
            return working_df, self._empty_report("No records to clean")

        operations: list[str] = []

        # 1. Text normalization
        text_opts = normalize_text_opts or {}
        if text_column in working_df.columns:
            working_df[text_column] = working_df[text_column].apply(
                lambda v: normalize_text(v, **text_opts)
            )
            operations.append("text_normalisation")

        # 2. Date standardisation
        if date_column in working_df.columns:
            working_df[date_column] = working_df[date_column].apply(
                lambda v: standardize_date(v)
            )
            operations.append("date_standardisation")

        # 3. Rating normalisation
        if rating_column in working_df.columns:
            working_df[rating_column] = working_df[rating_column].apply(
                lambda v: normalize_rating(v, scale=rating_scale)
            )
            operations.append("rating_normalisation")

        # 4. Optional fields standardisation
        if "product_or_service" in working_df.columns:
            working_df["product_or_service"] = working_df["product_or_service"].apply(
                lambda v: (
                    str(v).strip() if pd.notna(v) and str(v).strip() != "" else None
                )
            )
        if "source" in working_df.columns:
            working_df["source"] = working_df["source"].apply(
                lambda v: (
                    str(v).strip() if pd.notna(v) and str(v).strip() != "" else None
                )
            )
        if "language" in working_df.columns:
            working_df["language"] = working_df["language"].apply(
                lambda v: (
                    str(v).strip().lower()
                    if pd.notna(v) and str(v).strip() != ""
                    else None
                )
            )
        if "verified_label" in working_df.columns:
            working_df["verified_label"] = working_df["verified_label"].apply(
                lambda v: (
                    str(v).strip().lower()
                    if pd.notna(v) and str(v).strip() != ""
                    else None
                )
            )
        operations.append("optional_field_standardisation")

        # 5. Duplicate detection
        dup_count = 0
        dup_groups: list[tuple] = []
        if should_detect_duplicates and id_column in working_df.columns:
            working_df, dup_count, dup_groups = detect_duplicates_by_id(
                working_df,
                id_column=id_column,
                text_column=text_column,
                id_match_only=id_match_only,
            )
            operations.append("duplicate_detection")
        else:
            working_df["is_duplicate"] = False

        # Compute missing value counts before analytical filtering
        missing_before = self._calculate_missing_counts(working_df)

        # 6. Analytical inclusion filtering
        input_total = (
            len(self.result.accepted)
            + len(self.result.flagged)
            + len(self.result.rejected)
        )
        if include_flagged:
            retained_df = working_df[
                working_df["_status"].isin(["accepted", "flagged"])
            ].copy()
            exclusion_reasons = [
                "Rejected records excluded (fail mandatory validation)"
            ]
        else:
            retained_df = working_df[working_df["_status"] == "accepted"].copy()
            exclusion_reasons = [
                "Rejected records excluded (fail mandatory validation)",
                "Flagged records excluded by inclusion policy (include_flagged=False)",
            ]

        missing_after = self._calculate_missing_counts(retained_df)
        rows_retained = len(retained_df)
        rows_excluded = input_total - rows_retained

        # 7. Assemble cleaning report
        report = {
            "report": "cleaning_report",  # explicit report identifier
            "input_row_count": input_total,
            "accepted_count": len(self.result.accepted),
            "rejected_count": len(self.result.rejected),
            "flagged_count": len(self.result.flagged),
            "rows_retained": rows_retained,
            "rows_excluded": rows_excluded,
            "exclusion_reasons": exclusion_reasons,
            "missing_value_counts_before": missing_before,
            "missing_value_counts_after": missing_after,
            "duplicate_count": dup_count,
            "invalid_date_count": self.result.quality_report.get("invalid_dates", 0),
            "invalid_rating_count": self.result.quality_report.get(
                "invalid_ratings", 0
            ),
            "cleaning_operations": operations,
            "warnings": list(self.result.warnings),
            "errors": list(self.result.errors),
            "rating_scale": rating_scale,
            "source_file": self.result.raw_filename,
            "row_accounting": self.result.row_accounting,
        }

        return retained_df, report

    @staticmethod
    def _calculate_missing_counts(df: pd.DataFrame) -> dict[str, int]:
        """Count missing values per column, excluding internal metadata columns."""
        res: dict[str, int] = {}
        for col in df.columns:
            if col.startswith("_"):
                continue
            cnt = int(df[col].isna().sum())
            if cnt > 0:
                res[col] = cnt
        return res

    @staticmethod
    def _empty_report(reason: str) -> dict[str, Any]:
        """Construct report structure for empty or unparseable input."""
        return {
            "report": "cleaning_report",
            "input_row_count": 0,
            "accepted_count": 0,
            "rejected_count": 0,
            "flagged_count": 0,
            "rows_retained": 0,
            "rows_excluded": 0,
            "exclusion_reasons": [],
            "missing_value_counts_before": {},
            "missing_value_counts_after": {},
            "duplicate_count": 0,
            "invalid_date_count": 0,
            "invalid_rating_count": 0,
            "cleaning_operations": [],
            "warnings": [],
            "errors": [reason],
            "rating_scale": "1-5",
            "source_file": "unknown",
            "row_accounting": {},
        }


# ---------------------------------------------------------------------------
# Analysis-ready dataset interface
# ---------------------------------------------------------------------------


def get_accepted_data(
    result: IngestionResult,
    *,
    include_flagged: bool = False,
    rating_scale: str = "1-5",
    **pipeline_kwargs: Any,
) -> Tuple[pd.DataFrame, dict[str, Any]]:
    """
    Convenience interface providing an analysis-ready dataset from an IngestionResult.

    Parameters
    ----------
    result : IngestionResult
        The result returned by ingest().
    include_flagged : bool, default False
        Whether to retain flagged records.
    rating_scale : str, default "1-5"
        The scale for rating validation and normalization.
    **pipeline_kwargs : Any
        Additional options passed to CleaningPipeline.run().

    Returns
    -------
    Tuple[pd.DataFrame, dict[str, Any]]
        Analysis-ready cleaned DataFrame and structured cleaning report.
    """
    pipeline = CleaningPipeline(result)
    return pipeline.run(
        include_flagged=include_flagged,
        rating_scale=rating_scale,
        **pipeline_kwargs,
    )


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def save_cleaned_data(
    df: pd.DataFrame,
    source_file: str,
    output_dir: str = "data/processed",
    *,
    include_flagged: bool = False,
    filename: Optional[str] = None,
) -> str:
    """
    Save cleaned DataFrame to a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned DataFrame.
    source_file : str
        Source filename or path for naming.
    output_dir : str, default "data/processed"
        Destination directory.
    include_flagged : bool, default False
        Whether flagged records were included in the export.
    filename : Optional[str], default None
        Explicit filename override.

    Returns
    -------
    str
        Path to the saved CSV file.
    """
    os.makedirs(output_dir, exist_ok=True)

    if filename is None:
        base = os.path.splitext(os.path.basename(source_file))[0]
        suffix = "_with_flagged" if include_flagged else ""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{base}_cleaned{suffix}_{ts}.csv"

    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False, encoding="utf-8")
    logger.info("Saved cleaned dataset to: %s", filepath)
    return filepath


def save_cleaning_report(
    report: dict[str, Any],
    output_dir: str = "reports",
    filename: Optional[str] = None,
) -> str:
    """
    Save cleaning report to a JSON file.

    Parameters
    ----------
    report : dict[str, Any]
        The cleaning report dictionary.
    output_dir : str, default "reports"
        Destination directory.
    filename : Optional[str], default None
        Explicit filename override.

    Returns
    -------
    str
        Path to the saved JSON report.
    """
    os.makedirs(output_dir, exist_ok=True)

    if filename is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cleaning_report_{ts}.json"

    filepath = os.path.join(output_dir, filename)
    report_copy = dict(report)
    report_copy["saved_at"] = datetime.now().isoformat()

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report_copy, f, indent=2, ensure_ascii=False)

    logger.info("Saved cleaning report to: %s", filepath)
    return filepath


def load_cleaned_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load a previously saved cleaned dataset CSV.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded cleaned DataFrame.
    """
    return pd.read_csv(filepath, encoding="utf-8")


def load_cleaning_report(filepath: Union[str, Path]) -> dict[str, Any]:
    """
    Load a previously saved cleaning report JSON.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to the JSON report file.

    Returns
    -------
    dict[str, Any]
        Report contents dictionary.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Reproducibility entry point
# ---------------------------------------------------------------------------


def run_pipeline(
    csv_path: Union[str, Path],
    *,
    include_flagged: bool = False,
    rating_scale: str = "1-5",
    text_normalise_opts: Optional[dict[str, Any]] = None,
    date_range: Optional[Tuple[Optional[Any], Optional[Any]]] = None,
    rating_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
    language: Optional[str] = None,
    product: Optional[str] = None,
    source: Optional[str] = None,
    output_dir: str = "data/processed",
    report_dir: str = "reports",
) -> dict[str, Any]:
    """
    Reproducible end-to-end execution: CSV -> Ingestion -> Cleaning -> EDA -> Analytics.

    Parameters
    ----------
    csv_path : Union[str, Path]
        Path to the raw input CSV.
    include_flagged : bool, default False
        Whether to include flagged records in analytical outputs.
    rating_scale : str, default "1-5"
        Configured rating scale.
    text_normalise_opts : Optional[dict], default None
        Options for text normalization.
    date_range : Optional[Tuple], default None
        (start_date, end_date) filter.
    rating_range : Optional[Tuple], default None
        (min_rating, max_rating) filter.
    language : Optional[str], default None
        Language code filter.
    product : Optional[str], default None
        Product filter.
    source : Optional[str], default None
        Source channel filter.
    output_dir : str, default "data/processed"
        Directory for cleaned CSV output.
    report_dir : str, default "reports"
        Directory for JSON reports.

    Returns
    -------
    dict[str, Any]
        Complete analytics summary and file paths.
    """
    from marketlens.ingestion import ingest

    # 1. Ingestion
    result = ingest(csv_path, rating_scale=rating_scale)

    # 2. Cleaning
    pipeline = CleaningPipeline(result)
    cleaned_df, cleaning_report = pipeline.run(
        include_flagged=include_flagged,
        rating_scale=rating_scale,
        normalize_text_opts=text_normalise_opts,
    )

    # 3. Composable filtering
    s_date = date_range[0] if date_range else None
    e_date = date_range[1] if date_range else None
    r_min = rating_range[0] if rating_range else None
    r_max = rating_range[1] if rating_range else None

    filtered_df = filter_dataset(
        cleaned_df,
        start_date=s_date,
        end_date=e_date,
        min_rating=r_min,
        max_rating=r_max,
        product=product,
        source=source,
        language=language,
        include_flagged=include_flagged,
        rating_scale=rating_scale,
    )

    # 4. Exploratory Data Analysis & Analytics
    overview = dataset_overview(filtered_df)
    rating_sum = rating_analysis(filtered_df, rating_scale=rating_scale)
    temporal_sum = temporal_analysis(filtered_df)
    text_sum = text_analysis(filtered_df)
    prod_sum = product_service_analysis(filtered_df)

    # 5. Persistence
    csv_out = save_cleaned_data(
        filtered_df,
        source_file=str(csv_path),
        output_dir=output_dir,
        include_flagged=include_flagged,
    )
    report_out = save_cleaning_report(cleaning_report, output_dir=report_dir)

    # 6. Analytics JSON export
    analytics_payload = {
        "source_file": Path(csv_path).name,
        "processed_at": datetime.now().isoformat(),
        "rating_scale": rating_scale,
        "filters_applied": {
            "include_flagged": include_flagged,
            "date_range": [str(d) if d else None for d in (date_range or (None, None))],
            "rating_range": list(rating_range) if rating_range else None,
            "language": language,
            "product": product,
            "source": source,
        },
        "overview": overview,
        "rating_analysis": rating_sum,
        "temporal_analysis": temporal_sum,
        "text_analysis": text_sum,
        "product_service_analysis": prod_sum,
    }
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    analytics_path = os.path.join(report_dir, f"analytics_summary_{ts}.json")
    with open(analytics_path, "w", encoding="utf-8") as f:
        json.dump(analytics_payload, f, indent=2, ensure_ascii=False)

    return {
        "csv_path": csv_out,
        "cleaning_report_path": report_out,
        "analytics_report_path": analytics_path,
        "ingest_summary": result.summary(),
        "overview": overview,
        "rating_analysis": rating_sum,
        "temporal_analysis": temporal_sum,
        "text_analysis": text_sum,
        "product_service_analysis": prod_sum,
        "row_counts": {
            "input": cleaning_report["input_row_count"],
            "retained": len(filtered_df),
            "excluded": cleaning_report["rows_excluded"],
        },
        "cleaning_operations": cleaning_report["cleaning_operations"],
        "rating_scale": rating_scale,
    }
