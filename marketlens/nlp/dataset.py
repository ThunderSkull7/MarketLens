"""
MarketLens - Sentiment Dataset & Label Preparation Module.

Handles rating-to-sentiment label mapping, dataset preparation, train/test splitting,
and data leakage prevention for sentiment classification.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from marketlens.cleaning import normalize_rating, normalize_text

logger = logging.getLogger("marketlens.nlp.dataset")


# ---------------------------------------------------------------------------
# Rating to sentiment label derivation
# ---------------------------------------------------------------------------


def rating_to_sentiment(
    rating: Any,
    scale: str = "1-5",
    mapping: Optional[dict[str, tuple[float, float]]] = None,
) -> Optional[str]:
    """
    Derive a categorical sentiment label from a numeric rating.

    Important Data-Integrity Policy:
    Ratings outside the configured scale range or invalid/missing ratings
    are NEVER silently converted to valid sentiment labels. They return None.
    Rating-derived sentiment labels represent WEAK LABELS and must be distinguished
    from independently annotated ground-truth sentiment.

    Parameters
    ----------
    rating : Any
        The numeric rating value (e.g. 1.0, 5.0).
    scale : str, default "1-5"
        The active rating scale identifier ("1-5", "1-10", "1-3", "5-star", "thumbs").
    mapping : Optional[dict[str, tuple[float, float]]]
        Optional custom boundary mapping {label: (inclusive_min, inclusive_max)}.

    Returns
    -------
    Optional[str]
        One of "negative", "neutral", "positive", or None if invalid/unparseable.
    """
    if rating is None or (isinstance(rating, float) and pd.isna(rating)):
        return None

    # Validate rating against scale without boundary clamping (data integrity)
    norm_rating = normalize_rating(rating, scale=scale, clamp=False)
    if norm_rating is None:
        return None

    scale_clean = scale.lower().replace(" ", "")

    # Custom mapping if provided
    if mapping is not None:
        for label, (low, high) in mapping.items():
            if low <= norm_rating <= high:
                return label.lower()
        return None

    # Documented default mappings per rating scale
    if scale_clean in ("1-5", "5-star"):
        if 1.0 <= norm_rating <= 2.0:
            return "negative"
        if 2.0 < norm_rating < 4.0:  # e.g., 2.5, 3.0, 3.5
            return "neutral"
        if 4.0 <= norm_rating <= 5.0:
            return "positive"

    elif scale_clean == "1-10":
        if 1.0 <= norm_rating <= 4.0:
            return "negative"
        if 4.0 < norm_rating < 7.0:  # e.g., 5.0, 6.0
            return "neutral"
        if 7.0 <= norm_rating <= 10.0:
            return "positive"

    elif scale_clean == "1-3":
        if norm_rating == 1.0:
            return "negative"
        if norm_rating == 2.0:
            return "neutral"
        if norm_rating == 3.0:
            return "positive"

    elif scale_clean == "thumbs":
        if norm_rating == 0.0:
            return "negative"
        if norm_rating == 1.0:
            return "positive"

    return None


# ---------------------------------------------------------------------------
# Dataset preparation & leakage prevention
# ---------------------------------------------------------------------------


def prepare_sentiment_dataset(
    df: pd.DataFrame,
    *,
    text_column: str = "review_text",
    rating_column: str = "rating",
    rating_scale: str = "1-5",
    include_flagged: bool = False,
    drop_empty_text: bool = True,
    deduplicate_text: bool = True,
) -> pd.DataFrame:
    """
    Prepare an analysis-ready DataFrame for supervised sentiment modelling.

    Rules:
    - Reusable text normalization is applied.
    - Excludes rejected records (failed mandatory validation).
    - Configurable inclusion of flagged records (include_flagged).
    - Unparseable and invalid ratings are dropped rather than converted to sentiment.
    - Deduplicates identical review texts to prevent cross-partition leakage.
    - Preserves review_id and _row_index for full traceability.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (from IngestionResult or CleaningPipeline).
    text_column : str, default "review_text"
        Column containing review text.
    rating_column : str, default "rating"
        Column containing ratings.
    rating_scale : str, default "1-5"
        Rating scale for label derivation.
    include_flagged : bool, default False
        Whether to include records with quality flags.
    drop_empty_text : bool, default True
        Whether to drop blank or whitespace-only texts.
    deduplicate_text : bool, default True
        Whether to keep only the first occurrence of identical texts.

    Returns
    -------
    pd.DataFrame
        Cleaned modelling DataFrame with "sentiment" column and traceability keys.
    """
    if df.empty:
        return pd.DataFrame(columns=[text_column, "sentiment", "_row_index"])

    work_df = df.copy()

    # 1. Filter rejected records
    if "_status" in work_df.columns:
        if include_flagged:
            work_df = work_df[work_df["_status"].isin(["accepted", "flagged"])].copy()
        else:
            work_df = work_df[work_df["_status"] == "accepted"].copy()

    if work_df.empty:
        return pd.DataFrame(columns=[text_column, "sentiment", "_row_index"])

    # 2. Text normalization
    if text_column in work_df.columns:
        work_df[text_column] = work_df[text_column].apply(
            lambda t: normalize_text(t, collapse_whitespace=True)
        )
    else:
        raise ValueError(
            f"Required text column '{text_column}' not found in DataFrame."
        )

    # 3. Drop empty text
    if drop_empty_text:
        work_df = work_df[work_df[text_column].str.strip() != ""].copy()
        work_df = work_df.dropna(subset=[text_column]).copy()

    # 4. Derive sentiment label (if rating_column present)
    if "sentiment" not in work_df.columns and rating_column in work_df.columns:
        work_df["sentiment"] = work_df[rating_column].apply(
            lambda r: rating_to_sentiment(r, scale=rating_scale)
        )

    # 5. Drop rows with invalid / unmapped sentiment
    if "sentiment" in work_df.columns:
        work_df = work_df.dropna(subset=["sentiment"]).copy()

    # 6. Prevent train/test leakage by deduplicating exact text
    if deduplicate_text and text_column in work_df.columns:
        work_df = work_df.drop_duplicates(subset=[text_column], keep="first").copy()

    return work_df.reset_index(drop=True)


def split_sentiment_dataset(
    df: pd.DataFrame,
    *,
    text_column: str = "review_text",
    label_column: str = "sentiment",
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split sentiment dataset into train and test partitions.

    Ensures stratified sampling when class distribution permits and verifies
    zero text leakage between train and test partitions.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared modelling DataFrame.
    text_column : str, default "review_text"
        Text feature column.
    label_column : str, default "sentiment"
        Target label column.
    test_size : float, default 0.2
        Fraction of data reserved for test set.
    random_state : int, default 42
        Deterministic random seed.
    stratify : bool, default True
        Whether to use stratified sampling.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df) partitioned DataFrames.
    """
    if len(df) == 0:
        return df.copy(), df.copy()

    if len(df) < 5:
        # Too small for split
        return df.copy(), pd.DataFrame(columns=df.columns)

    y = df[label_column]
    class_counts = y.value_counts()

    # Stratified split requires at least 2 samples per class
    can_stratify = stratify and (class_counts >= 2).all() and len(class_counts) > 1

    stratify_target = y if can_stratify else None
    if stratify and not can_stratify:
        logger.warning(
            "Stratified split disabled: some classes have fewer than 2 instances: %s",
            dict(class_counts),
        )

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )

    # Leakage check: ensure no identical review text spans both sets
    train_texts = set(train_df[text_column].str.strip())
    test_texts = set(test_df[text_column].str.strip())
    overlap = train_texts.intersection(test_texts)
    if overlap:
        logger.warning(
            "Data leakage alert: %d identical texts overlap across sets.",
            len(overlap),
        )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
