"""
MarketLens - Exploratory Data Analysis & Analytics Module.

Provides reusable analytical summaries, exploratory data analysis functions,
filtering, and aggregations for customer feedback datasets.
Designed to power downstream ML models and dashboard KPI cards.
"""

from __future__ import annotations

import re
from typing import Any, Optional, Union

import pandas as pd

from marketlens.ingestion.validators import RATING_SCALES

DEFAULT_STOPWORDS = frozenset(
    {
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "aren't",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can't",
        "cannot",
        "could",
        "couldn't",
        "did",
        "didn't",
        "do",
        "does",
        "doesn't",
        "doing",
        "don't",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "hadn't",
        "has",
        "hasn't",
        "have",
        "haven't",
        "having",
        "he",
        "he'd",
        "he'll",
        "he's",
        "her",
        "here",
        "here's",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "how's",
        "i",
        "i'd",
        "i'll",
        "i'm",
        "i've",
        "if",
        "in",
        "into",
        "is",
        "isn't",
        "it",
        "it's",
        "its",
        "itself",
        "let's",
        "me",
        "more",
        "most",
        "mustn't",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "ought",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "shan't",
        "she",
        "she'd",
        "she'll",
        "she's",
        "should",
        "shouldn't",
        "so",
        "some",
        "such",
        "than",
        "that",
        "that's",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "there's",
        "these",
        "they",
        "they'd",
        "they'll",
        "they're",
        "they've",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "wasn't",
        "we",
        "we'd",
        "we'll",
        "we're",
        "we've",
        "were",
        "weren't",
        "what",
        "what's",
        "when",
        "when's",
        "where",
        "where's",
        "which",
        "while",
        "who",
        "who's",
        "whom",
        "why",
        "why's",
        "with",
        "won't",
        "would",
        "wouldn't",
        "you",
        "you'd",
        "you'll",
        "you're",
        "you've",
        "your",
        "yours",
        "yourself",
        "yourselves",
    }
)


# ---------------------------------------------------------------------------
# A. Dataset overview
# ---------------------------------------------------------------------------


def dataset_overview(df: pd.DataFrame) -> dict[str, Any]:
    """
    Generate a comprehensive dataset overview for KPI cards and dashboard summaries.

    Parameters
    ----------
    df : pd.DataFrame
        The feedback DataFrame.

    Returns
    -------
    dict[str, Any]
        Dataset overview statistics including record counts, category breakdowns,
        date coverage, missing value distributions, and quality indicators.
    """
    total = len(df)

    # Product/service breakdown
    product_counts: dict[str, int] = {}
    if "product_or_service" in df.columns and total > 0:
        vc = df["product_or_service"].dropna()
        if len(vc) > 0:
            product_counts = {
                str(k): int(v) for k, v in vc.value_counts().to_dict().items()
            }

    # Source breakdown
    source_counts: dict[str, int] = {}
    if "source" in df.columns and total > 0:
        vc = df["source"].dropna()
        if len(vc) > 0:
            source_counts = {
                str(k): int(v) for k, v in vc.value_counts().to_dict().items()
            }

    # Language breakdown
    language_counts: dict[str, int] = {}
    if "language" in df.columns and total > 0:
        vc = df["language"].dropna()
        if len(vc) > 0:
            language_counts = {
                str(k): int(v) for k, v in vc.value_counts().to_dict().items()
            }

    # Date coverage
    min_date_str: Optional[str] = None
    max_date_str: Optional[str] = None
    date_range_days: Optional[int] = None

    if "created_at" in df.columns and total > 0:
        date_series = pd.to_datetime(df["created_at"], errors="coerce").dropna()
        if len(date_series) > 0:
            min_date = date_series.min()
            max_date = date_series.max()
            min_date_str = (
                str(min_date.date()) if hasattr(min_date, "date") else str(min_date)
            )
            max_date_str = (
                str(max_date.date()) if hasattr(max_date, "date") else str(max_date)
            )
            date_range_days = int((max_date - min_date).days)

    # Missing value counts and percentages
    missing_dist: dict[str, int] = {}
    missing_pcts: dict[str, float] = {}
    for col in df.columns:
        if col.startswith("_"):
            continue
        cnt = int(df[col].isna().sum())
        if cnt > 0:
            missing_dist[col] = cnt
            missing_pcts[col] = round((cnt / total) * 100.0, 2) if total > 0 else 0.0

    # Data quality summary
    canonical_cols = [
        c
        for c in ["review_id", "review_text", "rating", "created_at"]
        if c in df.columns
    ]
    complete_count = (
        int(df[canonical_cols].notna().all(axis=1).sum())
        if canonical_cols and total > 0
        else 0
    )
    duplicate_count = (
        int(df["is_duplicate"].sum())
        if "is_duplicate" in df.columns and total > 0
        else 0
    )
    flagged_count = (
        int((df["_status"] == "flagged").sum())
        if "_status" in df.columns and total > 0
        else 0
    )

    quality_summary = {
        "total_records": total,
        "complete_records": complete_count,
        "complete_records_pct": round((complete_count / total) * 100.0, 2)
        if total > 0
        else 0.0,
        "duplicate_records": duplicate_count,
        "flagged_records": flagged_count,
    }

    return {
        "total_records": total,
        "product_breakdown": product_counts,
        "source_breakdown": source_counts,
        "language_breakdown": language_counts,
        "date_coverage": {
            "min_date": min_date_str,
            "max_date": max_date_str,
            "date_range_days": date_range_days,
        },
        "missing_value_distribution": missing_dist,
        "missing_value_percentages": missing_pcts,
        "quality_summary": quality_summary,
    }


# ---------------------------------------------------------------------------
# B. Rating analysis
# ---------------------------------------------------------------------------


def rating_analysis(
    df: pd.DataFrame,
    rating_scale: str = "1-5",
    rating_column: str = "rating",
    date_column: str = "created_at",
) -> dict[str, Any]:
    """
    Compute rating distribution, central tendencies, and temporal trend.

    Respects the selected rating scale and avoids assuming 1-5.

    Parameters
    ----------
    df : pd.DataFrame
        The feedback DataFrame.
    rating_scale : str, default "1-5"
        The scale identifier ("1-5", "1-10", "1-3", "5-star", "thumbs").
    rating_column : str, default "rating"
        The canonical rating column.
    date_column : str, default "created_at"
        The date column for rating trend analysis.

    Returns
    -------
    dict[str, Any]
        Rating summary statistics and distribution.
    """
    scale_cfg = RATING_SCALES.get(
        rating_scale.lower().replace(" ", ""),
        {"min": 1.0, "max": 5.0, "description": "Custom scale"},
    )
    min_scale = float(scale_cfg["min"])
    max_scale = float(scale_cfg["max"])

    if rating_column not in df.columns:
        return {
            "distribution": {},
            "percentages": {},
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "total_rated": 0,
            "scale": rating_scale,
            "trend_over_time": {},
        }

    valid_ratings = pd.to_numeric(df[rating_column], errors="coerce").dropna()
    total_rated = len(valid_ratings)

    if total_rated == 0:
        return {
            "distribution": {},
            "percentages": {},
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "total_rated": 0,
            "scale": rating_scale,
            "trend_over_time": {},
        }

    # Distribution by exact rounded values or buckets
    value_counts = valid_ratings.value_counts().sort_index()
    distribution: dict[str, int] = {str(k): int(v) for k, v in value_counts.items()}
    percentages: dict[str, float] = {
        str(k): round((v / total_rated) * 100.0, 2) for k, v in distribution.items()
    }

    # Rating trends over time
    trend_over_time: dict[str, float] = {}
    if date_column in df.columns:
        dates = pd.to_datetime(df[date_column], errors="coerce")
        df_valid = df.loc[valid_ratings.index].copy()
        df_valid["_dt"] = dates
        df_valid = df_valid.dropna(subset=["_dt"])
        if len(df_valid) > 0:
            df_valid["_period"] = df_valid["_dt"].dt.to_period("M").astype(str)
            period_mean = df_valid.groupby("_period")[rating_column].mean().sort_index()
            trend_over_time = {
                str(k): round(float(v), 2) for k, v in period_mean.items()
            }

    return {
        "distribution": distribution,
        "percentages": percentages,
        "min": float(valid_ratings.min()),
        "max": float(valid_ratings.max()),
        "mean": round(float(valid_ratings.mean()), 2),
        "median": round(float(valid_ratings.median()), 2),
        "total_rated": total_rated,
        "scale": rating_scale,
        "scale_min": min_scale,
        "scale_max": max_scale,
        "trend_over_time": trend_over_time,
    }


# ---------------------------------------------------------------------------
# C. Temporal analysis
# ---------------------------------------------------------------------------


def temporal_analysis(
    df: pd.DataFrame,
    period: str = "M",
    date_column: str = "created_at",
    rating_column: str = "rating",
) -> dict[str, Any]:
    """
    Compute feedback volume and rating trends over time.

    Includes period-over-period changes and identifies periods with
    insufficient observations rather than producing misleading comparisons.

    Parameters
    ----------
    df : pd.DataFrame
        The feedback DataFrame.
    period : str, default "M"
        Frequency string ("D" for daily, "W" for weekly, "M" for monthly).
    date_column : str, default "created_at"
        The timestamp column.
    rating_column : str, default "rating"
        The numeric rating column.

    Returns
    -------
    dict[str, Any]
        Temporal summaries and period-over-period comparisons.
    """
    if date_column not in df.columns or len(df) == 0:
        return {
            "volume_by_period": {},
            "rating_trend_by_period": {},
            "period_over_period_changes": {},
            "period_warnings": ["No 'created_at' column or empty dataset"],
        }

    dates = pd.to_datetime(df[date_column], errors="coerce")
    valid_mask = dates.notna()

    if not valid_mask.any():
        return {
            "volume_by_period": {},
            "rating_trend_by_period": {},
            "period_over_period_changes": {},
            "period_warnings": ["No parseable dates found in dataset"],
        }

    work_df = df[valid_mask].copy()
    work_df["_period"] = dates[valid_mask].dt.to_period(period).astype(str)

    # Volume by period (sorted chronologically)
    vol_series = work_df.groupby("_period").size().sort_index()
    volume_by_period = {str(k): int(v) for k, v in vol_series.items()}

    # Rating trend by period
    rating_trend_by_period: dict[str, float] = {}
    if rating_column in work_df.columns:
        work_df["_num_rating"] = pd.to_numeric(work_df[rating_column], errors="coerce")
        rating_sub = work_df.dropna(subset=["_num_rating"])
        if len(rating_sub) > 0:
            trend_series = (
                rating_sub.groupby("_period")["_num_rating"].mean().sort_index()
            )
            rating_trend_by_period = {
                str(k): round(float(v), 2) for k, v in trend_series.items()
            }

    # Period-over-period changes
    period_keys = sorted(volume_by_period.keys())
    pop_changes: dict[str, dict[str, Any]] = {}
    for i in range(1, len(period_keys)):
        prev_p = period_keys[i - 1]
        curr_p = period_keys[i]
        prev_vol = volume_by_period[prev_p]
        curr_vol = volume_by_period[curr_p]
        vol_change = curr_vol - prev_vol
        vol_change_pct = (
            round((vol_change / prev_vol) * 100.0, 2) if prev_vol > 0 else None
        )

        rating_change = None
        if prev_p in rating_trend_by_period and curr_p in rating_trend_by_period:
            rating_change = round(
                rating_trend_by_period[curr_p] - rating_trend_by_period[prev_p], 2
            )

        pop_changes[curr_p] = {
            "compared_to": prev_p,
            "volume_change": vol_change,
            "volume_change_pct": vol_change_pct,
            "rating_change": rating_change,
        }

    # Data sufficiency checks
    warnings: list[str] = []
    if len(period_keys) < 3:
        warnings.append(
            f"Insufficient temporal periods ({len(period_keys)} observed; "
            "minimum 3 recommended for reliable trend detection)."
        )

    low_volume_periods = [p for p, count in volume_by_period.items() if count < 5]
    if low_volume_periods:
        sample_str = ", ".join(low_volume_periods[:5])
        suffix = "..." if len(low_volume_periods) > 5 else ""
        warnings.append(
            f"Periods with low observation counts (< 5): {sample_str}{suffix}"
        )

    return {
        "volume_by_period": volume_by_period,
        "rating_trend_by_period": rating_trend_by_period,
        "period_over_period_changes": pop_changes,
        "period_warnings": warnings,
    }


# ---------------------------------------------------------------------------
# D. Text analysis
# ---------------------------------------------------------------------------


def text_analysis(
    df: pd.DataFrame,
    *,
    text_column: str = "review_text",
    min_frequency: int = 2,
    max_terms: int = 50,
    stopwords: Optional[set[str]] = None,
) -> dict[str, Any]:
    """
    Compute term frequencies (unigrams and bigrams) from customer review text.

    Uses a lightweight regex tokenization approach without heavyweight NLP models.
    Removes standard stopwords and avoids counting empty or invalid strings.

    Note: Frequency analysis identifies lexical term occurrences only; it does
    not perform sentiment classification or topic modelling.

    Parameters
    ----------
    df : pd.DataFrame
        The feedback DataFrame.
    text_column : str, default "review_text"
        The column containing text reviews.
    min_frequency : int, default 2
        Minimum occurrences required for a term/ngram to be included.
    max_terms : int, default 50
        Maximum number of top terms to return in ranked lists.
    stopwords : Optional[set[str]], default None
        Custom stopword set. If None, default English stopwords are used.

    Returns
    -------
    dict[str, Any]
        Top unigrams, bigrams, total word counts, and metadata.
    """
    stopword_set = DEFAULT_STOPWORDS if stopwords is None else set(stopwords)

    disclaimer_msg = (
        "Term frequency analysis identifies occurrences only and does not "
        "determine sentiment or topics."
    )

    if text_column not in df.columns or len(df) == 0:
        return {
            "terms": {},
            "unigrams": {},
            "bigrams": {},
            "total_words": 0,
            "unique_terms": 0,
            "disclaimer": disclaimer_msg,
        }

    texts = df[text_column].dropna()
    texts = texts[texts.apply(lambda t: isinstance(t, str) and t.strip() != "")]

    if len(texts) == 0:
        return {
            "terms": {},
            "unigrams": {},
            "bigrams": {},
            "total_words": 0,
            "unique_terms": 0,
            "disclaimer": disclaimer_msg,
        }

    word_regex = re.compile(r"\b[a-zA-Z]{2,}\b")

    unigram_counts: dict[str, int] = {}
    bigram_counts: dict[str, int] = {}
    total_tokens = 0

    for text in texts:
        tokens = [w.lower() for w in word_regex.findall(text)]
        total_tokens += len(tokens)

        # Unigrams (filtering stopwords)
        filtered_tokens = [w for w in tokens if w not in stopword_set]
        for w in filtered_tokens:
            unigram_counts[w] = unigram_counts.get(w, 0) + 1

        # Bigrams (adjacent non-stopword tokens)
        for i in range(len(tokens) - 1):
            w1, w2 = tokens[i], tokens[i + 1]
            if w1 not in stopword_set and w2 not in stopword_set:
                bg = f"{w1} {w2}"
                bigram_counts[bg] = bigram_counts.get(bg, 0) + 1

    # Filter by minimum frequency and sort deterministically
    filtered_unigrams = {k: v for k, v in unigram_counts.items() if v >= min_frequency}
    sorted_unigrams = dict(
        sorted(filtered_unigrams.items(), key=lambda item: (-item[1], item[0]))[
            :max_terms
        ]
    )

    filtered_bigrams = {k: v for k, v in bigram_counts.items() if v >= min_frequency}
    sorted_bigrams = dict(
        sorted(filtered_bigrams.items(), key=lambda item: (-item[1], item[0]))[
            :max_terms
        ]
    )

    return {
        "terms": sorted_unigrams,  # backward compatibility alias
        "unigrams": sorted_unigrams,
        "bigrams": sorted_bigrams,
        "total_words": total_tokens,
        "unique_terms": len(unigram_counts),
        "disclaimer": disclaimer_msg,
    }


# ---------------------------------------------------------------------------
# E. Product/service and channel analysis
# ---------------------------------------------------------------------------


def product_service_analysis(
    df: pd.DataFrame,
    product_column: str = "product_or_service",
    source_column: str = "source",
    rating_column: str = "rating",
    min_observations: int = 5,
) -> dict[str, Any]:
    """
    Analyse customer feedback volumes and ratings across products and channels.

    Identifies categories with insufficient observations (< min_observations)
    to prevent spurious conclusions.

    Parameters
    ----------
    df : pd.DataFrame
        The feedback DataFrame.
    product_column : str, default "product_or_service"
        The product/service category column.
    source_column : str, default "source"
        The acquisition channel or platform column.
    rating_column : str, default "rating"
        The numeric rating column.
    min_observations : int, default 5
        Threshold for reliable category observation counts.

    Returns
    -------
    dict[str, Any]
        Breakdowns by product and source, plus low-observation alerts.
    """
    results: dict[str, Any] = {
        "volume_by_product": {},
        "avg_rating_by_product": {},
        "rating_stats_by_product": {},
        "volume_by_source": {},
        "avg_rating_by_source": {},
        "rating_stats_by_source": {},
        "insufficient_observation_categories": [],
    }

    if len(df) == 0:
        return results

    has_rating = rating_column in df.columns
    if has_rating:
        df = df.copy()
        df["_num_rating"] = pd.to_numeric(df[rating_column], errors="coerce")

    # Product/service analysis
    if product_column in df.columns:
        prod_vc = df[product_column].dropna().value_counts().sort_index()
        results["volume_by_product"] = {str(k): int(v) for k, v in prod_vc.items()}

        if has_rating:
            rated_prod = df.dropna(subset=[product_column, "_num_rating"])
            if len(rated_prod) > 0:
                stats = (
                    rated_prod.groupby(product_column)["_num_rating"]
                    .agg(["mean", "median", "count", "min", "max"])
                    .sort_index()
                )
                results["avg_rating_by_product"] = {
                    str(k): round(float(v), 2) for k, v in stats["mean"].items()
                }
                results["rating_stats_by_product"] = {
                    str(k): {
                        "mean": round(float(row["mean"]), 2),
                        "median": round(float(row["median"]), 2),
                        "count": int(row["count"]),
                        "min": float(row["min"]),
                        "max": float(row["max"]),
                    }
                    for k, row in stats.iterrows()
                }

    # Source/channel analysis
    if source_column in df.columns:
        src_vc = df[source_column].dropna().value_counts().sort_index()
        results["volume_by_source"] = {str(k): int(v) for k, v in src_vc.items()}

        if has_rating:
            rated_src = df.dropna(subset=[source_column, "_num_rating"])
            if len(rated_src) > 0:
                stats = (
                    rated_src.groupby(source_column)["_num_rating"]
                    .agg(["mean", "median", "count", "min", "max"])
                    .sort_index()
                )
                results["avg_rating_by_source"] = {
                    str(k): round(float(v), 2) for k, v in stats["mean"].items()
                }
                results["rating_stats_by_source"] = {
                    str(k): {
                        "mean": round(float(row["mean"]), 2),
                        "median": round(float(row["median"]), 2),
                        "count": int(row["count"]),
                        "min": float(row["min"]),
                        "max": float(row["max"]),
                    }
                    for k, row in stats.iterrows()
                }

    # Categories with insufficient observations
    insufficient: list[str] = []
    for prod, count in results["volume_by_product"].items():
        if count < min_observations:
            insufficient.append(f"product:{prod} ({count} obs)")
    for src, count in results["volume_by_source"].items():
        if count < min_observations:
            insufficient.append(f"source:{src} ({count} obs)")

    results["insufficient_observation_categories"] = insufficient
    return results


# ---------------------------------------------------------------------------
# Filtering and aggregation interface
# ---------------------------------------------------------------------------


def filter_by_date_range(
    df: pd.DataFrame,
    start: Optional[Any] = None,
    end: Optional[Any] = None,
    start_date: Optional[Any] = None,
    end_date: Optional[Any] = None,
    date_column: str = "created_at",
) -> pd.DataFrame:
    """
    Filter DataFrame to records within an inclusive date range.

    Accepts date strings, date objects, datetime objects, or Timestamps.
    Missing date columns or empty inputs are handled without errors.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to filter.
    start : Optional[Any]
        Start date (inclusive). Alias for start_date.
    end : Optional[Any]
        End date (inclusive). Alias for end_date.
    start_date : Optional[Any]
        Start date (inclusive).
    end_date : Optional[Any]
        End date (inclusive).
    date_column : str, default "created_at"
        Name of the date column.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame copy.
    """
    if date_column not in df.columns or len(df) == 0:
        return df.copy()

    s_val = start if start is not None else start_date
    e_val = end if end is not None else end_date

    if s_val is None and e_val is None:
        return df.copy()

    dates = pd.to_datetime(df[date_column], errors="coerce")
    mask = pd.Series(True, index=df.index)

    if s_val is not None:
        s_ts = pd.to_datetime(s_val)
        mask &= dates >= s_ts

    if e_val is not None:
        e_ts = pd.to_datetime(e_val)
        mask &= dates <= e_ts

    return df[mask].copy()


def filter_by_rating_range(
    df: pd.DataFrame,
    min_rating: Optional[float] = None,
    max_rating: Optional[float] = None,
    rating_column: str = "rating",
    rating_scale: str = "1-5",
) -> pd.DataFrame:
    """
    Filter DataFrame to records within an inclusive rating range.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to filter.
    min_rating : Optional[float]
        Minimum rating threshold.
    max_rating : Optional[float]
        Maximum rating threshold.
    rating_column : str, default "rating"
        The rating column name.
    rating_scale : str, default "1-5"
        Rating scale identifier.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame copy.
    """
    if rating_column not in df.columns or len(df) == 0:
        return df.copy()

    if min_rating is None and max_rating is None:
        return df.copy()

    ratings = pd.to_numeric(df[rating_column], errors="coerce")
    mask = pd.Series(True, index=df.index)

    if min_rating is not None:
        mask &= ratings >= min_rating
    if max_rating is not None:
        mask &= ratings <= max_rating

    return df[mask].copy()


def filter_by_language(
    df: pd.DataFrame,
    language: Union[str, list[str]],
    language_column: str = "language",
) -> pd.DataFrame:
    """
    Filter DataFrame to one or more language codes.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to filter.
    language : Union[str, list[str]]
        A single language code or a list of codes.
    language_column : str, default "language"
        Language column name.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame copy.
    """
    if language_column not in df.columns or len(df) == 0 or language is None:
        return df.copy()

    if isinstance(language, (list, tuple, set)):
        langs = {str(item).lower() for item in language}
        mask = df[language_column].astype(str).str.lower().isin(langs)
    else:
        lang_str = str(language).lower()
        mask = df[language_column].astype(str).str.lower() == lang_str

    return df[mask].copy()


def filter_by_product_service(
    df: pd.DataFrame,
    product: Union[str, list[str], None] = None,
    product_column: str = "product_or_service",
) -> pd.DataFrame:
    """
    Filter DataFrame to specific products or services.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to filter.
    product : Union[str, list[str], None]
        Product name or list of product names.
    product_column : str, default "product_or_service"
        Product/service column name.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame copy.
    """
    if product_column not in df.columns or len(df) == 0 or product is None:
        return df.copy()

    if isinstance(product, (list, tuple, set)):
        prods = {str(p) for p in product}
        mask = df[product_column].astype(str).isin(prods)
    else:
        mask = df[product_column].astype(str) == str(product)

    return df[mask].copy()


def filter_by_source(
    df: pd.DataFrame,
    source: Union[str, list[str], None] = None,
    source_column: str = "source",
) -> pd.DataFrame:
    """
    Filter DataFrame to specific feedback sources/channels.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to filter.
    source : Union[str, list[str], None]
        Source name or list of sources.
    source_column : str, default "source"
        Source column name.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame copy.
    """
    if source_column not in df.columns or len(df) == 0 or source is None:
        return df.copy()

    if isinstance(source, (list, tuple, set)):
        srcs = {str(s) for s in source}
        mask = df[source_column].astype(str).isin(srcs)
    else:
        mask = df[source_column].astype(str) == str(source)

    return df[mask].copy()


def filter_by_status(
    df: pd.DataFrame,
    include_flagged: bool = False,
    status_column: str = "_status",
) -> pd.DataFrame:
    """
    Filter records by acceptance / flag status.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to filter.
    include_flagged : bool, default False
        If True, retains both accepted and flagged records.
        If False, retains accepted records only.
    status_column : str, default "_status"
        Status column name.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame copy.
    """
    if status_column not in df.columns or len(df) == 0:
        return df.copy()

    if include_flagged:
        mask = df[status_column].isin(["accepted", "flagged"])
    else:
        mask = df[status_column] == "accepted"

    return df[mask].copy()


def filter_dataset(
    df: pd.DataFrame,
    *,
    start_date: Optional[Any] = None,
    end_date: Optional[Any] = None,
    min_rating: Optional[float] = None,
    max_rating: Optional[float] = None,
    product: Optional[Union[str, list[str]]] = None,
    source: Optional[Union[str, list[str]]] = None,
    language: Optional[Union[str, list[str]]] = None,
    include_flagged: bool = False,
    rating_scale: str = "1-5",
) -> pd.DataFrame:
    """
    Composable multi-criteria filter combining date, rating, category, source,
    language, and flagged-record inclusion.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    start_date : Optional[Any]
        Earliest date to include.
    end_date : Optional[Any]
        Latest date to include.
    min_rating : Optional[float]
        Minimum rating.
    max_rating : Optional[float]
        Maximum rating.
    product : Optional[Union[str, list[str]]]
        Target product(s).
    source : Optional[Union[str, list[str]]]
        Target source(s).
    language : Optional[Union[str, list[str]]]
        Target language(s).
    include_flagged : bool, default False
        Whether to retain flagged records.
    rating_scale : str, default "1-5"
        Active rating scale.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame copy.
    """
    filtered = df.copy()
    filtered = filter_by_status(filtered, include_flagged=include_flagged)

    if start_date is not None or end_date is not None:
        filtered = filter_by_date_range(filtered, start=start_date, end=end_date)

    if min_rating is not None or max_rating is not None:
        filtered = filter_by_rating_range(
            filtered,
            min_rating=min_rating,
            max_rating=max_rating,
            rating_scale=rating_scale,
        )

    if product is not None:
        filtered = filter_by_product_service(filtered, product=product)

    if source is not None:
        filtered = filter_by_source(filtered, source=source)

    if language is not None:
        filtered = filter_by_language(filtered, language=language)

    return filtered


def aggregate_by(
    df: pd.DataFrame,
    group_by: str,
    value: str,
    agg_func: str = "count",
    weight_by: Optional[str] = None,
) -> pd.DataFrame:
    """
    Perform a deterministic aggregation on a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The feedback DataFrame.
    group_by : str
        Column name to group by.
    value : str
        Column name to aggregate.
    agg_func : str, default "count"
        Aggregation function ("count", "mean", "median", "sum", "min", "max").
    weight_by : Optional[str], default None
        Optional weighting column for weighted averages.

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame with group column and value column.
    """
    if group_by not in df.columns or value not in df.columns or len(df) == 0:
        return pd.DataFrame(columns=[group_by, agg_func])

    # Missing values should not be treated as zeros
    valid_df = df.dropna(subset=[group_by, value])
    if len(valid_df) == 0:
        return pd.DataFrame(columns=[group_by, agg_func])

    if agg_func == "count":
        res = (
            valid_df.groupby(group_by, observed=False)[value]
            .count()
            .reset_index(name="count")
            .sort_values(by=["count", group_by], ascending=[False, True])
        )
    elif agg_func == "mean":
        if weight_by and weight_by in valid_df.columns:

            def _weighted_mean(g: pd.DataFrame) -> float:
                w_sum = g[weight_by].sum()
                return (
                    (g[value] * g[weight_by]).sum() / w_sum
                    if w_sum != 0
                    else float("nan")
                )

            res = (
                valid_df.groupby(group_by, observed=False)
                .apply(_weighted_mean)
                .reset_index(name="mean")
                .sort_values(by=["mean", group_by], ascending=[False, True])
            )
        else:
            numeric_vals = pd.to_numeric(valid_df[value], errors="coerce")
            valid_num = valid_df[numeric_vals.notna()].copy()
            valid_num[value] = numeric_vals.dropna()
            res = (
                valid_num.groupby(group_by, observed=False)[value]
                .mean()
                .reset_index(name="mean")
                .sort_values(by=["mean", group_by], ascending=[False, True])
            )
    elif agg_func == "median":
        numeric_vals = pd.to_numeric(valid_df[value], errors="coerce")
        valid_num = valid_df[numeric_vals.notna()].copy()
        valid_num[value] = numeric_vals.dropna()
        res = (
            valid_num.groupby(group_by, observed=False)[value]
            .median()
            .reset_index(name="median")
            .sort_values(by=["median", group_by], ascending=[False, True])
        )
    elif agg_func == "sum":
        numeric_vals = pd.to_numeric(valid_df[value], errors="coerce")
        valid_num = valid_df[numeric_vals.notna()].copy()
        valid_num[value] = numeric_vals.dropna()
        res = (
            valid_num.groupby(group_by, observed=False)[value]
            .sum()
            .reset_index(name="sum")
            .sort_values(by=["sum", group_by], ascending=[False, True])
        )
    else:
        res = (
            valid_df.groupby(group_by, observed=False)[value]
            .agg(agg_func)
            .reset_index(name=agg_func)
            .sort_values(by=[agg_func, group_by], ascending=[False, True])
        )

    return res.reset_index(drop=True)
