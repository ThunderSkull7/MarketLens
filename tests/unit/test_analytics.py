"""Unit tests for MarketLens exploratory data analysis, analytics, and filtering."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

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
from marketlens.cleaning import run_pipeline

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_clean_df():
    """Create a diverse, predictable cleaned DataFrame for testing analytics."""
    data = [
        {
            "review_id": "R001",
            "review_text": "Great battery life on this laptop. Highly recommended!",
            "rating": 5.0,
            "created_at": date(2024, 1, 15),
            "product_or_service": "Laptop",
            "source": "Amazon",
            "language": "en",
            "_status": "accepted",
            "is_duplicate": False,
        },
        {
            "review_id": "R002",
            "review_text": "Fast shipping, but battery life is mediocre.",
            "rating": 3.0,
            "created_at": date(2024, 1, 20),
            "product_or_service": "Laptop",
            "source": "Amazon",
            "language": "en",
            "_status": "accepted",
            "is_duplicate": False,
        },
        {
            "review_id": "R003",
            "review_text": "Headphones sound quality is outstanding. Love it!",
            "rating": 5.0,
            "created_at": date(2024, 2, 10),
            "product_or_service": "Headphones",
            "source": "BestBuy",
            "language": "en",
            "_status": "accepted",
            "is_duplicate": False,
        },
        {
            "review_id": "R004",
            "review_text": "Headphones stopped working after two weeks. Poor quality.",
            "rating": 1.0,
            "created_at": date(2024, 2, 15),
            "product_or_service": "Headphones",
            "source": "BestBuy",
            "language": "en",
            "_status": "accepted",
            "is_duplicate": False,
        },
        {
            "review_id": "R005",
            "review_text": "Très bon produit, livraison rapide.",
            "rating": 4.0,
            "created_at": date(2024, 3, 5),
            "product_or_service": "Keyboard",
            "source": "Direct",
            "language": "fr",
            "_status": "flagged",
            "is_duplicate": False,
        },
        {
            "review_id": "R006",
            "review_text": "No rating given but decent mouse.",
            "rating": None,  # Missing rating
            "created_at": None,  # Missing date
            "product_or_service": "Mouse",
            "source": "Amazon",
            "language": "en",
            "_status": "accepted",
            "is_duplicate": False,
        },
    ]
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Dataset overview tests
# ---------------------------------------------------------------------------


def test_dataset_overview(sample_clean_df):
    """Test dataset overview statistics and distributions."""
    res = dataset_overview(sample_clean_df)
    assert res["total_records"] == 6
    assert res["product_breakdown"]["Laptop"] == 2
    assert res["product_breakdown"]["Headphones"] == 2
    assert res["source_breakdown"]["Amazon"] == 3
    assert res["language_breakdown"]["en"] == 5
    assert res["language_breakdown"]["fr"] == 1

    # Date coverage
    assert res["date_coverage"]["min_date"] == "2024-01-15"
    assert res["date_coverage"]["max_date"] == "2024-03-05"
    assert res["date_coverage"]["date_range_days"] == 50

    # Missing value distribution
    assert res["missing_value_distribution"]["rating"] == 1
    assert res["missing_value_distribution"]["created_at"] == 1
    assert res["missing_value_percentages"]["rating"] == round((1 / 6) * 100.0, 2)

    # Quality summary
    assert res["quality_summary"]["total_records"] == 6
    assert res["quality_summary"]["complete_records"] == 5
    assert res["quality_summary"]["flagged_records"] == 1


def test_dataset_overview_empty():
    """Ensure dataset overview handles empty DataFrames safely."""
    res = dataset_overview(pd.DataFrame())
    assert res["total_records"] == 0
    assert res["product_breakdown"] == {}
    assert res["date_coverage"]["min_date"] is None


# ---------------------------------------------------------------------------
# Rating analysis tests
# ---------------------------------------------------------------------------


def test_rating_analysis(sample_clean_df):
    """Test rating summaries, percentages, and trend over time."""
    res = rating_analysis(sample_clean_df, rating_scale="1-5")
    # 5 rated records: 5.0, 3.0, 5.0, 1.0, 4.0
    assert res["total_rated"] == 5
    assert res["min"] == 1.0
    assert res["max"] == 5.0
    assert res["mean"] == round((5.0 + 3.0 + 5.0 + 1.0 + 4.0) / 5, 2)
    assert res["median"] == 4.0
    assert res["distribution"]["5.0"] == 2
    assert res["percentages"]["5.0"] == 40.0
    assert res["scale"] == "1-5"
    assert "2024-01" in res["trend_over_time"]


def test_rating_analysis_respects_scale():
    """Ensure rating analysis respects configured scale without assuming 1-5."""
    df = pd.DataFrame({"rating": [8.0, 9.0, 10.0]})
    res = rating_analysis(df, rating_scale="1-10")
    assert res["scale"] == "1-10"
    assert res["scale_min"] == 1.0
    assert res["scale_max"] == 10.0
    assert res["mean"] == 9.0


def test_rating_analysis_empty():
    """Test rating analysis with no valid ratings."""
    df = pd.DataFrame({"rating": [None, "invalid"]})
    res = rating_analysis(df)
    assert res["total_rated"] == 0
    assert res["mean"] is None
    assert res["distribution"] == {}


# ---------------------------------------------------------------------------
# Temporal analysis tests
# ---------------------------------------------------------------------------


def test_temporal_analysis(sample_clean_df):
    """Test volume by period and period-over-period changes."""
    res = temporal_analysis(sample_clean_df, period="M")
    vol = res["volume_by_period"]
    assert vol["2024-01"] == 2
    assert vol["2024-02"] == 2
    assert vol["2024-03"] == 1

    # Period-over-period changes
    pop = res["period_over_period_changes"]
    assert "2024-02" in pop
    assert pop["2024-02"]["compared_to"] == "2024-01"
    assert pop["2024-02"]["volume_change"] == 0

    assert "2024-03" in pop
    assert pop["2024-03"]["volume_change"] == -1
    assert pop["2024-03"]["volume_change_pct"] == -50.0

    # Trend by period
    trends = res["rating_trend_by_period"]
    assert trends["2024-01"] == 4.0  # (5 + 3) / 2
    assert trends["2024-02"] == 3.0  # (5 + 1) / 2
    assert trends["2024-03"] == 4.0  # 4.0


def test_temporal_analysis_insufficient_data():
    """Verify warnings when periods or observations are sparse."""
    df = pd.DataFrame(
        {
            "created_at": ["2024-01-01"],
            "rating": [5.0],
        }
    )
    res = temporal_analysis(df)
    assert len(res["period_warnings"]) > 0
    assert any("Insufficient" in w for w in res["period_warnings"])


def test_temporal_analysis_missing_date_column():
    """Ensure missing created_at column returns clean structure with warning."""
    df = pd.DataFrame({"rating": [5.0]})
    res = temporal_analysis(df)
    assert res["volume_by_period"] == {}
    assert len(res["period_warnings"]) > 0


# ---------------------------------------------------------------------------
# Text analysis tests
# ---------------------------------------------------------------------------


def test_text_analysis(sample_clean_df):
    """Test term frequency analysis, stopwords removal, unigrams, and bigrams."""
    res = text_analysis(sample_clean_df, min_frequency=2)
    assert "disclaimer" in res
    assert "battery" in res["unigrams"]
    assert "headphones" in res["unigrams"]
    assert "battery life" in res["bigrams"]
    # Common stopwords like 'on', 'this', 'is' should not be in unigrams
    assert "on" not in res["unigrams"]
    assert "this" not in res["unigrams"]
    assert res["total_words"] > 0
    assert res["unique_terms"] > 0


def test_text_analysis_empty():
    """Test text analysis on blank or empty input."""
    df = pd.DataFrame({"review_text": ["", None, "   "]})
    res = text_analysis(df)
    assert res["total_words"] == 0
    assert res["unigrams"] == {}


# ---------------------------------------------------------------------------
# Product/service and channel analysis tests
# ---------------------------------------------------------------------------


def test_product_service_analysis(sample_clean_df):
    """Test product and channel volume and rating statistics."""
    res = product_service_analysis(sample_clean_df, min_observations=2)
    assert res["volume_by_product"]["Laptop"] == 2
    assert res["avg_rating_by_product"]["Laptop"] == 4.0
    assert res["rating_stats_by_product"]["Laptop"]["min"] == 3.0
    assert res["rating_stats_by_product"]["Laptop"]["max"] == 5.0

    assert res["volume_by_source"]["Amazon"] == 3
    assert res["volume_by_source"]["BestBuy"] == 2

    # Categories with < 2 observations: Keyboard (1 obs), Mouse (1 obs)
    insufficient = res["insufficient_observation_categories"]
    assert any("product:Keyboard" in item for item in insufficient)
    assert any("product:Mouse" in item for item in insufficient)


# ---------------------------------------------------------------------------
# Filtering tests
# ---------------------------------------------------------------------------


def test_filter_by_date_range(sample_clean_df):
    """Test filtering by date range."""
    # Jan 2024 only
    res = filter_by_date_range(sample_clean_df, start="2024-01-01", end="2024-01-31")
    assert len(res) == 2
    assert set(res["review_id"]) == {"R001", "R002"}

    # Missing column returns copy safely
    df_no_date = pd.DataFrame({"rating": [1, 2]})
    assert len(filter_by_date_range(df_no_date, start="2024-01-01")) == 2


def test_filter_by_rating_range(sample_clean_df):
    """Test filtering by rating range."""
    res = filter_by_rating_range(sample_clean_df, min_rating=4.0, max_rating=5.0)
    # Ratings 5.0, 5.0, 4.0
    assert len(res) == 3
    assert set(res["review_id"]) == {"R001", "R003", "R005"}


def test_filter_by_language(sample_clean_df):
    """Test language filtering."""
    res_en = filter_by_language(sample_clean_df, "en")
    assert len(res_en) == 5

    res_fr = filter_by_language(sample_clean_df, ["fr"])
    assert len(res_fr) == 1
    assert res_fr.iloc[0]["review_id"] == "R005"


def test_filter_by_product_and_source(sample_clean_df):
    """Test category and channel filtering."""
    prod_res = filter_by_product_service(sample_clean_df, "Laptop")
    assert len(prod_res) == 2

    src_res = filter_by_source(sample_clean_df, "Amazon")
    assert len(src_res) == 3


def test_filter_by_status(sample_clean_df):
    """Test filtering by acceptance / flag status."""
    acc_only = filter_by_status(sample_clean_df, include_flagged=False)
    assert len(acc_only) == 5
    assert "flagged" not in acc_only["_status"].values

    with_flagged = filter_by_status(sample_clean_df, include_flagged=True)
    assert len(with_flagged) == 6


def test_filter_dataset_composition(sample_clean_df):
    """Test composable multi-criteria filter."""
    res = filter_dataset(
        sample_clean_df,
        product="Laptop",
        source="Amazon",
        min_rating=4.0,
        include_flagged=False,
    )
    # Only R001 matches: Laptop, Amazon, rating=5.0, accepted
    assert len(res) == 1
    assert res.iloc[0]["review_id"] == "R001"


def test_filter_empty_result(sample_clean_df):
    """Ensure filtering yielding 0 rows does not crash."""
    res = filter_dataset(sample_clean_df, product="NonExistent")
    assert res.empty
    assert len(res) == 0


# ---------------------------------------------------------------------------
# Aggregation tests
# ---------------------------------------------------------------------------


def test_aggregate_by_count(sample_clean_df):
    """Test deterministic count aggregation."""
    res = aggregate_by(
        sample_clean_df,
        group_by="product_or_service",
        value="review_id",
        agg_func="count",
    )
    assert len(res) == 4
    # Deterministic descending sort by count
    assert res.iloc[0]["count"] >= res.iloc[-1]["count"]


def test_aggregate_by_mean(sample_clean_df):
    """Test mean aggregation excluding null ratings."""
    res = aggregate_by(
        sample_clean_df, group_by="product_or_service", value="rating", agg_func="mean"
    )
    laptop_mean = res[res["product_or_service"] == "Laptop"]["mean"].values[0]
    assert laptop_mean == 4.0


# ---------------------------------------------------------------------------
# End-to-end reproducibility pipeline test
# ---------------------------------------------------------------------------


def test_run_pipeline_reproducibility(tmp_path):
    """Test full CSV -> Ingestion -> Cleaning -> EDA -> Analytics pipeline."""
    csv_file = Path("data/raw/synthetic_feedback_demo.csv")
    summary = run_pipeline(
        csv_file,
        include_flagged=False,
        rating_scale="1-5",
        output_dir=str(tmp_path / "processed"),
        report_dir=str(tmp_path / "reports"),
    )

    assert "csv_path" in summary
    assert "cleaning_report_path" in summary
    assert "analytics_report_path" in summary
    assert "overview" in summary
    assert "rating_analysis" in summary
    assert "temporal_analysis" in summary
    assert "text_analysis" in summary
    assert "product_service_analysis" in summary
    assert summary["row_counts"]["retained"] > 0
    assert Path(summary["csv_path"]).exists()
    assert Path(summary["cleaning_report_path"]).exists()
    assert Path(summary["analytics_report_path"]).exists()
