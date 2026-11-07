"""
MarketLens - Context Preparation & Data Minimization.

Prepares compact, privacy-conscious prompt context from existing analytics,
topic models, and stratified review samples. Establishes the authoritative
set of allowed evidence IDs prior to LLM dispatch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from marketlens.cleaning import normalize_text

logger = logging.getLogger("marketlens.intelligence.context")


@dataclass
class IntelligenceContext:
    """Prepared context payload ready for LLM prompt construction."""

    prompt_context_text: str
    allowed_evidence_ids: list[str]
    review_evidence_map: dict[str, dict[str, Any]] = field(default_factory=dict)
    metric_evidence_map: dict[str, Any] = field(default_factory=dict)
    total_dataset_rows: int = 0
    sampled_review_count: int = 0


def _truncate_text(text: Any, max_chars: int) -> str:
    """Normalize and cleanly truncate review excerpt without cutting words abruptly."""
    clean = normalize_text(text)
    if len(clean) <= max_chars:
        return clean
    truncated = clean[:max_chars].rsplit(" ", 1)[0]
    return f"{truncated}..."


def prepare_intelligence_context(
    df: pd.DataFrame,
    *,
    analytics_summary: Optional[dict[str, Any]] = None,
    nlp_results: Optional[dict[str, Any]] = None,
    max_reviews: int = 30,
    max_review_chars: int = 300,
    id_column: str = "review_id",
    text_column: str = "review_text",
    rating_column: str = "rating",
    sentiment_column: str = "sentiment",
    product_column: str = "product_id",
) -> IntelligenceContext:
    """
    Construct a data-minimized, evidence-grounded context for intelligence.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned customer feedback dataset.
    analytics_summary : Optional[dict[str, Any]]
        Precomputed metrics (overview, rating_analysis, temporal_analysis).
    nlp_results : Optional[dict[str, Any]]
        Precomputed NLP outputs (evaluation results, topic clusters).
    max_reviews : int, default 30
        Maximum representative reviews to include in the prompt context.
    max_review_chars : int, default 300
        Maximum length per review text excerpt.
    id_column : str
        Review identifier column.
    text_column : str
        Review text column.
    rating_column : str
        Rating column.
    sentiment_column : str
        Sentiment column.
    product_column : str
        Product / service column.

    Returns
    -------
    IntelligenceContext
        Context structure containing prompt text and validated allowed evidence IDs.
    """
    allowed_evidence_ids: list[str] = []
    review_evidence_map: dict[str, dict[str, Any]] = {}
    metric_evidence_map: dict[str, Any] = {}

    sections: list[str] = []

    # 1. Dataset Overview & Aggregate Metrics
    sections.append("### 1. Dataset Overview & Aggregate Metrics")
    total_rows = len(df)
    overview_metric_id = "METRIC:dataset_overview.total_records"
    allowed_evidence_ids.append(overview_metric_id)
    metric_evidence_map[overview_metric_id] = total_rows

    if analytics_summary and "overview" in analytics_summary:
        ov = analytics_summary["overview"]
        total_rec = ov.get("total_records", ov.get("total_rows", total_rows))
        sections.append(f"- Total Records: {total_rec} [{overview_metric_id}]")
        health = ov.get("quality_summary", ov.get("data_health", {}))
        if health:
            complete_recs = health.get("complete_records", "N/A")
            flagged_recs = health.get("flagged_records", "N/A")
            sections.append(f"- Complete Records: {complete_recs}")
            sections.append(f"- Flagged Records: {flagged_recs}")
    else:
        sections.append(f"- Total Records: {total_rows} [{overview_metric_id}]")

    # Rating Analytics
    if analytics_summary and "rating_analysis" in analytics_summary:
        ra = analytics_summary["rating_analysis"]
        mean_id = "METRIC:rating.mean"
        nps_id = "METRIC:rating.nps"
        allowed_evidence_ids.extend([mean_id, nps_id])
        metric_evidence_map[mean_id] = ra.get("mean")
        metric_evidence_map[nps_id] = ra.get("nps_estimate")

        sections.append(f"- Mean Rating: {ra.get('mean')} [{mean_id}]")
        sections.append(
            f"- Median Rating: {ra.get('median')} | Std Dev: {ra.get('std')}"
        )
        if ra.get("nps_estimate") is not None:
            sections.append(
                f"- Estimated NPS Proxy: {ra.get('nps_estimate')} [{nps_id}]"
            )
        if "distribution" in ra:
            dist_str = ", ".join(f"{k}: {v}" for k, v in ra["distribution"].items())
            sections.append(f"- Rating Breakdown: {dist_str}")

    # Temporal Trends
    if analytics_summary and "temporal_analysis" in analytics_summary:
        ta = analytics_summary["temporal_analysis"]
        temp_id = "METRIC:temporal.date_span"
        allowed_evidence_ids.append(temp_id)
        metric_evidence_map[temp_id] = {
            "earliest": ta.get("earliest_date"),
            "latest": ta.get("latest_date"),
        }
        earliest = ta.get("earliest_date", "N/A")
        latest = ta.get("latest_date", "N/A")
        sections.append(f"- Date Span: {earliest} to {latest} [{temp_id}]")
        if "trend" in ta:
            sections.append(f"- Feedback Volume Trend: {ta.get('trend', 'stable')}")

    # Product Breakdown (sample size guard >= 3)
    if analytics_summary and "product_analysis" in analytics_summary:
        pa = analytics_summary["product_analysis"]
        by_prod = pa.get("by_product", {})
        if by_prod:
            sections.append("\n### 2. Product Comparisons (Sample Sizes & Ratings)")
            for prod_name, p_data in list(by_prod.items())[:6]:
                p_cnt = p_data.get("count", 0)
                p_mean = p_data.get("mean_rating", "N/A")
                p_metric_id = f"METRIC:product.{prod_name}"
                allowed_evidence_ids.append(p_metric_id)
                metric_evidence_map[p_metric_id] = p_data
                is_adequate = "adequate sample" if p_cnt >= 5 else "insufficient sample"
                sections.append(
                    f"- {prod_name}: N={p_cnt} ({is_adequate}), "
                    f"Mean Rating={p_mean} [{p_metric_id}]"
                )

    # NLP Topics & Sentiment Distribution
    if nlp_results:
        sections.append("\n### 3. Topic Discovery Clusters")
        clusters = nlp_results.get("topic_clusters", [])
        for cl in clusters:
            cid = cl.get("cluster_id")
            c_metric_id = f"METRIC:topic.cluster_{cid}"
            allowed_evidence_ids.append(c_metric_id)
            metric_evidence_map[c_metric_id] = cl

            keywords = ", ".join(cl.get("top_keywords", [])[:6])
            size = cl.get("size", 0)
            pct = cl.get("percentage", 0.0)
            mean_r = cl.get("mean_rating", "N/A")
            sections.append(
                f"- Cluster {cid} [{c_metric_id}]: {size} reviews ({pct}%), "
                f"Mean Rating={mean_r}, Top Terms: [{keywords}]"
            )

    # 4. Stratified Representative Customer Reviews
    sections.append(
        "\n### 4. Representative Customer Reviews (Sampled with Verified IDs)"
    )
    sections.append(
        "Note: Every quote or observation in your insights must cite one of these "
        "exact review IDs."
    )

    sampled_rows: list[pd.Series] = []
    if not df.empty and id_column in df.columns and text_column in df.columns:
        # Filter non-empty texts
        valid_df = df[df[text_column].fillna("").astype(str).str.strip() != ""].copy()

        if len(valid_df) > 0:
            # Stratified sampling across sentiment or ratings to ensure balance
            strat_col = (
                sentiment_column if sentiment_column in valid_df.columns else None
            )
            if not strat_col and rating_column in valid_df.columns:
                strat_col = rating_column

            if strat_col and valid_df[strat_col].nunique() > 1:
                per_group = max(1, max_reviews // max(1, valid_df[strat_col].nunique()))
                sampled_indices: list[Any] = []
                for _, group in valid_df.groupby(strat_col):
                    sampled_indices.extend(group.head(per_group).index)
                sampled_df = valid_df.loc[sampled_indices].head(max_reviews)
            else:
                sampled_df = valid_df.head(max_reviews)

            for _, row in sampled_df.iterrows():
                rid = str(row[id_column])
                raw_text = str(row[text_column])
                rating_val = row.get(rating_column, "N/A")
                sent_val = row.get(sentiment_column, "unlabeled")
                prod_val = row.get(product_column, None)

                truncated = _truncate_text(raw_text, max_review_chars)
                allowed_evidence_ids.append(rid)

                review_evidence_map[rid] = {
                    "review_id": rid,
                    "rating": rating_val,
                    "sentiment": sent_val,
                    "text": truncated,
                    "product": prod_val,
                }

                prod_str = f" | Product: {prod_val}" if prod_val else ""
                sections.append(
                    f"- [{rid}] (Rating: {rating_val} | "
                    f'Sentiment: {sent_val}{prod_str}): "{truncated}"'
                )
                sampled_rows.append(row)

    prompt_text = "\n".join(sections)

    return IntelligenceContext(
        prompt_context_text=prompt_text,
        allowed_evidence_ids=allowed_evidence_ids,
        review_evidence_map=review_evidence_map,
        metric_evidence_map=metric_evidence_map,
        total_dataset_rows=total_rows,
        sampled_review_count=len(sampled_rows),
    )
