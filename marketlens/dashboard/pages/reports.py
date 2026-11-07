"""
MarketLens Dashboard - Reports & Export Hub.

Provides consolidated export facilities for cleaned datasets, analytics summaries,
sentiment predictions, topic clusters, and executive intelligence reports in
industry-standard CSV, JSON, and Markdown formats.
"""

from __future__ import annotations

import json
from typing import Optional

import streamlit as st

from marketlens.analytics import (
    dataset_overview,
    product_service_analysis,
    rating_analysis,
    temporal_analysis,
    text_analysis,
)
from marketlens.dashboard.components import render_empty_state
from marketlens.dashboard.state import (
    KEY_ANALYTICS_SUMMARY,
    KEY_CLEANED_DF,
    KEY_INTELLIGENCE_REPORT,
    KEY_NLP_MANAGER,
    KEY_NLP_RESULTS,
    get_active_dataframe,
    load_demo_dataset,
)
from marketlens.intelligence import IntelligenceReport
from marketlens.nlp.unified import NLPManager


def _format_markdown_report(report: IntelligenceReport) -> str:
    """Render an IntelligenceReport into clean executive markdown."""
    lines: list[str] = [
        "# MarketLens Executive Intelligence Report",
        f"**Report ID:** `{report.report_id}`  ",
        f"**Generated:** {report.generated_at}  ",
        f"**Status:** {report.status}  ",
        "",
        "---",
        "",
    ]

    if report.executive_summary:
        es = report.executive_summary
        lines.extend(
            [
                "## 1. Executive Summary",
                f"{es.overview}",
                "",
                "### Key Signals",
            ]
        )
        for sig in es.positive_signals:
            lines.append(f"- 🟢 **Positive:** {sig}")
        for iss in es.recurring_issues:
            lines.append(f"- 🔴 **Issue:** {iss}")
        if es.business_interpretation:
            lines.extend(["", f"**Strategic Context:** {es.business_interpretation}"])
        lines.append("")

    if report.sentiment_intelligence:
        si = report.sentiment_intelligence
        lines.extend(
            [
                "## 2. Sentiment Intelligence",
                f"- **Dominant Polarity:** `{si.dominant_sentiment}`",
                f"- **Distribution Summary:** {si.distribution_summary}",
                f"- **Model Limitations:** {si.model_limitations}",
                "",
            ]
        )

    if report.recommendations:
        lines.extend(["## 3. Strategic Recommendations", ""])
        for idx, rec in enumerate(report.recommendations, 1):
            lines.extend(
                [
                    f"### {idx}. [{rec.priority.upper()}] {rec.title}",
                    f"**Category:** {rec.impact_category.title()} | "
                    f"**Confidence:** {rec.confidence:.2f}",
                    "",
                    f"{rec.description}",
                    "",
                    f"**Action:** {rec.suggested_action}",
                    f"**Supporting Evidence IDs:** `{', '.join(rec.evidence_ids)}`",
                    "",
                ]
            )

    if report.limitations_disclaimer:
        lines.extend(
            [
                "---",
                "## Limitations & Audit Disclaimer",
                f"*{report.limitations_disclaimer}*",
            ]
        )

    return "\n".join(lines)


def render_reports_page() -> None:
    """Render the Reports & Export Hub page."""
    st.title("📦 Reports & Data Export Hub")
    st.markdown(
        "Download data artifacts, analytical summaries, model scoring, "
        "and executive AI reports in structured formats."
    )

    df = get_active_dataframe()
    if df is None or df.empty:
        render_empty_state(
            title="No Dataset Loaded",
            message=(
                "Please upload a feedback dataset or load the demo "
                "dataset to export reports."
            ),
            icon="📦",
            action_button_label="Load Demo Dataset",
            on_action=lambda: (load_demo_dataset(), st.rerun()),
        )
        return

    cleaned_df = st.session_state.get(KEY_CLEANED_DF)
    manager: Optional[NLPManager] = st.session_state.get(KEY_NLP_MANAGER)
    nlp_results = st.session_state.get(KEY_NLP_RESULTS)
    report: Optional[IntelligenceReport] = st.session_state.get(KEY_INTELLIGENCE_REPORT)

    col1, col2 = st.columns(2)

    # 1. Cleaned Data Export
    with col1:
        st.subheader("🧹 Cleaned Dataset")
        st.markdown(
            "Validated and normalized feedback records, free of duplicates "
            "and unrecoverable formatting errors."
        )
        if cleaned_df is not None and not cleaned_df.empty:
            st.caption(
                f"Ready: {len(cleaned_df)} records | {len(cleaned_df.columns)} columns"
            )
            csv_data = cleaned_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Cleaned Dataset (CSV)",
                data=csv_data,
                file_name="marketlens_cleaned_feedback.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.info(
                "Cleaned dataset not available. Run cleaning pipeline in Data Upload."
            )

    # 2. Marketing Analytics Summary Export
    with col2:
        st.subheader("📊 Marketing Analytics Summary")
        st.markdown(
            "Consolidated statistical summary across ratings, periods, "
            "product categories, and n-gram vocabulary."
        )
        analytics_sum = st.session_state.get(KEY_ANALYTICS_SUMMARY)
        if analytics_sum is None:
            analytics_sum = {
                "overview": dataset_overview(df),
                "rating": rating_analysis(df) if "rating" in df.columns else {},
                "temporal": temporal_analysis(df)
                if "created_at" in df.columns or "date" in df.columns
                else {},
                "product_service": product_service_analysis(df)
                if "product_or_service" in df.columns
                else {},
                "text": text_analysis(df) if "review_text" in df.columns else {},
            }
            st.session_state[KEY_ANALYTICS_SUMMARY] = analytics_sum

        analytics_json = json.dumps(analytics_sum, indent=2, default=str)
        st.download_button(
            "⬇️ Download Analytics Summary (JSON)",
            data=analytics_json,
            file_name="marketlens_analytics_summary.json",
            mime="application/json",
            use_container_width=True,
        )

    st.divider()

    col3, col4 = st.columns(2)

    # 3. Sentiment Model Predictions Export
    with col3:
        st.subheader("🎭 Sentiment Predictions")
        st.markdown(
            "Record-level sentiment classifications, confidence scores, "
            "and prediction uncertainty flags."
        )
        if manager and manager.sentiment_model and manager.sentiment_model.is_fitted:
            scored_df = manager.sentiment_model.predict_batch(
                df, text_column="review_text", id_column="review_id"
            )
            s_csv = scored_df.to_csv(index=False).encode("utf-8")
            st.caption(
                f"Ready: {len(scored_df)} scored records with confidence & margin"
            )
            st.download_button(
                "⬇️ Download Sentiment Predictions (CSV)",
                data=s_csv,
                file_name="marketlens_sentiment_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.info(
                "Sentiment model not fitted. Train model in Sentiment Analysis first."
            )

    # 4. Topic Discovery Clusters Export
    with col4:
        st.subheader("🔍 Topic Discovery Summary")
        st.markdown(
            "K-Means cluster summaries, top keywords, silhouette metrics, "
            "and representative customer feedback."
        )
        if nlp_results and "topic_clusters" in nlp_results:
            topics_data = {
                "topic_clusters": nlp_results["topic_clusters"],
                "metadata": nlp_results.get("topic_metadata", {}),
            }
            t_json = json.dumps(topics_data, indent=2, default=str)
            st.caption(f"Ready: {len(nlp_results['topic_clusters'])} discovered themes")
            st.download_button(
                "⬇️ Download Topic Summary (JSON)",
                data=t_json,
                file_name="marketlens_topic_discovery.json",
                mime="application/json",
                use_container_width=True,
            )
        else:
            st.info(
                "Topic discovery not executed. "
                "Run clustering in Topic & Theme Discovery."
            )

    st.divider()

    # 5. AI Marketing Intelligence Executive Report
    st.subheader("🤖 AI Marketing Intelligence Report")
    st.markdown(
        "Complete synthesized intelligence report including Executive Summary, "
        "Sentiment Insights, Topic Analysis, and Prioritized Recommendations."
    )
    if report and report.status == "completed":
        st.caption(f"Ready: Report `{report.report_id}` ({report.generated_at})")
        r_col1, r_col2 = st.columns(2)
        with r_col1:
            rep_json = report.model_dump_json(indent=2)
            st.download_button(
                "⬇️ Download Executive Report (JSON)",
                data=rep_json,
                file_name=f"marketlens_intelligence_{report.report_id}.json",
                mime="application/json",
                use_container_width=True,
            )
        with r_col2:
            rep_md = _format_markdown_report(report)
            st.download_button(
                "⬇️ Download Executive Brief (Markdown)",
                data=rep_md,
                file_name=f"marketlens_executive_brief_{report.report_id}.md",
                mime="text/markdown",
                use_container_width=True,
            )
    else:
        st.info(
            "No completed intelligence report available. "
            "Generate an AI report in AI Marketing Intelligence."
        )
