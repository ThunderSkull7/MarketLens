"""
MarketLens Dashboard - Overview Page.

Presents high-level marketing intelligence KPIs, rating & sentiment distributions,
topic summaries, and data quality indicators with zero fabricated metrics.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from marketlens.analytics import rating_analysis
from marketlens.dashboard.components import (
    render_bar_chart,
    render_empty_state,
    render_kpi_card,
)
from marketlens.dashboard.state import (
    KEY_CLEANED_DF,
    KEY_INGESTION_RESULT,
    KEY_INTELLIGENCE_REPORT,
    KEY_NLP_RESULTS,
    KEY_RAW_DF,
    get_active_dataframe,
    load_demo_dataset,
)


def render_overview_page() -> None:
    """Render the Overview Dashboard view."""
    st.title("📊 Executive Marketing Overview")
    st.markdown(
        "Consolidated marketing intelligence synthesis across customer ratings, "
        "data quality health, sentiment polarity, and discovered themes."
    )

    df = get_active_dataframe()

    if df is None or df.empty:
        render_empty_state(
            title="No Dataset Loaded",
            message=(
                "Upload a customer feedback CSV file in **Data Upload & Quality** "
                "or load our pre-packaged synthetic feedback demo dataset to explore."
            ),
            icon="📥",
            action_button_label="Load Demo Dataset",
            on_action=lambda: (load_demo_dataset(), st.rerun()),
        )
        return

    # 1. Top KPI Row: Record counts & Data Health
    ingestion_res = st.session_state.get(KEY_INGESTION_RESULT)
    cleaned_df = st.session_state.get(KEY_CLEANED_DF)
    raw_df = st.session_state.get(KEY_RAW_DF)

    total_records = len(raw_df) if raw_df is not None else len(df)
    accepted_count = len(cleaned_df) if cleaned_df is not None else total_records
    rejected_count = len(ingestion_res.rejected) if ingestion_res is not None else 0
    flagged_count = len(ingestion_res.flagged) if ingestion_res is not None else 0

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        render_kpi_card("Total Source Records", total_records, "Raw ingest volume")
    with kpi2:
        render_kpi_card("Accepted Clean", accepted_count, "Analysis-ready records")
    with kpi3:
        render_kpi_card(
            "Flagged Rows",
            flagged_count,
            "Quality warnings",
            delta=f"-{flagged_count}" if flagged_count > 0 else "0",
        )
    with kpi4:
        render_kpi_card(
            "Rejected Rows",
            rejected_count,
            "Omitted from analysis",
            delta=f"-{rejected_count}" if rejected_count > 0 else "0",
        )

    st.divider()

    # 2. Rating & Sentiment Overview
    col_rating, col_sentiment = st.columns(2)

    with col_rating:
        st.subheader("⭐ Customer Ratings")
        if "rating" in df.columns:
            r_stats = rating_analysis(df, rating_column="rating")
            mean_rating = r_stats.get("mean")
            median_rating = r_stats.get("median")
            nps_est = r_stats.get("nps_estimate")

            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                render_kpi_card(
                    "Average Rating", mean_rating if mean_rating is not None else "N/A"
                )
            with m_col2:
                render_kpi_card(
                    "Median Rating",
                    median_rating if median_rating is not None else "N/A",
                )
            with m_col3:
                render_kpi_card(
                    "Est. NPS Proxy",
                    f"{nps_est:+.1f}" if nps_est is not None else "N/A",
                    help_text="Calculated as % Promoters - % Detractors",
                )

            dist = r_stats.get("distribution", {})
            if dist:
                plot_data = pd.DataFrame(
                    [{"Rating": str(k), "Count": v} for k, v in dist.items()]
                )
                render_bar_chart(
                    plot_data,
                    x_col="Rating",
                    y_col="Count",
                    title="Rating Distribution",
                )
        else:
            st.info("No 'rating' column found in active dataset.")

    with col_sentiment:
        st.subheader("🎭 Sentiment Distribution")
        nlp_res = st.session_state.get(KEY_NLP_RESULTS)

        # Check if sentiment column is in DF (either pre-existing or scored)
        sent_col = (
            "sentiment"
            if "sentiment" in df.columns
            else (
                "predicted_sentiment" if "predicted_sentiment" in df.columns else None
            )
        )

        if sent_col:
            sent_counts = df[sent_col].dropna().value_counts().to_dict()
            s_plot_data = pd.DataFrame(
                [
                    {"Sentiment": str(k).title(), "Volume": v}
                    for k, v in sent_counts.items()
                ]
            )
            render_bar_chart(
                s_plot_data,
                x_col="Sentiment",
                y_col="Volume",
                title="Class Distribution",
            )
            st.caption(
                "Source: "
                + (
                    "Model Predictions"
                    if "predicted" in sent_col
                    else "Rating-Derived Weak Labels"
                )
            )
        elif nlp_res and "baseline_evaluation" in nlp_res:
            dist = nlp_res["baseline_evaluation"].get("class_distribution", {})
            s_plot_data = pd.DataFrame(
                [{"Sentiment": str(k).title(), "Volume": v} for k, v in dist.items()]
            )
            render_bar_chart(
                s_plot_data,
                x_col="Sentiment",
                y_col="Volume",
                title="Model Class Distribution",
            )
        else:
            st.info(
                "Sentiment analysis not yet evaluated on this dataset. "
                "Navigate to **Sentiment Analysis** to train or load a classifier."
            )

    st.divider()

    # 3. Topic & Qualitative Summary
    col_topics, col_intel = st.columns(2)

    with col_topics:
        st.subheader("🔍 Discovered Customer Themes")
        if nlp_res and "topic_clusters" in nlp_res:
            clusters = nlp_res["topic_clusters"]
            render_kpi_card(
                "Discovered Topics", len(clusters), "Unsupervised K-Means clusters"
            )
            for cl in clusters[:4]:
                cid = cl.get("cluster_id")
                keywords = ", ".join(cl.get("top_keywords", [])[:5])
                pct = cl.get("percentage", 0.0)
                mean_r = cl.get("mean_rating", "N/A")
                st.markdown(
                    f"**Theme {cid}** ({pct:.1f}% of reviews | Rating: {mean_r}):  \n"
                    f"`{keywords}`"
                )
        else:
            st.info(
                "Unsupervised topic discovery has not been executed yet. "
                "Navigate to **Topic & Theme Discovery** to generate themes."
            )

    with col_intel:
        st.subheader("🤖 AI Intelligence Highlights")
        intel_rep = st.session_state.get(KEY_INTELLIGENCE_REPORT)
        if (
            intel_rep
            and intel_rep.status == "completed"
            and intel_rep.executive_summary
        ):
            es = intel_rep.executive_summary
            st.markdown(f"**Synthesis:** {es.overview}")
            if es.positive_signals:
                st.markdown(f"🟢 **Signals:** {es.positive_signals[0]}")
            if es.recurring_issues:
                st.markdown(f"🔴 **Issues:** {es.recurring_issues[0]}")
            if intel_rep.recommendations:
                top_rec = intel_rep.recommendations[0]
                p_label = top_rec.priority.upper()
                st.success(f"**Top Recommendation ({p_label}):** {top_rec.title}")
        else:
            st.info(
                "No active AI intelligence report. "
                "Generate evidence-grounded insights in **AI Marketing Intelligence**."
            )
