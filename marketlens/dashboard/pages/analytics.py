"""
MarketLens Dashboard - Marketing Analytics Page.

Displays rating distributions, temporal trends, product/service comparisons,
and vocabulary/text analysis using backend analytics functions.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from marketlens.analytics import (
    filter_dataset,
    product_service_analysis,
    rating_analysis,
    temporal_analysis,
    text_analysis,
)
from marketlens.dashboard.components import (
    render_bar_chart,
    render_empty_state,
    render_kpi_card,
)
from marketlens.dashboard.state import get_active_dataframe, load_demo_dataset


def render_analytics_page() -> None:
    """Render the Marketing Analytics page."""
    st.title("📈 Marketing Analytics & EDA")
    st.markdown(
        "Explore multidimensional customer feedback analytics across ratings, "
        "temporal dynamics, product comparisons, and customer vocabulary."
    )

    df = get_active_dataframe()
    if df is None or df.empty:
        render_empty_state(
            title="No Dataset Loaded",
            message=(
                "Please upload a feedback dataset or load the demo "
                "dataset to view analytics."
            ),
            icon="📊",
            action_button_label="Load Demo Dataset",
            on_action=lambda: (load_demo_dataset(), st.rerun()),
        )
        return

    # Determine standard columns
    rating_col = "rating" if "rating" in df.columns else None
    date_col = (
        "created_at"
        if "created_at" in df.columns
        else ("date" if "date" in df.columns else None)
    )
    product_col = (
        "product_or_service"
        if "product_or_service" in df.columns
        else ("product" if "product" in df.columns else None)
    )
    source_col = "source" if "source" in df.columns else None
    text_col = "review_text" if "review_text" in df.columns else None

    # 1. Interactive Filters
    with st.expander("🎯 Filter Dataset", expanded=True):
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)

        # Rating Filter
        min_r, max_r = 1.0, 5.0
        with f_col1:
            if rating_col:
                min_val = float(df[rating_col].min(skipna=True))
                max_val = float(df[rating_col].max(skipna=True))
                r_range = st.slider(
                    "Rating Range",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                    step=0.5,
                )
                min_r, max_r = r_range
            else:
                st.caption("No rating column available.")

        # Product Filter
        selected_prods = None
        with f_col2:
            if product_col:
                avail_prods = sorted(df[product_col].dropna().unique().tolist())
                selected_prods = st.multiselect(
                    "Product / Service",
                    options=avail_prods,
                    default=[],
                    help="Leave blank to include all products",
                )
            else:
                st.caption("No product column available.")

        # Source Filter
        selected_sources = None
        with f_col3:
            if source_col:
                avail_sources = sorted(df[source_col].dropna().unique().tolist())
                selected_sources = st.multiselect(
                    "Feedback Source",
                    options=avail_sources,
                    default=[],
                    help="Leave blank to include all sources",
                )
            else:
                st.caption("No source column available.")

        # Date Filter
        start_d, end_d = None, None
        with f_col4:
            if date_col:
                date_series = pd.to_datetime(df[date_col], errors="coerce").dropna()
                if not date_series.empty:
                    d_min = date_series.min().date()
                    d_max = date_series.max().date()
                    d_range = st.date_input(
                        "Date Range",
                        value=(d_min, d_max),
                        min_value=d_min,
                        max_value=d_max,
                    )
                    if isinstance(d_range, (list, tuple)) and len(d_range) == 2:
                        start_d, end_d = d_range
                else:
                    st.caption("Date format unrecognized.")
            else:
                st.caption("No date column available.")

    # Apply backend filter
    filtered_df = filter_dataset(
        df,
        min_rating=min_r if rating_col else None,
        max_rating=max_r if rating_col else None,
        product=selected_prods if selected_prods else None,
        source=selected_sources if selected_sources else None,
        start_date=start_d,
        end_date=end_d,
    )

    st.caption(
        f"Displaying **{len(filtered_df)}** of **{len(df)}** records "
        "based on active filters."
    )

    if filtered_df.empty:
        st.warning(
            "No records match the selected filter criteria. "
            "Please broaden your selection."
        )
        return

    st.divider()

    # 2. Rating Analytics Section
    if rating_col:
        st.subheader("⭐ Rating Dynamics & Sentiment Proxy")
        r_stats = rating_analysis(filtered_df, rating_column=rating_col)

        rc1, rc2, rc3, rc4 = st.columns(4)
        with rc1:
            render_kpi_card(
                "Mean Rating", r_stats.get("mean", "N/A"), "Arithmetic average"
            )
        with rc2:
            render_kpi_card(
                "Median Rating", r_stats.get("median", "N/A"), "50th percentile"
            )
        with rc3:
            render_kpi_card(
                "Standard Dev.", r_stats.get("std_dev", "N/A"), "Dispersion"
            )
        with rc4:
            nps = r_stats.get("nps_estimate")
            render_kpi_card(
                "NPS Estimate",
                f"{nps:+.1f}" if nps is not None else "N/A",
                "Promoters - Detractors proxy",
            )

        dist = r_stats.get("distribution", {})
        if dist:
            plot_df = pd.DataFrame(
                [{"Rating": str(k), "Count": v} for k, v in dist.items()]
            )
            render_bar_chart(
                plot_df, x_col="Rating", y_col="Count", title="Rating Distribution"
            )

        st.divider()

    # 3. Temporal Trends
    if date_col:
        st.subheader("⏳ Temporal Trajectory")
        t_stats = temporal_analysis(filtered_df, date_column=date_col, period="M")

        vol_period = t_stats.get("volume_by_period", {})
        trend_period = t_stats.get("rating_trend_by_period", {})

        if vol_period:
            periods = sorted(vol_period.keys())
            chart_df = pd.DataFrame(
                {
                    "Period": periods,
                    "Volume": [vol_period.get(p, 0) for p in periods],
                    "Avg Rating": [trend_period.get(p, None) for p in periods],
                }
            )

            tc1, tc2 = st.columns(2)
            with tc1:
                render_bar_chart(
                    chart_df,
                    x_col="Period",
                    y_col="Volume",
                    title="Feedback Ingestion Volume by Period",
                )
            with tc2:
                # Line chart for average rating over time
                if (
                    "Avg Rating" in chart_df.columns
                    and chart_df["Avg Rating"].notna().any()
                ):
                    line_chart = (
                        alt.Chart(chart_df.dropna(subset=["Avg Rating"]))
                        .mark_line(point=True, color="#10B981")
                        .encode(
                            x=alt.X("Period:N", title="Period"),
                            y=alt.Y(
                                "Avg Rating:Q",
                                scale=alt.Scale(domain=[1, 5]),
                                title="Avg Rating",
                            ),
                            tooltip=["Period", "Avg Rating"],
                        )
                        .properties(title="Average Customer Rating Trend", height=260)
                    )
                    st.altair_chart(line_chart, use_container_width=True)

        if t_stats.get("period_warnings"):
            for w in t_stats["period_warnings"]:
                st.caption(f"ℹ️ {w}")

        st.divider()

    # 4. Product / Service & Source Comparison
    if product_col or source_col:
        st.subheader("🏢 Comparative Segmentation")
        p_stats = product_service_analysis(
            filtered_df,
            product_column=product_col or "product_or_service",
            source_column=source_col or "source",
            min_observations=1,
        )

        col_prod, col_src = st.columns(2)
        with col_prod:
            prod_vol = p_stats.get("volume_by_product", {})
            if prod_vol:
                p_chart_df = pd.DataFrame(
                    [{"Product": str(k), "Volume": v} for k, v in prod_vol.items()]
                )
                render_bar_chart(
                    p_chart_df,
                    x_col="Product",
                    y_col="Volume",
                    title="Volume by Product/Service",
                    horizontal=True,
                )
            else:
                st.caption("No product breakdown available.")

        with col_src:
            src_vol = p_stats.get("volume_by_source", {})
            if src_vol:
                s_chart_df = pd.DataFrame(
                    [{"Source": str(k), "Volume": v} for k, v in src_vol.items()]
                )
                render_bar_chart(
                    s_chart_df,
                    x_col="Source",
                    y_col="Volume",
                    title="Volume by Ingestion Source",
                    horizontal=True,
                )
            else:
                st.caption("No source breakdown available.")

        st.divider()

    # 5. Customer Vocabulary & Text Analysis
    if text_col:
        st.subheader("💬 Customer Vocabulary & N-Gram Frequencies")
        t_res = text_analysis(filtered_df, text_column=text_col, max_terms=15)

        uni = t_res.get("unigrams", {})
        bi = t_res.get("bigrams", {})

        col_uni, col_bi = st.columns(2)
        with col_uni:
            if uni:
                uni_df = pd.DataFrame(
                    [{"Term": k, "Frequency": v} for k, v in list(uni.items())[:12]]
                )
                render_bar_chart(
                    uni_df,
                    x_col="Term",
                    y_col="Frequency",
                    title="Top Unigrams (Single Words)",
                    horizontal=True,
                )
            else:
                st.caption("No unigrams discovered above frequency threshold.")

        with col_bi:
            if bi:
                bi_df = pd.DataFrame(
                    [{"Phrase": k, "Frequency": v} for k, v in list(bi.items())[:12]]
                )
                render_bar_chart(
                    bi_df,
                    x_col="Phrase",
                    y_col="Frequency",
                    title="Top Bigrams (Two-Word Phrases)",
                    horizontal=True,
                )
            else:
                st.caption("No bigrams discovered above frequency threshold.")

        if t_res.get("disclaimer"):
            st.caption(f"📝 {t_res['disclaimer']}")
