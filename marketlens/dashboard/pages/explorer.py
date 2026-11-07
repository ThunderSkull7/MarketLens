"""
MarketLens Dashboard - Customer Feedback Explorer Page.

Enables granular searching, filtering, inspecting, and exporting individual
customer feedback records across ratings, sentiments, topics, and sources.
"""

from __future__ import annotations

import streamlit as st

from marketlens.analytics import filter_dataset
from marketlens.dashboard.components import (
    paginate_dataframe,
    render_empty_state,
    render_kpi_card,
)
from marketlens.dashboard.state import (
    KEY_NLP_MANAGER,
    get_active_dataframe,
    load_demo_dataset,
)
from marketlens.nlp.unified import NLPManager


def render_explorer_page() -> None:
    """Render the Customer Feedback Explorer page."""
    st.title("🔎 Customer Feedback Explorer")
    st.markdown(
        "Search, filter, inspect, and export individual customer feedback records "
        "with enriched sentiment, topics, and metadata."
    )

    df = get_active_dataframe()
    if df is None or df.empty:
        render_empty_state(
            title="No Dataset Loaded",
            message=(
                "Please upload a feedback dataset or load the demo "
                "dataset to browse records."
            ),
            icon="🔎",
            action_button_label="Load Demo Dataset",
            on_action=lambda: (load_demo_dataset(), st.rerun()),
        )
        return

    work_df = df.copy()

    # Enrich with model predictions if available
    manager: NLPManager = st.session_state.get(KEY_NLP_MANAGER) or NLPManager()
    if manager.sentiment_model and manager.sentiment_model.is_fitted:
        if (
            "predicted_sentiment" not in work_df.columns
            and "review_text" in work_df.columns
        ):
            try:
                work_df = manager.sentiment_model.predict_batch(
                    work_df, text_column="review_text", id_column="review_id"
                )
            except Exception:
                pass

    if manager.topic_model and manager.topic_model.is_fitted:
        if "cluster_id" not in work_df.columns and "review_text" in work_df.columns:
            try:
                texts = work_df["review_text"].fillna("").tolist()
                work_df["cluster_id"] = manager.topic_model.assign_clusters(texts)
            except Exception:
                pass

    # 1. Search Bar
    search_query = st.text_input(
        "🔍 Keyword Search (case-insensitive substring in review text)",
        value="",
        placeholder="e.g. shipping, broken, battery, customer service...",
    )

    # 2. Filter Controls
    with st.expander("⚙️ Refine & Filter Criteria", expanded=False):
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            rating_col = "rating" if "rating" in work_df.columns else None
            min_r, max_r = None, None
            if rating_col:
                min_v = float(work_df[rating_col].min(skipna=True))
                max_v = float(work_df[rating_col].max(skipna=True))
                r_range = st.slider(
                    "Rating",
                    min_value=min_v,
                    max_value=max_v,
                    value=(min_v, max_v),
                    step=0.5,
                    key="exp_r_slider",
                )
                min_r, max_r = r_range

        with c2:
            sent_col = (
                "predicted_sentiment"
                if "predicted_sentiment" in work_df.columns
                else ("sentiment" if "sentiment" in work_df.columns else None)
            )
            sel_sents = []
            if sent_col:
                avail_sents = sorted(work_df[sent_col].dropna().unique().tolist())
                sel_sents = st.multiselect(
                    "Sentiment",
                    options=avail_sents,
                    default=[],
                    key="exp_sent_select",
                )

        with c3:
            cluster_col = "cluster_id" if "cluster_id" in work_df.columns else None
            sel_clusters = []
            if cluster_col:
                avail_cids = sorted(work_df[cluster_col].dropna().unique().tolist())
                sel_clusters = st.multiselect(
                    "Topic Cluster",
                    options=[f"Theme {cid}" for cid in avail_cids],
                    default=[],
                    key="exp_cid_select",
                )

        with c4:
            src_col = "source" if "source" in work_df.columns else None
            sel_sources = []
            if src_col:
                avail_srcs = sorted(work_df[src_col].dropna().unique().tolist())
                sel_sources = st.multiselect(
                    "Source",
                    options=avail_srcs,
                    default=[],
                    key="exp_src_select",
                )

    # Apply filters
    filtered_df = work_df

    # Rating filter
    if rating_col and min_r is not None and max_r is not None:
        filtered_df = filter_dataset(filtered_df, min_rating=min_r, max_rating=max_r)

    # Sentiment filter
    if sent_col and sel_sents:
        filtered_df = filtered_df[filtered_df[sent_col].isin(sel_sents)]

    # Cluster filter
    if cluster_col and sel_clusters:
        parsed_cids = [int(s.replace("Theme ", "")) for s in sel_clusters]
        filtered_df = filtered_df[filtered_df[cluster_col].isin(parsed_cids)]

    # Source filter
    if src_col and sel_sources:
        filtered_df = filtered_df[filtered_df[src_col].isin(sel_sources)]

    # Text search
    if search_query.strip() and "review_text" in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df["review_text"]
            .fillna("")
            .str.contains(search_query.strip(), case=False, regex=False)
        ]

    # Metrics Row
    m1, m2, m3 = st.columns(3)
    with m1:
        render_kpi_card("Matching Records", len(filtered_df), f"Out of {len(df)} total")
    with m2:
        if rating_col and not filtered_df.empty:
            render_kpi_card("Average Rating", f"{filtered_df[rating_col].mean():.2f}★")
    with m3:
        # Download filtered CSV
        csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
        st.write("")
        st.download_button(
            "⬇️ Download Filtered CSV",
            data=csv_bytes,
            file_name="marketlens_feedback_export.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.divider()

    if filtered_df.empty:
        st.warning("No records match the current filter and search criteria.")
        return

    # Pagination controls & Table view
    col_ps, _ = st.columns([2, 4])
    with col_ps:
        page_sz = st.selectbox("Records per page", options=[10, 25, 50], index=0)

    slice_df, page_idx, total_pages = paginate_dataframe(
        filtered_df, page_size=page_sz, key_prefix="explorer_tbl"
    )

    st.dataframe(slice_df, use_container_width=True)

    # 3. Granular Review Inspector
    st.subheader("🔍 Review Detail Inspector")
    id_col = "review_id" if "review_id" in filtered_df.columns else None

    if id_col:
        sample_ids = slice_df[id_col].astype(str).tolist()
        sel_id = st.selectbox("Select Record ID to inspect:", options=sample_ids)

        if sel_id:
            row_match = filtered_df[filtered_df[id_col].astype(str) == sel_id]
            if not row_match.empty:
                record = row_match.iloc[0].to_dict()
                st.markdown(f"#### Review `{sel_id}`")
                st.info(
                    f"**Customer Feedback:**\n\n> {record.get('review_text', 'N/A')}"
                )

                col_det1, col_det2 = st.columns(2)
                with col_det1:
                    st.write(f"**Rating:** {record.get('rating', 'N/A')}")
                    st.write(f"**Created At:** {record.get('created_at', 'N/A')}")
                    st.write(f"**Source:** {record.get('source', 'N/A')}")
                    prod_val = record.get("product_or_service", "N/A")
                    st.write(f"**Product / Service:** {prod_val}")
                with col_det2:
                    pred_sent = record.get("predicted_sentiment", "N/A")
                    st.write(f"**Predicted Sentiment:** {pred_sent}")
                    sent_conf = record.get("sentiment_confidence", "N/A")
                    st.write(f"**Confidence:** {sent_conf}")
                    st.write(f"**Topic Cluster:** {record.get('cluster_id', 'N/A')}")
                    st.write(f"**Is Uncertain:** {record.get('is_uncertain', 'N/A')}")
