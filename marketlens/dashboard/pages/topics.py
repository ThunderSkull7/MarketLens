"""
MarketLens Dashboard - Topic & Theme Discovery Page.

Discovers unsupervised customer feedback clusters using K-Means and TF-IDF.
Provides cluster inspection, silhouette evaluation, and representative reviews.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from marketlens.dashboard.components import (
    paginate_dataframe,
    render_bar_chart,
    render_empty_state,
    render_kpi_card,
)
from marketlens.dashboard.state import (
    KEY_NLP_MANAGER,
    KEY_NLP_RESULTS,
    get_active_dataframe,
    load_demo_dataset,
)
from marketlens.nlp.topics import TopicDiscoveryModel
from marketlens.nlp.unified import NLPManager


def render_topics_page() -> None:
    """Render the Topic & Theme Discovery page."""
    st.title("🔍 Topic & Theme Discovery")
    st.markdown(
        "Discover recurring customer themes and operational issues through "
        "unsupervised K-Means clustering of customer feedback text."
    )

    df = get_active_dataframe()
    if df is None or df.empty:
        render_empty_state(
            title="No Dataset Loaded",
            message=(
                "Please upload a feedback dataset or load the demo "
                "dataset to discover topics."
            ),
            icon="🔍",
            action_button_label="Load Demo Dataset",
            on_action=lambda: (load_demo_dataset(), st.rerun()),
        )
        return

    if "review_text" not in df.columns:
        st.error("Dataset must contain a 'review_text' column for topic discovery.")
        return

    manager: NLPManager = st.session_state.get(KEY_NLP_MANAGER) or NLPManager()
    st.session_state[KEY_NLP_MANAGER] = manager

    total_records = len(df)
    max_k = min(10, max(2, total_records - 1))

    # 1. Topic Discovery Configuration Bar
    col_k, col_btn = st.columns([3, 1])

    with col_k:
        k_clusters = st.slider(
            "Number of Topic Clusters (k)",
            min_value=2,
            max_value=max_k,
            value=min(3, max_k),
            step=1,
            help="Higher values create more granular thematic sub-clusters.",
        )

    with col_btn:
        st.write("")
        st.write("")
        run_discovery = st.button(
            "🚀 Discover Topics", type="primary", use_container_width=True
        )

    if run_discovery:
        with st.spinner(
            f"Clustering {total_records} reviews into {k_clusters} themes..."
        ):
            try:
                texts = df["review_text"].fillna("").tolist()
                ratings = df["rating"].tolist() if "rating" in df.columns else None
                sentiments = (
                    df["sentiment"].tolist()
                    if "sentiment" in df.columns
                    else (
                        df["predicted_sentiment"].tolist()
                        if "predicted_sentiment" in df.columns
                        else None
                    )
                )

                topic_model = TopicDiscoveryModel(
                    n_clusters=k_clusters, random_state=42
                )
                topic_model.fit(texts, ratings=ratings, sentiments=sentiments)
                manager.topic_model = topic_model

                # Update session NLP results
                nlp_results = st.session_state.get(KEY_NLP_RESULTS) or {}
                nlp_results["topic_clusters"] = topic_model.get_cluster_summaries()
                nlp_results["topic_metadata"] = topic_model.training_metadata
                st.session_state[KEY_NLP_RESULTS] = nlp_results
                st.session_state[KEY_NLP_MANAGER] = manager

                st.success(f"Successfully generated {k_clusters} thematic clusters!")
                st.rerun()
            except Exception as exc:
                st.error(f"Topic discovery failed: {exc}")

    st.divider()

    # 2. Check Topic Model Readiness
    topic_model = manager.topic_model
    if topic_model is None or not topic_model.is_fitted:
        st.info(
            "No topic discovery model has been run on this dataset. "
            "Select cluster count and click **Discover Topics** above."
        )
        return

    # 3. Model Diagnostic Metrics
    sil_score = topic_model.silhouette_
    clusters = topic_model.get_cluster_summaries()

    st.subheader("📊 Clustering Quality & Architecture")
    kc1, kc2, kc3 = st.columns(3)

    with kc1:
        render_kpi_card("Clusters Discovered", len(clusters), "K-Means clusters")
    with kc2:
        sil_label = f"{sil_score:.3f}" if sil_score is not None else "N/A"
        render_kpi_card(
            "Silhouette Score",
            sil_label,
            "Cluster cohesion and separation (-1 to +1)",
            help_text=(
                "> 0.5: Strong structure | 0.25 - 0.5: Moderate | "
                "< 0.25: Weak/overlapping clusters in noisy customer feedback"
            ),
        )
    with kc3:
        render_kpi_card("Total Text Corpus", total_records, "Reviews partitioned")

    # Semantic disclaimer alert
    st.info(
        "💡 **Semantic Interpretation Guidance:** Unsupervised clusters "
        "represent statistical n-gram co-occurrences, not human ground truth. "
        "Scores below 0.25 reflect natural variance in short text."
    )

    st.divider()

    # 4. Cluster Volume Distribution Chart
    cluster_vols = [
        {"Cluster": f"Theme {c['cluster_id']}", "Count": c["record_count"]}
        for c in clusters
    ]
    render_bar_chart(
        pd.DataFrame(cluster_vols),
        x_col="Cluster",
        y_col="Count",
        title="Cluster Record Distribution",
    )

    st.divider()

    # 5. Granular Thematic Deep-Dive Cards
    st.subheader("📑 Thematic Breakdown & Representative Customer Quotes")

    for cl in clusters:
        cid = cl.get("cluster_id")
        lbl = cl.get("label", f"Cluster {cid}")
        cnt = cl.get("record_count", 0)
        pct = cl.get("record_percentage", 0.0)
        top_terms = cl.get("top_terms", [])
        avg_rating = cl.get("average_rating")
        reps = cl.get("representative_records", [])

        with st.expander(
            f"**Theme {cid}: {lbl}** — {cnt} records ({pct:.1f}%)"
            + (f" | Avg Rating: {avg_rating:.2f}★" if avg_rating is not None else ""),
            expanded=True,
        ):
            c_col1, c_col2 = st.columns([1, 2])

            with c_col1:
                st.markdown("**Top Distinguishing Keywords:**")
                st.write(", ".join([f"`{t}`" for t in top_terms[:8]]))

                if avg_rating is not None:
                    st.metric("Mean Rating in Theme", f"{avg_rating:.2f}★")

            with c_col2:
                st.markdown("**Representative Feedback Excerpts:**")
                if reps:
                    for quote in reps[:3]:
                        st.markdown(f'> *"{quote}"*')
                else:
                    st.caption("No representative quotes recorded.")

    st.divider()

    # 6. Assigned Clusters Review Browser
    st.subheader("🗂️ Explore Assigned Reviews")
    texts = df["review_text"].fillna("").tolist()
    assigned_cids = topic_model.assign_clusters(texts)
    annotated_df = df.copy()
    annotated_df["cluster_id"] = assigned_cids
    annotated_df["cluster_label"] = [f"Theme {cid}" for cid in assigned_cids]

    selected_cluster_filter = st.selectbox(
        "Filter by cluster:",
        options=["All Themes"] + [f"Theme {c['cluster_id']}" for c in clusters],
    )
    if selected_cluster_filter != "All Themes":
        cid_filter = int(selected_cluster_filter.replace("Theme ", ""))
        annotated_df = annotated_df[annotated_df["cluster_id"] == cid_filter]

    display_cols = [
        c
        for c in ["cluster_id", "review_id", "review_text", "rating", "created_at"]
        if c in annotated_df.columns
    ]
    p_df, _, _ = paginate_dataframe(
        annotated_df[display_cols], page_size=10, key_prefix="topic_revs"
    )
    st.dataframe(p_df, use_container_width=True)
