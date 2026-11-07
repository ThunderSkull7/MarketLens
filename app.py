"""
MarketLens - Marketing Intelligence Platform.

Main Streamlit Application Entry Point.
Coordinates multipage navigation, centralized session state, and global layout.
"""

from __future__ import annotations

import streamlit as st

from marketlens.dashboard.pages.analytics import render_analytics_page
from marketlens.dashboard.pages.explorer import render_explorer_page
from marketlens.dashboard.pages.intelligence import render_intelligence_page
from marketlens.dashboard.pages.overview import render_overview_page
from marketlens.dashboard.pages.reports import render_reports_page
from marketlens.dashboard.pages.sentiment import render_sentiment_page
from marketlens.dashboard.pages.topics import render_topics_page
from marketlens.dashboard.pages.upload import render_upload_page
from marketlens.dashboard.state import (
    KEY_ACTIVE_NAV,
    KEY_CLEANED_DF,
    KEY_RAW_DF,
    KEY_SOURCE_NAME,
    init_session_state,
    load_demo_dataset,
)

# 1. Page Configuration
st.set_page_config(
    page_title="MarketLens | Marketing Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Centralized Session State Initialization
init_session_state()

# 3. Sidebar Header & Navigation
with st.sidebar:
    st.title("📈 MarketLens")
    st.caption("Enterprise Marketing Intelligence Platform")
    st.divider()

    # Active Dataset Status Badge
    raw_df = st.session_state.get(KEY_RAW_DF)
    cleaned_df = st.session_state.get(KEY_CLEANED_DF)
    source_name = st.session_state.get(KEY_SOURCE_NAME)

    if raw_df is not None and not raw_df.empty:
        st.success(
            f"**Active Dataset:** `{source_name}`\n\n"
            f"- Total: `{len(raw_df)}` rows\n"
            f"- Cleaned: `{len(cleaned_df) if cleaned_df is not None else 0}` rows"
        )
    else:
        st.warning("⚠️ **No Dataset Loaded**")
        if st.button(
            "🚀 Load Demo Dataset", key="sidebar_demo_btn", use_container_width=True
        ):
            if load_demo_dataset():
                st.success("Loaded demo dataset!")
                st.rerun()

    st.divider()

    # Multipage Navigation Menu
    nav_options = [
        "Overview",
        "Data Upload & Quality",
        "Marketing Analytics",
        "Sentiment Analysis",
        "Topic & Theme Discovery",
        "Feedback Explorer",
        "AI Marketing Intelligence",
        "Reports & Export",
    ]

    current_nav = st.session_state.get(KEY_ACTIVE_NAV, "Overview")
    default_idx = nav_options.index(current_nav) if current_nav in nav_options else 0

    selected_nav = st.radio(
        "Navigation",
        options=nav_options,
        index=default_idx,
        key="nav_radio",
    )
    st.session_state[KEY_ACTIVE_NAV] = selected_nav

    st.divider()
    st.caption("MarketLens v0.5.0 • Powered by DeepMind & Scikit-Learn")

# 4. Route to Selected Page
PAGE_ROUTER = {
    "Overview": render_overview_page,
    "Data Upload & Quality": render_upload_page,
    "Marketing Analytics": render_analytics_page,
    "Sentiment Analysis": render_sentiment_page,
    "Topic & Theme Discovery": render_topics_page,
    "Feedback Explorer": render_explorer_page,
    "AI Marketing Intelligence": render_intelligence_page,
    "Reports & Export": render_reports_page,
}

render_func = PAGE_ROUTER.get(selected_nav, render_overview_page)
render_func()
