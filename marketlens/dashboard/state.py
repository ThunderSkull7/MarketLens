"""
MarketLens Dashboard - Centralized Session State & Caching Management.

Handles initialization, persistence across multipage navigation,
deterministic demo loading, and cache invalidation when datasets change.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st

from marketlens.cleaning import CleaningPipeline
from marketlens.ingestion import IngestionResult, ingest

logger = logging.getLogger("marketlens.dashboard.state")

DEMO_CSV_PATH = Path("data/raw/synthetic_feedback_demo.csv")

# Core Session State Keys
KEY_RAW_DF = "raw_df"
KEY_SOURCE_NAME = "source_filename"
KEY_INGESTION_RESULT = "ingestion_result"
KEY_CLEANED_DF = "cleaned_df"
KEY_CLEANING_REPORT = "cleaning_report"
KEY_CLEANING_OPTIONS = "cleaning_options"
KEY_ANALYTICS_SUMMARY = "analytics_summary"
KEY_NLP_MANAGER = "nlp_manager"
KEY_NLP_RESULTS = "nlp_results"
KEY_INTELLIGENCE_REPORT = "intelligence_report"
KEY_ACTIVE_NAV = "active_nav"


def init_session_state() -> None:
    """Initialize all session state keys with default values if not already set."""
    defaults: dict[str, Any] = {
        KEY_RAW_DF: None,
        KEY_SOURCE_NAME: None,
        KEY_INGESTION_RESULT: None,
        KEY_CLEANED_DF: None,
        KEY_CLEANING_REPORT: None,
        KEY_CLEANING_OPTIONS: {"include_flagged": False, "deduplicate": True},
        KEY_ANALYTICS_SUMMARY: None,
        KEY_NLP_MANAGER: None,
        KEY_NLP_RESULTS: None,
        KEY_INTELLIGENCE_REPORT: None,
        KEY_ACTIVE_NAV: "Overview",
    }
    for key, default_val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_val


def reset_downstream_state() -> None:
    """Invalidate all downstream analytical and NLP caches when dataset changes."""
    st.session_state[KEY_CLEANED_DF] = None
    st.session_state[KEY_CLEANING_REPORT] = None
    st.session_state[KEY_ANALYTICS_SUMMARY] = None
    st.session_state[KEY_NLP_MANAGER] = None
    st.session_state[KEY_NLP_RESULTS] = None
    st.session_state[KEY_INTELLIGENCE_REPORT] = None
    logger.info(
        "Downstream analytical and NLP caches invalidated due to dataset change."
    )


def set_raw_dataset(
    df: pd.DataFrame,
    source_name: str,
    ingestion_result: Optional[IngestionResult] = None,
) -> None:
    """Store newly uploaded raw dataset and invalidate downstream state."""
    # Only reset if dataset actually changed
    current_source = st.session_state.get(KEY_SOURCE_NAME)
    if current_source != source_name or st.session_state.get(KEY_RAW_DF) is None:
        reset_downstream_state()
        st.session_state[KEY_RAW_DF] = df
        st.session_state[KEY_SOURCE_NAME] = source_name
        st.session_state[KEY_INGESTION_RESULT] = ingestion_result
        logger.info("Loaded new raw dataset '%s' (%d rows).", source_name, len(df))


def execute_cleaning(
    include_flagged: bool = False,
    deduplicate: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Run CleaningPipeline on accepted records and store in session state.

    Reuses existing cleaned data if options and input dataset have not changed.
    """
    ingestion_res: Optional[IngestionResult] = st.session_state.get(
        KEY_INGESTION_RESULT
    )
    raw_df: Optional[pd.DataFrame] = st.session_state.get(KEY_RAW_DF)

    if ingestion_res is not None:
        pipeline = CleaningPipeline(ingestion_res)
        cleaned_df, report = pipeline.run(
            include_flagged=include_flagged,
            should_detect_duplicates=deduplicate,
        )
    elif raw_df is not None and not raw_df.empty:
        cleaned_df = raw_df.copy()
        report = {"summary": "Direct DataFrame loaded without IngestionResult"}
    else:
        empty_df = pd.DataFrame()
        empty_rep: dict[str, Any] = {"summary": "No data available to clean"}
        st.session_state[KEY_CLEANED_DF] = empty_df
        st.session_state[KEY_CLEANING_REPORT] = empty_rep
        return empty_df, empty_rep

    st.session_state[KEY_CLEANED_DF] = cleaned_df
    st.session_state[KEY_CLEANING_REPORT] = report
    st.session_state[KEY_CLEANING_OPTIONS] = {
        "include_flagged": include_flagged,
        "deduplicate": deduplicate,
    }
    # Invalidate subsequent NLP/analytics since cleaning output changed
    st.session_state[KEY_ANALYTICS_SUMMARY] = None
    st.session_state[KEY_NLP_RESULTS] = None
    st.session_state[KEY_INTELLIGENCE_REPORT] = None

    return cleaned_df, report


def load_demo_dataset() -> bool:
    """Load the built-in synthetic feedback demo dataset with automatic cleaning."""
    if not DEMO_CSV_PATH.exists():
        logger.error("Demo dataset not found at %s", DEMO_CSV_PATH)
        return False

    try:
        res = ingest(DEMO_CSV_PATH)
        raw_df = res.df_validated
        set_raw_dataset(
            raw_df, source_name="synthetic_feedback_demo.csv", ingestion_result=res
        )
        execute_cleaning(include_flagged=False, deduplicate=True)
        return True
    except Exception as exc:
        logger.error("Failed to load demo dataset: %s", exc)
        return False


def get_active_dataframe() -> Optional[pd.DataFrame]:
    """Return the cleaned DataFrame if available, otherwise raw DataFrame, or None."""
    cleaned = st.session_state.get(KEY_CLEANED_DF)
    if cleaned is not None and not cleaned.empty:
        return cleaned
    raw = st.session_state.get(KEY_RAW_DF)
    if raw is not None and not raw.empty:
        return raw
    return None
