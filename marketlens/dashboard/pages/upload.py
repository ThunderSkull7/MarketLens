"""
MarketLens Dashboard - Data Upload & Quality Page.

Provides CSV upload, file-size validation, schema mapping, data health checks,
problematic records inspection, and cleaning pipeline execution.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from marketlens.analytics import dataset_overview
from marketlens.config import settings
from marketlens.dashboard.components import paginate_dataframe, render_kpi_card
from marketlens.dashboard.state import (
    KEY_CLEANED_DF,
    KEY_CLEANING_OPTIONS,
    KEY_CLEANING_REPORT,
    KEY_INGESTION_RESULT,
    KEY_RAW_DF,
    KEY_SOURCE_NAME,
    execute_cleaning,
    load_demo_dataset,
    set_raw_dataset,
)
from marketlens.ingestion import IngestionResult, ingest


def render_upload_page() -> None:
    """Render the Data Upload & Quality page."""
    st.title("📁 Data Upload & Quality")
    st.markdown(
        "Upload customer feedback data in CSV format, inspect data quality "
        "and validation rules, and configure the cleaning pipeline."
    )

    # 1. Action Bar: File Uploader & Demo Loader
    col_upload, col_demo = st.columns([3, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help=f"Maximum allowed file size: {settings.max_upload_size_mb} MB",
        )

    with col_demo:
        st.write("")
        st.write("")
        if st.button("🚀 Load Demo Dataset", use_container_width=True):
            success = load_demo_dataset()
            if success:
                st.success("Successfully loaded demo dataset!")
                st.rerun()
            else:
                st.error("Demo dataset could not be found.")

    # Process uploaded file if provided
    if uploaded_file is not None:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > settings.max_upload_size_mb:
            st.error(
                f"File size ({file_size_mb:.2f} MB) exceeds maximum allowed "
                f"limit of {settings.max_upload_size_mb} MB."
            )
            return

        current_name = st.session_state.get(KEY_SOURCE_NAME)
        if current_name != uploaded_file.name:
            # Write to a temporary file so ingest() can inspect and parse it
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = Path(tmp.name)

            try:
                with st.spinner(f"Ingesting and validating '{uploaded_file.name}'..."):
                    result: IngestionResult = ingest(tmp_path)
                    raw_df = result.df_validated
                    set_raw_dataset(
                        df=raw_df,
                        source_name=uploaded_file.name,
                        ingestion_result=result,
                    )
                    # Automatically run default cleaning
                    execute_cleaning(include_flagged=False, deduplicate=True)
                    st.success(f"Successfully ingested '{uploaded_file.name}'!")
                    st.rerun()
            except Exception as exc:
                st.error(f"Ingestion failed: {exc}")
                return
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()

    # 2. Check if dataset is loaded
    raw_df: Optional[pd.DataFrame] = st.session_state.get(KEY_RAW_DF)
    ingestion_res: Optional[IngestionResult] = st.session_state.get(
        KEY_INGESTION_RESULT
    )
    source_name = st.session_state.get(KEY_SOURCE_NAME, "Unknown Source")

    if raw_df is None or raw_df.empty:
        st.info(
            "No dataset currently loaded. "
            "Upload a CSV file or load the demo dataset above."
        )
        return

    st.divider()

    # 3. Quality & Accounting Metrics
    st.subheader(f"📊 Dataset Accounting — {source_name}")
    col_a, col_b, col_c, col_d = st.columns(4)

    total_rows = len(raw_df)
    accepted_count = (
        len(ingestion_res.accepted) if ingestion_res is not None else total_rows
    )
    flagged_count = len(ingestion_res.flagged) if ingestion_res is not None else 0
    rejected_count = len(ingestion_res.rejected) if ingestion_res is not None else 0

    with col_a:
        render_kpi_card("Total Parsed Records", total_rows, "Rows identified in CSV")
    with col_b:
        render_kpi_card(
            "Accepted Records", accepted_count, "Passed schema & type validation"
        )
    with col_c:
        render_kpi_card(
            "Flagged Records",
            flagged_count,
            "Passed with warnings",
            delta=f"-{flagged_count}" if flagged_count > 0 else "0",
        )
    with col_d:
        render_kpi_card(
            "Rejected Records",
            rejected_count,
            "Violated required constraints",
            delta=f"-{rejected_count}" if rejected_count > 0 else "0",
        )

    # 4. Ingestion Warnings & Column Mappings
    if ingestion_res is not None:
        if ingestion_res.warnings:
            with st.expander(
                f"⚠️ Validation Warnings ({len(ingestion_res.warnings)})", expanded=False
            ):
                for w in ingestion_res.warnings:
                    st.warning(w)

        mapped_cols = (
            ingestion_res.mapping_info.get("mapped_columns", {})
            if ingestion_res.mapping_info
            else {}
        )
        if mapped_cols:
            with st.expander("🗺️ Canonical Column Mappings", expanded=False):
                mapping_df = pd.DataFrame(
                    [
                        {"Canonical Field": k, "Source Column": v}
                        for k, v in mapped_cols.items()
                    ]
                )
                st.dataframe(mapping_df, use_container_width=True)

    # 5. Missing Values & Structural Overview
    overview_stats = dataset_overview(raw_df)
    missing_dist = overview_stats.get("missing_value_distribution", {})
    if missing_dist:
        with st.expander("🔍 Missing Values Distribution", expanded=False):
            rows = []
            for col, cnt in missing_dist.items():
                c = cnt if isinstance(cnt, int) else cnt.get("count", 0)
                pct = (c / total_rows * 100) if total_rows > 0 else 0.0
                rows.append(
                    {"Column": col, "Missing Count": c, "Missing %": f"{pct:.1f}%"}
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # 6. Problematic Records Inspector
    if ingestion_res is not None and (ingestion_res.rejected or ingestion_res.flagged):
        with st.expander("🛠️ Problematic Records Inspector", expanded=False):
            tab_rej, tab_flg = st.tabs(
                [
                    f"Rejected Records ({len(ingestion_res.rejected)})",
                    f"Flagged Records ({len(ingestion_res.flagged)})",
                ]
            )
            with tab_rej:
                if ingestion_res.rejected:
                    rej_rows = [
                        {
                            "Row": r.row_index,
                            "Reasons": "; ".join(r.reasons),
                            **r.row_data,
                        }
                        for r in ingestion_res.rejected[:100]
                    ]
                    st.dataframe(pd.DataFrame(rej_rows), use_container_width=True)
                else:
                    st.success("No records were rejected.")

            with tab_flg:
                if ingestion_res.flagged:
                    flg_rows = [
                        {
                            "Row": r.row_index,
                            "Reasons": "; ".join(r.reasons),
                            **r.row_data,
                        }
                        for r in ingestion_res.flagged[:100]
                    ]
                    st.dataframe(pd.DataFrame(flg_rows), use_container_width=True)
                else:
                    st.success("No records were flagged.")

    st.divider()

    # 7. Cleaning Pipeline Configuration
    st.subheader("🧹 Cleaning Pipeline Configuration")
    opt = st.session_state.get(
        KEY_CLEANING_OPTIONS, {"include_flagged": False, "deduplicate": True}
    )

    col_cfg1, col_cfg2, col_cfg3 = st.columns([2, 2, 2])
    with col_cfg1:
        inc_flagged = st.checkbox(
            "Include flagged records",
            value=opt.get("include_flagged", False),
            help="If checked, records with non-fatal warnings are included.",
        )
    with col_cfg2:
        dedup = st.checkbox(
            "Deduplicate identical reviews",
            value=opt.get("deduplicate", True),
            help="Removes duplicate reviews based on ID and normalized text.",
        )
    with col_cfg3:
        st.write("")
        if st.button(
            "Apply Cleaning Pipeline", type="primary", use_container_width=True
        ):
            with st.spinner("Executing cleaning pipeline..."):
                cleaned_df, report = execute_cleaning(
                    include_flagged=inc_flagged, deduplicate=dedup
                )
                n_clean = len(cleaned_df)
                st.success(f"Cleaning completed! {n_clean} records produced.")
                st.rerun()

    # 8. Cleaned Dataset Preview
    cleaned_df: Optional[pd.DataFrame] = st.session_state.get(KEY_CLEANED_DF)
    cleaning_rep = st.session_state.get(KEY_CLEANING_REPORT)

    if cleaned_df is not None and not cleaned_df.empty:
        st.markdown(f"**Cleaned Records Ready for Analysis:** `{len(cleaned_df)}` rows")
        if cleaning_rep and "deduplication" in cleaning_rep:
            dedup_info = cleaning_rep["deduplication"]
            st.caption(
                f"Deduplication removed {dedup_info.get('duplicates_removed', 0)} "
                f"duplicate records."
            )

        preview_slice, _, _ = paginate_dataframe(
            cleaned_df, page_size=10, key_prefix="clean_prev"
        )
        st.dataframe(preview_slice, use_container_width=True)
