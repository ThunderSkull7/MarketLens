"""
MarketLens Dashboard - Reusable SaaS UI Components.

Provides standard KPI cards, Altair chart renderers, empty-state containers,
pagination controls, and status banners.
"""

from __future__ import annotations

from typing import Any, Optional

import altair as alt
import pandas as pd
import streamlit as st


def render_kpi_card(
    title: str,
    value: Any,
    subtitle: Optional[str] = None,
    delta: Optional[str] = None,
    help_text: Optional[str] = None,
) -> None:
    """Render a clean SaaS KPI metric card using Streamlit's native metric component."""
    st.metric(
        label=title,
        value=str(value),
        delta=delta,
        help=help_text,
    )
    if subtitle:
        st.caption(subtitle)


def render_empty_state(
    title: str,
    message: str,
    icon: str = "📂",
    action_button_label: Optional[str] = None,
    on_action: Optional[Any] = None,
) -> None:
    """Render a prominent, informative empty-state banner when no data is available."""
    st.info(f"### {icon} {title}\n\n{message}")
    if action_button_label and on_action:
        if st.button(action_button_label, type="primary"):
            on_action()


def paginate_dataframe(
    df: pd.DataFrame,
    page_size: int = 15,
    key_prefix: str = "df_page",
) -> tuple[pd.DataFrame, int, int]:
    """
    Paginate a DataFrame to prevent sluggish browser rendering on large datasets.

    Returns (paginated_slice_df, current_page_index, total_pages).
    """
    if df.empty:
        return df, 1, 1

    total_rows = len(df)
    total_pages = max(1, (total_rows + page_size - 1) // page_size)

    page_col1, page_col2 = st.columns([2, 4])
    with page_col1:
        page_num = st.number_input(
            "Page",
            min_value=1,
            max_value=total_pages,
            value=1,
            step=1,
            key=f"{key_prefix}_num",
        )
    with page_col2:
        st.caption(
            f"Showing page {page_num} of {total_pages} ({total_rows} total records)"
        )

    start_idx = (page_num - 1) * page_size
    end_idx = min(start_idx + page_size, total_rows)
    sliced_df = df.iloc[start_idx:end_idx]

    return sliced_df, page_num, total_pages


def render_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    color_col: Optional[str] = None,
    horizontal: bool = False,
) -> None:
    """Render a styled bar chart with Altair."""
    if df.empty:
        st.caption("No data available for chart.")
        return

    plot_df = df.copy()
    if horizontal:
        chart = (
            alt.Chart(plot_df)
            .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
            .encode(
                x=alt.X(y_col, title=None),
                y=alt.Y(x_col, sort="-x", title=None),
                color=alt.Color(color_col) if color_col else alt.value("#2563EB"),
                tooltip=[x_col, y_col],
            )
            .properties(title=title, height=max(200, len(plot_df) * 28))
        )
    else:
        chart = (
            alt.Chart(plot_df)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X(x_col, title=None),
                y=alt.Y(y_col, title=None),
                color=alt.Color(color_col) if color_col else alt.value("#2563EB"),
                tooltip=[x_col, y_col],
            )
            .properties(title=title, height=260)
        )

    st.altair_chart(chart, use_container_width=True)
