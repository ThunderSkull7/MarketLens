"""
MarketLens Dashboard - AI Marketing Intelligence Page.

Integrates the provider-neutral LLM Intelligence Engine to synthesize executive
summaries, sentiment intelligence, topic business relevance, and evidence-grounded
actionable recommendations. Operates gracefully in offline mode when credentials
are absent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from marketlens.analytics import (
    dataset_overview,
    product_service_analysis,
    rating_analysis,
    temporal_analysis,
)
from marketlens.dashboard.components import render_empty_state, render_kpi_card
from marketlens.dashboard.state import (
    KEY_ANALYTICS_SUMMARY,
    KEY_INTELLIGENCE_REPORT,
    KEY_NLP_RESULTS,
    get_active_dataframe,
    load_demo_dataset,
)
from marketlens.intelligence import (
    IntelligenceEngine,
    IntelligenceReport,
    create_llm_provider,
    load_intelligence_report,
    save_intelligence_report,
)


def render_intelligence_page() -> None:
    """Render the AI Marketing Intelligence page."""
    st.title("🤖 AI Marketing Intelligence Layer")
    st.markdown(
        "Synthesize structured marketing intelligence, strategic recommendations, "
        "and customer voice signals grounded strictly in verified feedback evidence."
    )

    df = get_active_dataframe()
    if df is None or df.empty:
        render_empty_state(
            title="No Dataset Loaded",
            message=(
                "Please upload a feedback dataset or load the demo "
                "dataset to generate AI intelligence."
            ),
            icon="🤖",
            action_button_label="Load Demo Dataset",
            on_action=lambda: (load_demo_dataset(), st.rerun()),
        )
        return

    # Check LLM Provider status
    provider = create_llm_provider()
    is_offline = provider is None

    if not is_offline:
        prov_name = provider.get_provider_name().title()
        mod_name = provider.get_model_name()
        st.warning(
            f"🔒 **Data Privacy Notice ({prov_name} / {mod_name}):** "
            "When generating an AI intelligence report, selected customer review "
            "excerpts, record identifiers, and aggregate metrics will be transmitted "
            "to the configured external LLM provider. **Automated PII removal is not "
            "implemented.** Do not submit confidential or personally identifiable "
            "customer information unless you have appropriate authorization and have "
            "reviewed the provider's data-handling terms. All core analytics, "
            "cleaning, sentiment classification, and topic discovery remain 100% "
            "functional offline without configuring an LLM provider."
        )
    else:
        st.info(
            "ℹ️ **Offline Mode:** No active LLM provider credentials configured. "
            "To enable live generative intelligence, configure environment variables "
            "or Streamlit secrets (e.g., `OPENAI_API_KEY`). All core ingestion, "
            "cleaning, analytics, sentiment modeling, and topic discovery remain "
            "100% functional offline without an external provider. You may still "
            "inspect previously generated intelligence reports below."
        )

    # 1. Action Bar: Generate & Saved Report Selector
    act_col1, act_col2, act_col3 = st.columns([2, 2, 2])

    with act_col1:
        if st.button(
            "✨ Generate Intelligence Report", type="primary", use_container_width=True
        ):
            with st.spinner(
                "Synthesizing evidence and invoking LLM intelligence engine..."
            ):
                try:
                    # Gather available analytics and NLP results
                    nlp_results = st.session_state.get(KEY_NLP_RESULTS)

                    # Build analytics summary if not cached
                    analytics_sum = st.session_state.get(KEY_ANALYTICS_SUMMARY)
                    if analytics_sum is None:
                        analytics_sum = {
                            "overview": dataset_overview(df),
                            "rating": rating_analysis(df)
                            if "rating" in df.columns
                            else {},
                            "temporal": temporal_analysis(df)
                            if "created_at" in df.columns or "date" in df.columns
                            else {},
                            "product_service": product_service_analysis(df)
                            if "product_or_service" in df.columns
                            else {},
                        }
                        st.session_state[KEY_ANALYTICS_SUMMARY] = analytics_sum

                    engine = IntelligenceEngine(provider=provider)
                    report = engine.generate_report(
                        df=df,
                        analytics_summary=analytics_sum,
                        nlp_results=nlp_results,
                    )
                    st.session_state[KEY_INTELLIGENCE_REPORT] = report
                    if report.status == "completed":
                        st.success("Intelligence Report successfully generated!")
                    else:
                        st.warning(
                            f"Report status: {report.status}. "
                            + (report.execution_metadata.error_message or "")
                        )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to generate intelligence report: {exc}")

    with act_col2:
        reports_dir = Path("reports")
        report_files = (
            sorted(
                reports_dir.glob("*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if reports_dir.exists()
            else []
        )
        if report_files:
            selected_file = st.selectbox(
                "Load Saved Report",
                options=["None"] + [p.name for p in report_files],
                index=0,
                key="load_report_select",
            )
            if selected_file != "None":
                if st.button("Load Report", use_container_width=True):
                    try:
                        loaded_rep = load_intelligence_report(
                            reports_dir / selected_file
                        )
                        st.session_state[KEY_INTELLIGENCE_REPORT] = loaded_rep
                        st.success(f"Loaded '{selected_file}' successfully!")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Failed to load report: {exc}")

    with act_col3:
        report: Optional[IntelligenceReport] = st.session_state.get(
            KEY_INTELLIGENCE_REPORT
        )
        if report and report.status == "completed":
            if st.button("💾 Save Report to Disk", use_container_width=True):
                try:
                    fn = f"intelligence_report_{report.report_id}.json"
                    saved_path = save_intelligence_report(
                        report, output_dir="reports", filename=fn
                    )
                    st.success(f"Saved report to `{saved_path}`")
                except Exception as exc:
                    st.error(f"Failed to save report: {exc}")

    st.divider()

    # 2. Check report availability
    report = st.session_state.get(KEY_INTELLIGENCE_REPORT)
    if report is None:
        st.info(
            "No intelligence report generated yet. "
            "Click **Generate Intelligence Report** or load a report above."
        )
        return

    if report.status != "completed":
        err_msg = report.execution_metadata.error_message or "No details."
        st.warning(
            f"Intelligence report status: **{report.status}**\n\nDetails: {err_msg}"
        )
        return

    # 3. Telemetry & Execution Metadata
    meta = report.execution_metadata
    st.subheader("⚡ Report Telemetry & Audit Metadata")
    tc1, tc2, tc3, tc4 = st.columns(4)
    with tc1:
        render_kpi_card(
            "Model / Provider", f"{meta.provider or 'N/A'}", meta.model or ""
        )
    with tc2:
        render_kpi_card(
            "Total Tokens",
            meta.total_tokens or "N/A",
            f"In: {meta.input_tokens} | Out: {meta.output_tokens}",
        )
    with tc3:
        cost_str = (
            f"${meta.estimated_cost_usd:.4f}"
            if meta.estimated_cost_usd is not None
            else "N/A"
        )
        render_kpi_card("Estimated Cost", cost_str, "Token usage estimate")
    with tc4:
        latency_str = (
            f"{meta.latency_ms:.0f} ms" if meta.latency_ms is not None else "N/A"
        )
        render_kpi_card("Inference Latency", latency_str, f"Status: {meta.status}")

    st.divider()

    # 4. Executive Summary
    if report.executive_summary:
        es = report.executive_summary
        st.subheader("📋 Executive Summary")
        st.markdown(f"**Strategic Overview:**\n\n{es.overview}")

        col_sig_pos, col_sig_neg = st.columns(2)
        with col_sig_pos:
            st.markdown("##### 🟢 Major Positive Signals")
            if es.positive_signals:
                for sig in es.positive_signals:
                    st.markdown(f"- {sig}")
            else:
                st.caption("No strong positive signals identified.")

        with col_sig_neg:
            st.markdown("##### 🔴 Recurring Customer Issues")
            if es.recurring_issues:
                for iss in es.recurring_issues:
                    st.markdown(f"- {iss}")
            else:
                st.caption("No recurring issues flagged.")

        if es.business_interpretation:
            st.markdown(f"**Business Interpretation:** {es.business_interpretation}")

        st.divider()

    # 5. Sentiment Intelligence
    if report.sentiment_intelligence:
        si = report.sentiment_intelligence
        st.subheader("🎭 Sentiment Intelligence & Thematic Polarity")

        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown(f"**Dominant Sentiment:** `{si.dominant_sentiment.upper()}`")
            st.markdown(f"**Summary:** {si.distribution_summary}")

        with sc2:
            st.markdown(f"**Model Limitations:** {si.model_limitations}")

        col_th_pos, col_th_neg = st.columns(2)
        with col_th_pos:
            st.markdown("##### Positive Themes")
            for t in si.positive_themes:
                st.markdown(f"- {t}")

        with col_th_neg:
            st.markdown("##### Negative Themes")
            for t in si.negative_themes:
                st.markdown(f"- {t}")

        st.divider()

    # 6. Topic & Theme Intelligence
    if report.topic_insights:
        st.subheader("🔍 Topic & Theme Business Relevance")
        for ti in report.topic_insights:
            with st.expander(f"Theme {ti.cluster_id}: {ti.topic_label}", expanded=True):
                st.markdown(f"**Qualitative Interpretation:** {ti.interpretation}")
                st.markdown(f"**Customer Concerns:** {ti.customer_concerns}")
                st.markdown(f"**Business Relevance:** {ti.business_relevance}")
                if ti.is_emerging:
                    st.warning(f"⚠️ Emerging Trend Flagged: {ti.temporal_evidence}")

        st.divider()

    # 7. Segment & Product Comparisons
    if report.comparison_insights:
        st.subheader("🏢 Comparative Segment Analysis")
        for ci in report.comparison_insights:
            with st.expander(
                f"Comparison: {ci.comparison_name} ({ci.entity_type})", expanded=True
            ):
                st.markdown(f"**Key Findings:** {ci.findings}")
                sample_ok = ci.sample_size_adequate
                st.caption(f"🛡️ Sample adequate: {sample_ok} | {ci.limitations}")

        st.divider()

    # 8. Actionable Marketing Recommendations
    if report.recommendations:
        st.subheader("🎯 Actionable Marketing Recommendations")
        for rec in report.recommendations:
            priority_color = {
                "high": "🔴",
                "medium": "🟡",
                "low": "🟢",
            }.get(rec.priority.lower(), "⚪")

            with st.expander(
                f"{priority_color} [{rec.priority.upper()}] {rec.title} "
                f"({rec.impact_category.title()})",
                expanded=True,
            ):
                st.markdown(f"**Description:** {rec.description}")
                st.markdown(f"**Suggested Action:** {rec.suggested_action}")
                st.markdown(f"**Supporting Observation:** {rec.supporting_observation}")
                ev_str = ", ".join(rec.evidence_ids) if rec.evidence_ids else "None"
                st.caption(f"Confidence: {rec.confidence:.2f} | Evidence: {ev_str}")

        st.divider()

    # 9. Evidence Reference Inspector
    if report.executive_summary and report.executive_summary.evidence_references:
        st.subheader("📜 Supporting Feedback Evidence Excerpts")
        ev_rows = [
            {
                "Feedback ID": ev.review_id,
                "Excerpt": ev.quote,
                "Rating": ev.rating or "N/A",
                "Context": ev.context or "N/A",
            }
            for ev in report.executive_summary.evidence_references
        ]
        st.dataframe(pd.DataFrame(ev_rows), use_container_width=True)

    # Disclaimer footer
    if report.limitations_disclaimer:
        st.caption(f"📝 **Disclaimer:** {report.limitations_disclaimer}")
