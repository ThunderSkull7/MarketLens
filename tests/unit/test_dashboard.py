"""
MarketLens - Dashboard Unit and Integration Tests.

Validates session state lifecycle, UI components, multipage navigation via
streamlit.testing.v1.AppTest, binary sentiment integrity, offline LLM handling,
and artifact export generation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from marketlens.dashboard.components import paginate_dataframe
from marketlens.dashboard.pages.reports import _format_markdown_report
from marketlens.dashboard.state import (
    KEY_ACTIVE_NAV,
    KEY_ANALYTICS_SUMMARY,
    KEY_CLEANED_DF,
    KEY_NLP_RESULTS,
    KEY_RAW_DF,
    KEY_SOURCE_NAME,
    get_active_dataframe,
    init_session_state,
    reset_downstream_state,
    set_raw_dataset,
)
from marketlens.intelligence.provider import MockLLMProvider
from marketlens.intelligence.schemas import (
    ExecutiveSummary,
    IntelligenceReport,
    LLMExecutionMetadata,
    MarketingRecommendation,
    SentimentInsight,
)
from marketlens.nlp.sentiment import TfidfLogisticRegressionSentiment

# ---------------------------------------------------------------------------
# 1. State Management Tests
# ---------------------------------------------------------------------------


def test_state_lifecycle_invalidation(monkeypatch):
    """Test session state initialization, setting raw data, and cache reset."""
    fake_session: dict = {}
    import streamlit as st

    monkeypatch.setattr(st, "session_state", fake_session)

    init_session_state()
    assert fake_session[KEY_RAW_DF] is None
    assert fake_session[KEY_ACTIVE_NAV] == "Overview"

    # Set downstream state
    fake_session[KEY_CLEANED_DF] = pd.DataFrame({"a": [1, 2]})
    fake_session[KEY_ANALYTICS_SUMMARY] = {"some": "data"}
    fake_session[KEY_NLP_RESULTS] = {"topic": "data"}

    # Reset downstream state
    reset_downstream_state()
    assert fake_session[KEY_CLEANED_DF] is None
    assert fake_session[KEY_ANALYTICS_SUMMARY] is None
    assert fake_session[KEY_NLP_RESULTS] is None


def test_set_raw_dataset_triggers_reset(monkeypatch):
    """Verify that changing dataset resets downstream caches."""
    fake_session: dict = {}
    import streamlit as st

    monkeypatch.setattr(st, "session_state", fake_session)
    init_session_state()

    df1 = pd.DataFrame({"review_id": ["1"], "review_text": ["test"], "rating": [5.0]})
    set_raw_dataset(df1, "test1.csv")
    assert fake_session[KEY_SOURCE_NAME] == "test1.csv"

    # Simulate downstream calculations
    fake_session[KEY_CLEANED_DF] = df1

    # Setting a new dataset must clear cleaned_df
    df2 = pd.DataFrame({"review_id": ["2"], "review_text": ["test2"], "rating": [4.0]})
    set_raw_dataset(df2, "test2.csv")
    assert fake_session[KEY_SOURCE_NAME] == "test2.csv"
    assert fake_session[KEY_CLEANED_DF] is None


def test_get_active_dataframe_priority(monkeypatch):
    """Verify get_active_dataframe prioritizes cleaned_df over raw_df."""
    fake_session: dict = {}
    import streamlit as st

    monkeypatch.setattr(st, "session_state", fake_session)
    init_session_state()

    assert get_active_dataframe() is None

    raw = pd.DataFrame({"id": [1, 2]})
    fake_session[KEY_RAW_DF] = raw
    assert len(get_active_dataframe()) == 2

    clean = pd.DataFrame({"id": [1]})
    fake_session[KEY_CLEANED_DF] = clean
    assert len(get_active_dataframe()) == 1


# ---------------------------------------------------------------------------
# 2. UI Component Tests
# ---------------------------------------------------------------------------


def test_paginate_dataframe_empty(monkeypatch):
    """Verify pagination gracefully handles an empty DataFrame."""
    empty_df = pd.DataFrame()
    slice_df, page_idx, total_pages = paginate_dataframe(empty_df, page_size=10)
    assert slice_df.empty
    assert page_idx == 1
    assert total_pages == 1


def test_paginate_dataframe_slicing(monkeypatch):
    """Verify pagination math and slicing."""
    df = pd.DataFrame({"id": list(range(25))})

    # Mock st.number_input to return page 2
    import streamlit as st

    monkeypatch.setattr(
        st,
        "columns",
        lambda n: (
            [st._DeltaGenerator() for _ in range(2)]
            if hasattr(st, "_DeltaGenerator")
            else [None, None]
        ),
    )
    monkeypatch.setattr(st, "number_input", lambda *args, **kwargs: 2)
    monkeypatch.setattr(st, "caption", lambda *args, **kwargs: None)

    slice_df, page_idx, total_pages = paginate_dataframe(df, page_size=10)
    assert total_pages == 3
    assert page_idx == 2
    assert len(slice_df) == 10
    assert list(slice_df["id"]) == list(range(10, 20))


# ---------------------------------------------------------------------------
# 3. Streamlit AppTest Full Integration Tests
# ---------------------------------------------------------------------------

APP_PATH = Path(__file__).resolve().parents[2] / "app.py"


def test_apptest_initial_load():
    """Verify app.py loads cleanly without exceptions on first render."""
    at = AppTest.from_file(APP_PATH, default_timeout=15).run()
    assert not at.exception
    assert "Overview" in at.session_state[KEY_ACTIVE_NAV]
    # Check that Overview title is present
    titles = [t.value for t in at.title]
    assert any("Overview" in t for t in titles)


def test_apptest_demo_dataset_loading_and_overview_kpis():
    """Verify demo dataset button loads 20 records and populates overview KPIs."""
    at = AppTest.from_file(APP_PATH, default_timeout=15).run()
    assert not at.exception

    # Click the "Load Demo Dataset" button in sidebar
    demo_btn = at.sidebar.button[0]
    at = demo_btn.click().run()
    assert not at.exception

    assert at.session_state[KEY_SOURCE_NAME] == "synthetic_feedback_demo.csv"
    assert at.session_state[KEY_RAW_DF] is not None
    assert len(at.session_state[KEY_RAW_DF]) == 20
    assert len(at.session_state[KEY_CLEANED_DF]) == 20

    # Verify metrics rendered on Overview page
    metric_labels = [m.label for m in at.metric]
    assert "Total Source Records" in metric_labels
    assert "Accepted Clean" in metric_labels
    assert "Average Rating" in metric_labels


def test_apptest_multipage_navigation():
    """Verify seamless navigation across all 8 dashboard pages without crashing."""
    at = AppTest.from_file(APP_PATH, default_timeout=15).run()
    at = at.sidebar.button[0].click().run()  # load demo dataset
    assert not at.exception

    pages = [
        "Data Upload & Quality",
        "Marketing Analytics",
        "Sentiment Analysis",
        "Topic & Theme Discovery",
        "Feedback Explorer",
        "AI Marketing Intelligence",
        "Reports & Export",
    ]

    for page_name in pages:
        at = at.sidebar.radio[0].set_value(page_name).run()
        assert not at.exception, f"Failed on navigation to {page_name}: {at.exception}"
        assert at.session_state[KEY_ACTIVE_NAV] == page_name


def test_sentiment_binary_model_integrity():
    """
    Epistemological Integrity Test:
    Verify binary model never emits neutral class label and handles empty input cleanly.
    """
    # 1. Train binary sentiment classifier
    X = ["amazing outstanding wonderful", "terrible awful broken garbage"]
    y = ["positive", "negative"]
    clf = TfidfLogisticRegressionSentiment(random_state=42)
    clf.fit(X, y)

    assert set(clf.classes_) == {"negative", "positive"}

    # 2. Valid input produces binary label only
    pred_pos = clf.predict_single("amazing product")
    assert pred_pos["predicted_label"] == "positive"
    assert "neutral" not in pred_pos["probabilities"]

    # 3. Empty input returns non-prediction with None label (never fabricates neutral!)
    pred_empty = clf.predict_single("")
    assert pred_empty["is_valid_input"] is False
    assert pred_empty["predicted_label"] is None
    assert pred_empty["status"] == "empty_input"
    assert pred_empty["confidence"] == 0.0

    # 4. Whitespace-only input returns non-prediction
    pred_space = clf.predict_single("   \n\t  ")
    assert pred_space["is_valid_input"] is False
    assert pred_space["predicted_label"] is None


def test_intelligence_markdown_export():
    """Verify intelligence report formats cleanly into markdown for executive export."""
    now_iso = "2026-11-07T10:00:00"
    report = IntelligenceReport(
        report_id="rep_test_123",
        generated_at=now_iso,
        status="completed",
        dataset_scope={"total_records": 50},
        executive_summary=ExecutiveSummary(
            overview="Overall strong positive customer sentiment.",
            positive_signals=["Fast delivery speeds", "High build quality"],
            recurring_issues=["Delayed customer service responses"],
            business_interpretation="Operational focus required on support SLA.",
        ),
        sentiment_intelligence=SentimentInsight(
            dominant_sentiment="positive",
            positive_themes=["speed", "build quality"],
            negative_themes=["support latency"],
            distribution_summary="70% positive feedback",
            model_limitations="Trained on binary customer feedback.",
        ),
        recommendations=[
            MarketingRecommendation(
                recommendation_id="REC-001",
                title="Expand Delivery Messaging",
                description="Promote delivery velocity in marketing campaigns.",
                suggested_action="Deploy banner ads featuring 2-day delivery.",
                supporting_observation="Fast delivery cited in 45% of reviews.",
                evidence_ids=["REV-001", "REV-002"],
                impact_category="retention",
                priority="high",
                confidence=0.92,
            )
        ],
        limitations_disclaimer="LLM synthesis grounded in sampled feedback.",
        execution_metadata=LLMExecutionMetadata(
            provider="mock",
            model="mock-v1",
            total_tokens=150,
            status="success",
            executed_at=now_iso,
        ),
    )

    md_output = _format_markdown_report(report)
    assert "# MarketLens Executive Intelligence Report" in md_output
    assert "rep_test_123" in md_output
    assert "Fast delivery speeds" in md_output
    assert "[HIGH] Expand Delivery Messaging" in md_output
    assert "LLM synthesis grounded in sampled feedback" in md_output


def test_intelligence_page_privacy_notice_offline():
    """Verify AI Intelligence page renders clear offline notice without provider."""
    at = AppTest.from_file(APP_PATH, default_timeout=15).run()
    # Load demo dataset
    at.sidebar.button[0].click().run()

    # Navigate to AI Marketing Intelligence
    at.sidebar.radio[0].set_value("AI Marketing Intelligence").run()
    assert not at.exception

    # Check info messages for offline notice
    info_texts = [info.value for info in at.info]
    assert any("Offline Mode" in t for t in info_texts)
    assert any("100% functional offline" in t for t in info_texts)


def test_intelligence_page_privacy_notice_when_provider_configured(monkeypatch):
    """Verify AI Intelligence page displays persistent privacy notice with provider."""
    mock_prov = MockLLMProvider(model_name="gpt-4o-mini")
    monkeypatch.setattr(
        "marketlens.dashboard.pages.intelligence.create_llm_provider",
        lambda *args, **kwargs: mock_prov,
    )

    at = AppTest.from_file(APP_PATH, default_timeout=15).run()
    # Load demo dataset
    at.sidebar.button[0].click().run()

    # Navigate to AI Marketing Intelligence
    at.sidebar.radio[0].set_value("AI Marketing Intelligence").run()
    assert not at.exception

    # Check warnings for data privacy notice
    warn_texts = [w.value for w in at.warning]
    assert any("Data Privacy Notice" in t for t in warn_texts)
    assert any("Automated PII removal is not implemented" in t for t in warn_texts)
    assert any("remain 100% functional offline" in t for t in warn_texts)
