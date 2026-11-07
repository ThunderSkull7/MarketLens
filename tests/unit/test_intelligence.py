"""
Unit tests for MarketLens LLM-Powered Intelligence Layer.

Covers:
1. Provider interface behavior.
2. Provider configuration and missing credentials.
3. No-provider graceful behavior.
4. Valid structured response parsing.
5. Malformed JSON handling.
6. Pydantic validation failures.
7. Evidence-ID validation.
8. Rejection of fabricated review IDs.
9. Aggregate evidence handling.
10. Context preparation and input limits.
11. Data minimization.
12. Timeout and rate-limit handling.
13. Bounded retries.
14. Provider failures without affecting existing analytics.
15. Token/cost metadata handling when available.
16. Persistence and report reload.
17. Empty dataset and insufficient evidence behavior.
18. End-to-end intelligence generation with deterministic mocked provider.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from marketlens.analytics import dataset_overview, rating_analysis
from marketlens.config import MarketLensSettings
from marketlens.exceptions import (
    LLMAuthenticationError,
    LLMTimeoutError,
)
from marketlens.intelligence.context import prepare_intelligence_context
from marketlens.intelligence.engine import IntelligenceEngine
from marketlens.intelligence.persistence import (
    load_intelligence_report,
    save_intelligence_report,
)
from marketlens.intelligence.provider import (
    MockLLMProvider,
    OpenAICompatibleProvider,
    estimate_token_cost,
)
from marketlens.intelligence.schemas import (
    IntelligenceReport,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_feedback_df() -> pd.DataFrame:
    """Deterministic sample customer feedback dataset with stable IDs."""
    return pd.DataFrame(
        [
            {
                "review_id": "R01",
                "review_text": "Battery life is incredible, easily lasts two days.",
                "rating": 5.0,
                "sentiment": "positive",
                "product_id": "Laptop Pro",
                "_status": "accepted",
            },
            {
                "review_id": "R02",
                "review_text": "Screen resolution and brightness are unmatched.",
                "rating": 5.0,
                "sentiment": "positive",
                "product_id": "Laptop Pro",
                "_status": "accepted",
            },
            {
                "review_id": "R03",
                "review_text": "Keyboard feels standard, nothing extraordinary.",
                "rating": 3.0,
                "sentiment": "neutral",
                "product_id": "Laptop Pro",
                "_status": "accepted",
            },
            {
                "review_id": "R04",
                "review_text": "Customer service was completely unhelpful on defect.",
                "rating": 1.0,
                "sentiment": "negative",
                "product_id": "Laptop Pro",
                "_status": "accepted",
            },
            {
                "review_id": "R05",
                "review_text": "Charger broke quickly, terrible build quality.",
                "rating": 1.0,
                "sentiment": "negative",
                "product_id": "Accessory",
                "_status": "accepted",
            },
        ]
    )


@pytest.fixture
def sample_valid_llm_json(sample_feedback_df) -> dict:
    """Valid structured response conforming to the JSON schema."""
    return {
        "executive_summary": {
            "overview": "Praise for battery hardware; complaints on chargers.",
            "positive_signals": ["Battery longevity", "Display clarity"],
            "negative_signals": ["Charger durability", "Unresponsive support"],
            "recurring_issues": ["Broken charger units within first month"],
            "business_interpretation": "Hardware solid; peripheral defect risks brand.",
            "evidence_references": [
                {
                    "evidence_id": "R01",
                    "description": "User validates battery life",
                },
                {
                    "evidence_id": "METRIC:rating.mean",
                    "description": "Average rating baseline",
                },
            ],
        },
        "sentiment_intelligence": {
            "dominant_sentiment": "positive",
            "positive_themes": ["Battery longevity", "Display quality"],
            "negative_themes": ["Charging accessories", "Support"],
            "distribution_summary": "60% positive ratings anchored by laptop.",
            "model_limitations": "Labels derived from star ratings; weak supervision.",
            "evidence_references": [{"evidence_id": "R02"}],
        },
        "topic_insights": [
            {
                "cluster_id": 0,
                "topic_label": "Battery Performance",
                "interpretation": "Strong positive sentiment cluster.",
                "customer_concerns": ["Reliability on travel"],
                "business_relevance": "Key competitive differentiator for Q4.",
                "is_emerging": False,
                "evidence_references": [{"evidence_id": "R01"}],
            }
        ],
        "comparison_insights": [
            {
                "entity_type": "product",
                "comparison_name": "Laptop Pro vs Accessory",
                "findings": ["Laptop Pro has higher satisfaction than accessories."],
                "sample_size_adequate": True,
                "limitations": "Sample size is small. Does not infer market share.",
                "evidence_references": [{"evidence_id": "R05"}],
            }
        ],
        "recommendations": [
            {
                "title": "Upgrade Charger Build Quality",
                "description": "Switch supplier for charging cords to prevent failure.",
                "supporting_observation": "Accessories receive 1-star ratings.",
                "evidence_ids": ["R05"],
                "impact_category": "product_quality",
                "priority": "high",
                "confidence": 0.9,
                "suggested_action": "Audit accessory supplier warranties and redesign.",
            }
        ],
    }


# ---------------------------------------------------------------------------
# 1. Provider Interface Behavior
# ---------------------------------------------------------------------------


def test_provider_interface_behavior():
    """Verify MockLLMProvider conforms to LLMProvider and records telemetry."""
    provider = MockLLMProvider(
        response_content='{"test": 1}',
        model_name="mock-test-model",
        input_tokens=100,
        output_tokens=50,
        latency_ms=25.0,
    )
    assert provider.is_available() is True
    assert provider.get_model_name() == "mock-test-model"
    assert provider.get_provider_name() == "mock"

    resp = provider.generate("Test prompt", system_prompt="System instructions")
    assert resp.content == '{"test": 1}'
    assert resp.input_tokens == 100
    assert resp.output_tokens == 50
    assert resp.total_tokens == 150
    assert resp.latency_ms == 25.0
    assert provider.last_prompt == "Test prompt"
    assert provider.last_system_prompt == "System instructions"
    assert provider.call_count == 1


# ---------------------------------------------------------------------------
# 2. Provider Configuration and Missing Credentials
# ---------------------------------------------------------------------------


def test_provider_configuration_and_missing_credentials(monkeypatch):
    """Verify OpenAICompatibleProvider raises auth error when key missing."""
    monkeypatch.delenv("MARKETLENS_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    provider = OpenAICompatibleProvider(api_key=None)
    assert provider.is_available() is False

    with pytest.raises(LLMAuthenticationError) as exc_info:
        provider.generate("Prompt requiring API key")
    assert "credentials are not configured" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 3. No-Provider Graceful Behavior
# ---------------------------------------------------------------------------


def test_no_provider_graceful_behavior(sample_feedback_df):
    """Verify IntelligenceEngine returns unavailable report when provider None."""
    engine = IntelligenceEngine(provider=None)
    assert engine.is_available() is False

    report = engine.generate_report(sample_feedback_df)
    assert report.status == "unavailable"
    assert report.execution_metadata.status == "unavailable"
    assert "LLM provider is not configured" in str(
        report.execution_metadata.error_message
    )
    assert report.executive_summary is None


# ---------------------------------------------------------------------------
# 4. Valid Structured Response Parsing
# ---------------------------------------------------------------------------


def test_valid_structured_response_parsing(sample_feedback_df, sample_valid_llm_json):
    """Verify valid structured JSON is parsed and populated into report."""
    provider = MockLLMProvider(response_content=sample_valid_llm_json)
    engine = IntelligenceEngine(provider=provider)

    report = engine.generate_report(sample_feedback_df)
    assert report.status == "completed"
    assert report.execution_metadata.status == "success"
    assert report.executive_summary is not None
    assert "battery" in report.executive_summary.overview.lower()
    assert len(report.recommendations) == 1
    assert report.recommendations[0].title == "Upgrade Charger Build Quality"
    assert report.recommendations[0].evidence_ids == ["R05"]


# ---------------------------------------------------------------------------
# 5. Malformed JSON Handling
# ---------------------------------------------------------------------------


def test_malformed_json_handling(sample_feedback_df):
    """Verify malformed JSON from provider is safely caught."""
    broken_content = "Here is your report: {not valid json at all...}"
    provider = MockLLMProvider(response_content=broken_content)
    engine = IntelligenceEngine(provider=provider)

    report = engine.generate_report(sample_feedback_df)
    assert report.status == "failed"
    assert report.execution_metadata.status == "validation_error"
    assert "Malformed JSON response" in str(report.execution_metadata.error_message)


# ---------------------------------------------------------------------------
# 6. Pydantic Validation Failures
# ---------------------------------------------------------------------------


def test_pydantic_validation_failures(sample_feedback_df):
    """Verify invalid fields violating Pydantic return validation_error."""
    invalid_schema = {
        "executive_summary": {
            "overview": "Valid overview",
            "business_interpretation": "Valid",
        },
        "recommendations": "invalid_string_not_list",
    }
    provider = MockLLMProvider(response_content=invalid_schema)
    engine = IntelligenceEngine(provider=provider)

    report = engine.generate_report(sample_feedback_df)
    assert report.status == "failed"
    assert report.execution_metadata.status == "validation_error"
    assert "Pydantic schema validation error" in str(
        report.execution_metadata.error_message
    )


# ---------------------------------------------------------------------------
# 7. Evidence-ID Validation & 8. Rejection of Fabricated Review IDs
# ---------------------------------------------------------------------------


def test_evidence_id_validation_and_rejection_of_hallucinated_ids(
    sample_feedback_df,
):
    """Verify that hallucinated evidence IDs not in allowed set are stripped."""
    payload_with_hallucinations = {
        "executive_summary": {
            "overview": "Overview text",
            "business_interpretation": "Interpretation",
            "evidence_references": [
                {"evidence_id": "R01"},  # Real ID
                {"evidence_id": "FABRICATED_REVIEW_999"},  # Fake ID
            ],
        },
        "recommendations": [
            {
                "title": "Valid recommendation",
                "description": "Fix accessories",
                "supporting_observation": "Observed failure in chargers",
                "evidence_ids": ["R05", "NON_EXISTENT_ID"],
                "impact_category": "product_quality",
                "priority": "high",
                "confidence": 0.85,
                "suggested_action": "Redesign",
            },
            {
                "title": "Ungrounded recommendation without observation",
                "description": "Do something",
                "supporting_observation": "   ",  # Empty observation -> dropped
                "evidence_ids": ["R01"],
                "impact_category": "general",
                "priority": "low",
                "confidence": 0.5,
                "suggested_action": "None",
            },
        ],
    }

    provider = MockLLMProvider(response_content=payload_with_hallucinations)
    engine = IntelligenceEngine(provider=provider)

    report = engine.generate_report(sample_feedback_df)
    assert report.status == "completed"

    # Verify fake evidence ID was removed from executive summary references
    exec_refs = [
        ref.evidence_id for ref in report.executive_summary.evidence_references
    ]
    assert "R01" in exec_refs
    assert "FABRICATED_REVIEW_999" not in exec_refs

    # Verify fake evidence ID was removed from recommendation evidence_ids
    assert len(report.recommendations) == 1
    assert report.recommendations[0].evidence_ids == ["R05"]
    assert "NON_EXISTENT_ID" not in report.recommendations[0].evidence_ids


# ---------------------------------------------------------------------------
# 9. Aggregate Evidence Handling
# ---------------------------------------------------------------------------


def test_aggregate_evidence_handling(sample_feedback_df):
    """Verify aggregate metric IDs are recognized as metric evidence."""
    payload = {
        "executive_summary": {
            "overview": "Overview with metric evidence",
            "business_interpretation": "Interpretation",
            "evidence_references": [
                {"evidence_id": "METRIC:rating.mean", "description": "Rating mean"},
                {"evidence_id": "R01", "description": "Review text"},
            ],
        }
    }
    provider = MockLLMProvider(response_content=payload)
    engine = IntelligenceEngine(provider=provider)

    analytics = {"rating_analysis": {"mean": 4.2}}
    report = engine.generate_report(sample_feedback_df, analytics_summary=analytics)

    refs = report.executive_summary.evidence_references
    ref_map = {r.evidence_id: r.evidence_type for r in refs}

    assert "METRIC:rating.mean" in ref_map
    assert ref_map["METRIC:rating.mean"] == "metric"
    assert "R01" in ref_map
    assert ref_map["R01"] == "review"


# ---------------------------------------------------------------------------
# 10. Context Preparation and Input Limits & 11. Data Minimization
# ---------------------------------------------------------------------------


def test_context_preparation_and_data_minimization(sample_feedback_df):
    """Verify context preparation respects limits, truncates, and minimizes."""
    context = prepare_intelligence_context(
        sample_feedback_df,
        max_reviews=3,
        max_review_chars=40,
    )

    assert context.sampled_review_count <= 3
    assert len(context.allowed_evidence_ids) >= context.sampled_review_count
    for _, rev_data in context.review_evidence_map.items():
        assert len(rev_data["text"]) <= 45  # including ellipsis margin
    assert "R01" in context.allowed_evidence_ids


# ---------------------------------------------------------------------------
# 12. Timeout and Rate-Limit Handling & 13. Bounded Retries
# ---------------------------------------------------------------------------


def test_timeout_and_rate_limit_bounded_retries(sample_feedback_df):
    """Verify transient errors retried up to max_retries before failed report."""
    mock_provider = MockLLMProvider()
    mock_provider.error_to_raise = LLMTimeoutError("Request timed out")

    settings = MarketLensSettings(
        llm_max_retries=2,
        llm_retry_backoff_factor=1.0,
    )
    engine = IntelligenceEngine(provider=mock_provider, settings=settings)

    with patch("time.sleep") as mock_sleep:
        report = engine.generate_report(sample_feedback_df)

    assert mock_provider.call_count == 3
    assert mock_sleep.call_count == 2
    assert report.status == "failed"
    assert "Request timed out" in str(report.execution_metadata.error_message)


# ---------------------------------------------------------------------------
# 14. Provider Failures Without Affecting Existing Analytics
# ---------------------------------------------------------------------------


def test_provider_failures_without_affecting_analytics(sample_feedback_df):
    """Verify that an LLM failure never impedes standard analytics execution."""
    broken_provider = MockLLMProvider(error_to_raise=RuntimeError("Provider crashed"))
    engine = IntelligenceEngine(provider=broken_provider)

    # Run analytics independently
    overview = dataset_overview(sample_feedback_df)
    r_analysis = rating_analysis(sample_feedback_df)

    assert overview["total_records"] == 5
    assert r_analysis["mean"] == 3.0

    # Run intelligence engine which fails gracefully
    report = engine.generate_report(
        sample_feedback_df, analytics_summary={"overview": overview}
    )
    assert report.status == "failed"

    # Confirm analytics can still be executed after the LLM failure
    post_overview = dataset_overview(sample_feedback_df)
    assert post_overview["total_records"] == 5


# ---------------------------------------------------------------------------
# 15. Token and Cost Metadata Handling
# ---------------------------------------------------------------------------


def test_token_and_cost_metadata_handling():
    """Verify exact token cost calculation for known models and None for unknown."""
    # gpt-4o-mini pricing: $0.15/1M input, $0.60/1M output
    cost = estimate_token_cost(
        "gpt-4o-mini", input_tokens=1_000_000, output_tokens=1_000_000
    )
    assert cost == pytest.approx(0.75, abs=1e-5)

    # Unknown model: must return None (no invented costs)
    assert estimate_token_cost("custom-unknown-llm", 100, 100) is None
    # Missing tokens: must return None
    assert estimate_token_cost("gpt-4o-mini", None, 100) is None


# ---------------------------------------------------------------------------
# 16. Persistence and Report Reload
# ---------------------------------------------------------------------------


def test_persistence_and_report_reload(
    sample_feedback_df, sample_valid_llm_json, tmp_path
):
    """Verify IntelligenceReport can be persisted and reloaded cleanly."""
    provider = MockLLMProvider(
        response_content=sample_valid_llm_json,
        model_name="gpt-4o-mini",
        input_tokens=500,
        output_tokens=300,
    )
    engine = IntelligenceEngine(provider=provider)

    report = engine.generate_report(sample_feedback_df)
    assert report.status == "completed"

    saved_path = save_intelligence_report(
        report, output_dir=tmp_path, filename="test_report.json"
    )
    assert saved_path.exists()

    loaded_report = load_intelligence_report(saved_path)
    assert isinstance(loaded_report, IntelligenceReport)
    assert loaded_report.report_id == report.report_id
    assert loaded_report.status == "completed"
    assert loaded_report.executive_summary.overview == report.executive_summary.overview
    assert len(loaded_report.recommendations) == len(report.recommendations)
    assert loaded_report.execution_metadata.estimated_cost_usd is not None


# ---------------------------------------------------------------------------
# 17. Empty Dataset and Insufficient Evidence Behavior
# ---------------------------------------------------------------------------


def test_empty_dataset_and_insufficient_evidence_behavior():
    """Verify empty dataset returns failed report with clear evidence message."""
    provider = MockLLMProvider(response_content='{"should_not_run": true}')
    engine = IntelligenceEngine(provider=provider)

    empty_df = pd.DataFrame()
    report = engine.generate_report(empty_df)

    assert report.status == "failed"
    assert (
        "insufficient evidence" in str(report.execution_metadata.error_message).lower()
    )
    assert provider.call_count == 0  # Does not waste LLM call on empty data


# ---------------------------------------------------------------------------
# 18. End-to-End Intelligence Generation
# ---------------------------------------------------------------------------


def test_end_to_end_intelligence_generation(
    sample_feedback_df, sample_valid_llm_json, tmp_path
):
    """Verify end-to-end intelligence pipeline from analytics to report."""
    provider = MockLLMProvider(
        response_content=sample_valid_llm_json,
        model_name="gpt-4o-mini",
        input_tokens=1200,
        output_tokens=450,
    )
    engine = IntelligenceEngine(provider=provider)

    analytics = {
        "overview": dataset_overview(sample_feedback_df),
        "rating_analysis": rating_analysis(sample_feedback_df),
    }
    nlp_results = {
        "topic_clusters": [
            {
                "cluster_id": 0,
                "top_keywords": ["battery", "charge", "life"],
                "size": 3,
                "percentage": 60.0,
                "mean_rating": 4.5,
            }
        ]
    }

    report = engine.generate_report(
        sample_feedback_df,
        analytics_summary=analytics,
        nlp_results=nlp_results,
    )

    assert report.status == "completed"
    assert report.execution_metadata.input_tokens == 1200
    assert report.execution_metadata.output_tokens == 450
    assert report.execution_metadata.estimated_cost_usd is not None
    assert report.limitations_disclaimer != ""
    assert len(report.allowed_evidence_ids) > 0

    # Save and verify
    file_path = save_intelligence_report(report, output_dir=tmp_path)
    assert Path(file_path).exists()
    reloaded = load_intelligence_report(file_path)
    assert reloaded.report_id == report.report_id
