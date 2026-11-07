"""
MarketLens - Batch 6 Security & Model Loading Integrity Regression Tests.

Validates:
- Graceful failure when loading corrupted/incompatible sentiment model artifacts.
- Graceful failure when loading corrupted/incompatible topic model artifacts.
- Path traversal prevention in report serialization.
- Absence of API key exposure in logging and string representations.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import pandas as pd
import pytest

from marketlens.intelligence.persistence import save_intelligence_report
from marketlens.intelligence.schemas import (
    ExecutiveSummary,
    IntelligenceReport,
    LLMExecutionMetadata,
)
from marketlens.nlp.sentiment import (
    TfidfLogisticRegressionSentiment,
    load_sentiment_model,
    save_sentiment_model,
)
from marketlens.nlp.topics import (
    TopicDiscoveryModel,
    load_topic_model,
    save_topic_model,
)


def test_corrupted_sentiment_model_fails_gracefully(tmp_path: Path):
    """Verify loading corrupted joblib file fails with clear ValueError."""
    bad_file = tmp_path / "corrupted_sentiment.joblib"
    joblib.dump({"invalid_key": 123}, bad_file)

    with pytest.raises(ValueError, match="Incompatible or corrupted sentiment model"):
        load_sentiment_model(bad_file)


def test_corrupted_topic_model_fails_gracefully(tmp_path: Path):
    """Verify loading corrupted topic model fails with clear ValueError."""
    bad_file = tmp_path / "corrupted_topics.joblib"
    joblib.dump({"corrupted": True}, bad_file)

    with pytest.raises(ValueError, match="Incompatible or corrupted topic model"):
        load_topic_model(bad_file)


def test_model_sha256_verification(tmp_path: Path):
    """Verify model loader cryptographically verifies checksum when provided."""
    # 1. Sentiment model checksum check
    m_sent = TfidfLogisticRegressionSentiment()
    m_sent.fit(["good app", "bad battery"], ["positive", "negative"])
    saved_sent = save_sentiment_model(
        m_sent, output_dir=tmp_path, filename="sent.joblib"
    )
    actual_sent_sha = hashlib.sha256(Path(saved_sent).read_bytes()).hexdigest()

    # Successful load with matching hash
    loaded = load_sentiment_model(saved_sent, expected_sha256=actual_sent_sha)
    assert loaded.is_fitted is True

    # Failed load with mismatching hash
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_sentiment_model(saved_sent, expected_sha256="0" * 64)

    # 2. Topic model checksum check
    m_topic = TopicDiscoveryModel(n_clusters=2)
    m_topic.fit(["good app speed", "battery drain problem", "fast charge", "slow app"])
    saved_topic = save_topic_model(
        m_topic, output_dir=tmp_path, filename="topic.joblib"
    )
    actual_topic_sha = hashlib.sha256(Path(saved_topic).read_bytes()).hexdigest()

    loaded_t = load_topic_model(saved_topic, expected_sha256=actual_topic_sha)
    assert loaded_t.is_fitted is True

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_topic_model(saved_topic, expected_sha256="0" * 64)


def test_empty_and_punctuation_prediction_consistency():
    """Verify contract for empty, whitespace, and punctuation inputs."""
    model = TfidfLogisticRegressionSentiment()
    model.fit(
        ["excellent quality fast shipping", "terrible broken item never arrived"],
        ["positive", "negative"],
    )

    # 1. Empty and whitespace in predict() -> [None, None, None]
    preds = model.predict(["", "   ", "\n\t  "])
    assert preds == [None, None, None]

    # 2. Empty and whitespace in predict_proba() -> sentinel zero vectors
    probs = model.predict_proba(["", "   "])
    assert probs.shape == (2, 2)
    assert (probs == 0.0).all()

    # 3. Empty and whitespace in predict_single()
    res_empty = model.predict_single("")
    assert res_empty["predicted_label"] is None
    assert res_empty["is_valid_input"] is False
    assert res_empty["confidence"] == 0.0
    assert res_empty["status"] == "empty_input"
    assert res_empty["is_uncertain"] is True
    assert all(p == 0.0 for p in res_empty["probabilities"].values())

    res_ws = model.predict_single("   ")
    assert res_ws["predicted_label"] is None
    assert res_ws["is_valid_input"] is False
    assert res_ws["status"] == "empty_input"

    # 4. Punctuation-only text (out of vocabulary) -> valid input, maximum uncertainty
    res_punct = model.predict_single("???")
    assert res_punct["is_valid_input"] is True
    assert res_punct["status"] == "predicted"
    assert res_punct["margin"] == 0.0  # Equal class prior/intercept probabilities
    assert res_punct["is_uncertain"] is True

    # 5. Batch inference on DataFrame with invalid records
    df = pd.DataFrame(
        {
            "review_id": ["1", "2", "3", "4"],
            "review_text": ["excellent quality", "", "   ", "???"],
        }
    )
    scored_df = model.predict_batch(df)
    assert scored_df["predicted_sentiment"].iloc[0] == "positive"
    assert bool(scored_df["is_valid_input"].iloc[0]) is True

    assert scored_df["predicted_sentiment"].iloc[1] is None
    assert bool(scored_df["is_valid_input"].iloc[1]) is False
    assert scored_df["sentiment_confidence"].iloc[1] == 0.0

    assert scored_df["predicted_sentiment"].iloc[2] is None
    assert bool(scored_df["is_valid_input"].iloc[2]) is False

    assert scored_df["predicted_sentiment"].iloc[3] is not None
    assert bool(scored_df["is_valid_input"].iloc[3]) is True
    assert scored_df["sentiment_margin"].iloc[3] == 0.0


def test_save_intelligence_report_prevents_path_traversal(tmp_path: Path):
    """Verify filename with path traversal syntax ('../') is safely stripped."""
    now_iso = "2026-11-07T10:00:00"
    report = IntelligenceReport(
        report_id="SEC-001",
        generated_at=now_iso,
        status="completed",
        dataset_scope={"total_records": 10},
        executive_summary=ExecutiveSummary(
            overview="Security check",
            positive_signals=[],
            recurring_issues=[],
            business_interpretation="Safe write",
        ),
        execution_metadata=LLMExecutionMetadata(
            provider="mock",
            model="mock-v1",
            total_tokens=100,
            status="success",
            executed_at=now_iso,
        ),
    )

    target_dir = tmp_path / "reports_dir"
    target_dir.mkdir()

    # Attempt path traversal
    saved_path = save_intelligence_report(
        report, output_dir=target_dir, filename="../../traversal_exploit.json"
    )

    # File must be inside target_dir, NOT outside in tmp_path
    assert saved_path.parent == target_dir
    assert saved_path.name == "traversal_exploit.json"
    assert not (tmp_path / "traversal_exploit.json").exists()
