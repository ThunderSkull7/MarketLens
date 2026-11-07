"""Unit tests for the unified NLPManager interface."""

from __future__ import annotations

import os

import pandas as pd
import pytest

from marketlens.nlp.unified import NLPManager


@pytest.fixture
def clean_feedback_dataset():
    """Create a sample feedback dataset suitable for end-to-end NLPManager testing."""
    return pd.DataFrame(
        [
            {
                "review_id": "R01",
                "review_text": "Battery life is outstanding, very satisfied.",
                "rating": 5.0,
                "_status": "accepted",
            },
            {
                "review_id": "R02",
                "review_text": "Excellent battery standby and quick charge.",
                "rating": 5.0,
                "_status": "accepted",
            },
            {
                "review_id": "R03",
                "review_text": "Standard battery performance, nothing special.",
                "rating": 3.0,
                "_status": "accepted",
            },
            {
                "review_id": "R04",
                "review_text": "Battery drains in one hour, completely unusable.",
                "rating": 1.0,
                "_status": "accepted",
            },
            {
                "review_id": "R05",
                "review_text": "Fast delivery and packaging was intact.",
                "rating": 5.0,
                "_status": "accepted",
            },
            {
                "review_id": "R06",
                "review_text": "Courier was delayed by three days.",
                "rating": 2.0,
                "_status": "accepted",
            },
            {
                "review_id": "R07",
                "review_text": "Delivery arrived on time as expected.",
                "rating": 4.0,
                "_status": "accepted",
            },
            {
                "review_id": "R08",
                "review_text": "Terrible shipping experience, box arrived crushed.",
                "rating": 1.0,
                "_status": "accepted",
            },
        ]
    )


def test_nlp_manager_train_and_evaluate(clean_feedback_dataset):
    """Test full training and evaluation through NLPManager."""
    manager = NLPManager()
    res = manager.train_and_evaluate(
        clean_feedback_dataset,
        n_clusters=2,
        test_size=0.25,
        random_state=42,
    )

    assert "baseline_evaluation" in res
    assert "logistic_regression_evaluation" in res
    assert "topic_clusters" in res
    assert len(res["topic_clusters"]) == 2
    assert manager.sentiment_model is not None
    assert manager.topic_model is not None


def test_nlp_manager_inference_decoupling(clean_feedback_dataset):
    """Verify inference runs without retraining."""
    manager = NLPManager()
    manager.train_and_evaluate(clean_feedback_dataset, n_clusters=2)

    # Single inference
    single_res = manager.predict_sentiment("Great battery life")
    assert single_res["predicted_label"] in ["positive", "neutral", "negative"]

    # Batch inference
    batch_res = manager.predict_sentiment(clean_feedback_dataset)
    assert "predicted_sentiment" in batch_res.columns

    # Topic assignment
    topics = manager.assign_topics(["Battery drains quickly", "Courier delayed"])
    assert len(topics) == 2


def test_nlp_manager_save_and_load_artifacts(clean_feedback_dataset, tmp_path):
    """Test saving models and reports and loading them into a fresh NLPManager."""
    manager = NLPManager()
    manager.train_and_evaluate(clean_feedback_dataset, n_clusters=2)

    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    saved = manager.save_models(output_dir=models_dir, report_dir=reports_dir)

    assert "sentiment_model" in saved
    assert "topic_model" in saved
    assert "evaluation_report" in saved
    assert os.path.exists(saved["sentiment_model"])
    assert os.path.exists(saved["topic_model"])
    assert os.path.exists(saved["evaluation_report"])

    # Fresh manager loading models
    fresh_manager = NLPManager()
    fresh_manager.load_models(models_dir=models_dir)

    assert fresh_manager.sentiment_model is not None
    assert fresh_manager.topic_model is not None

    # Inference without retraining
    pred = fresh_manager.predict_sentiment("Great battery life")
    assert pred["predicted_label"] in ["positive", "neutral", "negative"]
