"""Unit tests for MarketLens supervised sentiment analysis."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from marketlens.nlp.dataset import (
    prepare_sentiment_dataset,
    rating_to_sentiment,
    split_sentiment_dataset,
)
from marketlens.nlp.sentiment import (
    MajorityBaselineModel,
    TfidfLogisticRegressionSentiment,
    evaluate_sentiment_model,
    load_sentiment_model,
    save_sentiment_model,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_feedback_df():
    """Create a balanced synthetic dataset with known sentiments and edge cases."""
    return pd.DataFrame(
        [
            # Positive
            {
                "review_id": "R01",
                "review_text": "Excellent laptop, fast delivery and great quality!",
                "rating": 5.0,
                "_status": "accepted",
            },
            {
                "review_id": "R02",
                "review_text": "Outstanding sound, truly loved this headset.",
                "rating": 5.0,
                "_status": "accepted",
            },
            {
                "review_id": "R03",
                "review_text": "Highly recommend, best purchase this year.",
                "rating": 4.0,
                "_status": "accepted",
            },
            {
                "review_id": "R04",
                "review_text": "Very happy with the prompt support.",
                "rating": 4.5,
                "_status": "accepted",
            },
            # Neutral
            {
                "review_id": "R05",
                "review_text": "Item was okay but nothing special.",
                "rating": 3.0,
                "_status": "accepted",
            },
            {
                "review_id": "R06",
                "review_text": "Average product, does the job as described.",
                "rating": 3.0,
                "_status": "accepted",
            },
            {
                "review_id": "R07",
                "review_text": "Functional device, standard build.",
                "rating": 2.5,
                "_status": "accepted",
            },
            # Negative
            {
                "review_id": "R08",
                "review_text": "Poor quality, stopped working after a week.",
                "rating": 1.0,
                "_status": "accepted",
            },
            {
                "review_id": "R09",
                "review_text": "Arrived damaged, very disappointed with seller.",
                "rating": 1.5,
                "_status": "accepted",
            },
            {
                "review_id": "R10",
                "review_text": "Terrible customer service, will not buy again.",
                "rating": 2.0,
                "_status": "accepted",
            },
            {
                "review_id": "R11",
                "review_text": "Would not recommend, defective unit.",
                "rating": 1.0,
                "_status": "accepted",
            },
            # Edge cases
            {
                "review_id": "R12",
                "review_text": "Corrupted rating row.",
                "rating": 99.0,
                "_status": "flagged",
            },  # Invalid rating
            {
                "review_id": "R13",
                "review_text": "Missing rating row.",
                "rating": None,
                "_status": "accepted",
            },  # Missing rating
            {
                "review_id": "R14",
                "review_text": "   ",
                "rating": 5.0,
                "_status": "accepted",
            },  # Empty text
            {
                "review_id": "R15",
                "review_text": "Rejected record text.",
                "rating": 5.0,
                "_status": "rejected",
            },  # Rejected
        ]
    )


# ---------------------------------------------------------------------------
# Rating-to-Sentiment Mapping & Data-Integrity Tests
# ---------------------------------------------------------------------------


def test_rating_to_sentiment_1_to_5_scale():
    """Test standard 1-5 rating mapping into negative, neutral, positive."""
    assert rating_to_sentiment(1.0, scale="1-5") == "negative"
    assert rating_to_sentiment(2.0, scale="1-5") == "negative"
    assert rating_to_sentiment(3.0, scale="1-5") == "neutral"
    assert rating_to_sentiment(4.0, scale="1-5") == "positive"
    assert rating_to_sentiment(5.0, scale="1-5") == "positive"


def test_rating_to_sentiment_invalid_ratings_integrity():
    """Ensure invalid out-of-scale ratings NEVER become sentiment labels."""
    assert rating_to_sentiment(0.0, scale="1-5") is None
    assert rating_to_sentiment(-1.0, scale="1-5") is None
    assert rating_to_sentiment(6.0, scale="1-5") is None
    assert rating_to_sentiment(100.0, scale="1-5") is None
    assert rating_to_sentiment(None, scale="1-5") is None
    assert rating_to_sentiment("unparseable", scale="1-5") is None


def test_rating_to_sentiment_different_scales():
    """Test rating-to-sentiment mapping across non-1-5 scales."""
    # 1-10 scale
    assert rating_to_sentiment(2.0, scale="1-10") == "negative"
    assert rating_to_sentiment(5.0, scale="1-10") == "neutral"
    assert rating_to_sentiment(9.0, scale="1-10") == "positive"
    assert rating_to_sentiment(12.0, scale="1-10") is None

    # Thumbs scale (0/1)
    assert rating_to_sentiment(0.0, scale="thumbs") == "negative"
    assert rating_to_sentiment(1.0, scale="thumbs") == "positive"
    assert rating_to_sentiment(2.0, scale="thumbs") is None


# ---------------------------------------------------------------------------
# Dataset Preparation & Leakage Prevention Tests
# ---------------------------------------------------------------------------


def test_prepare_sentiment_dataset(synthetic_feedback_df):
    """Test dataset preparation filtering rejected rows and empty text."""
    prep_df = prepare_sentiment_dataset(
        synthetic_feedback_df,
        include_flagged=False,
    )

    # R12 (invalid), R13 (missing), R14 (empty text), R15 (rejected) excluded
    assert len(prep_df) == 11
    assert "sentiment" in prep_df.columns
    assert set(prep_df["sentiment"].unique()) == {"positive", "neutral", "negative"}
    # Verify no rejected records included
    if "_status" in prep_df.columns:
        assert (prep_df["_status"] == "rejected").sum() == 0


def test_prepare_sentiment_dataset_deduplication():
    """Test text deduplication to prevent cross-partition train/test leakage."""
    df = pd.DataFrame(
        [
            {
                "review_text": "Duplicate feedback review.",
                "rating": 5.0,
                "_status": "accepted",
            },
            {
                "review_text": "Duplicate feedback review.",
                "rating": 5.0,
                "_status": "accepted",
            },
            {
                "review_text": "Unique feedback review.",
                "rating": 1.0,
                "_status": "accepted",
            },
        ]
    )
    prep_df = prepare_sentiment_dataset(df, deduplicate_text=True)
    assert len(prep_df) == 2


def test_split_sentiment_dataset_stratification(synthetic_feedback_df):
    """Test deterministic stratified train/test split."""
    prep_df = prepare_sentiment_dataset(synthetic_feedback_df)
    train_df, test_df = split_sentiment_dataset(
        prep_df,
        test_size=0.25,
        random_state=42,
        stratify=True,
    )

    assert len(train_df) > 0
    assert len(test_df) > 0
    assert len(train_df) + len(test_df) == len(prep_df)

    # Ensure zero text overlap between train and test
    train_texts = set(train_df["review_text"])
    test_texts = set(test_df["review_text"])
    assert len(train_texts.intersection(test_texts)) == 0


# ---------------------------------------------------------------------------
# Baseline & Logistic Regression Model Tests
# ---------------------------------------------------------------------------


def test_majority_baseline_model():
    """Test majority baseline fitting and predictions."""
    X_train = ["Good", "Great", "Bad", "Superb"]
    y_train = ["positive", "positive", "negative", "positive"]

    baseline = MajorityBaselineModel()
    baseline.fit(X_train, y_train)

    assert baseline.majority_class_ == "positive"
    preds = baseline.predict(["Horrible quality", "Amazing"])
    assert preds == ["positive", "positive"]

    probs = baseline.predict_proba(["Test text"])
    assert probs.shape == (1, 2)


def test_tfidf_logistic_regression_fit_predict():
    """Test TF-IDF + Logistic Regression fitting and prediction."""
    X_train = [
        "This product is amazing and works great",
        "Excellent build quality and wonderful experience",
        "Terrible product, completely broken and useless",
        "Horrible service, worst purchase ever made",
    ]
    y_train = ["positive", "positive", "negative", "negative"]

    model = TfidfLogisticRegressionSentiment(random_state=42)
    model.fit(X_train, y_train)

    assert model.is_fitted
    assert len(model.classes_) == 2

    # Predict positive and negative instances
    preds = model.predict(["Amazing and great", "Terrible and broken"])
    assert preds[0] == "positive"
    assert preds[1] == "negative"


def test_negation_preservation_in_sentiment():
    """Verify negation words like 'not' are not stripped by stopword filtering."""
    X_train = [
        "Good product, happy with it",
        "Not good, defective and stopped working",
        "Great quality",
        "Not recommended at all",
    ]
    y_train = ["positive", "negative", "positive", "negative"]

    model = TfidfLogisticRegressionSentiment(random_state=42)
    model.fit(X_train, y_train)

    # Ensure 'not' is in the vocabulary
    vocab = model.vectorizer.vocabulary_
    assert "not" in vocab


def test_predict_single_with_uncertainty():
    """Test single review prediction structure and uncertainty scoring."""
    X_train = [
        "Love this laptop so much!",
        "Hate this product, waste of money.",
    ]
    y_train = ["positive", "negative"]

    model = TfidfLogisticRegressionSentiment(random_state=42)
    model.fit(X_train, y_train)

    res = model.predict_single("Love this laptop so much!", record_id="R001")
    assert res["record_id"] == "R001"
    assert res["predicted_label"] == "positive"
    assert res["confidence"] > 0.5
    assert "margin" in res
    assert "probabilities" in res
    assert res["is_valid_input"] is True

    # Empty string input
    empty_res = model.predict_single("   ", record_id="R002")
    assert empty_res["is_valid_input"] is False
    assert empty_res["is_uncertain"] is True
    assert empty_res["predicted_label"] is None
    assert empty_res["status"] == "empty_input"
    assert empty_res["confidence"] == 0.0


def test_predict_batch():
    """Test batch prediction appending columns to DataFrame."""
    X_train = ["Awesome product", "Defective garbage"]
    y_train = ["positive", "negative"]

    model = TfidfLogisticRegressionSentiment(random_state=42)
    model.fit(X_train, y_train)

    df = pd.DataFrame(
        {
            "review_id": ["A1", "A2"],
            "review_text": ["Awesome experience", "Defective and broken"],
        }
    )

    batch_df = model.predict_batch(df)
    assert "predicted_sentiment" in batch_df.columns
    assert "sentiment_confidence" in batch_df.columns
    assert "sentiment_margin" in batch_df.columns
    assert "is_uncertain" in batch_df.columns
    assert len(batch_df) == 2


# ---------------------------------------------------------------------------
# Evaluation & Persistence Tests
# ---------------------------------------------------------------------------


def test_evaluate_sentiment_model():
    """Test model evaluation metric calculation."""
    X_train = ["Great product", "Poor product"]
    y_train = ["positive", "negative"]
    X_test = ["Great", "Poor"]
    y_test = ["positive", "negative"]

    model = TfidfLogisticRegressionSentiment(random_state=42)
    model.fit(X_train, y_train)

    eval_res = evaluate_sentiment_model(model, X_test, y_test)
    assert "accuracy" in eval_res
    assert "macro_f1" in eval_res
    assert "weighted_f1" in eval_res
    assert "confusion_matrix" in eval_res
    assert "precision_by_class" in eval_res
    assert "recall_by_class" in eval_res
    assert eval_res["accuracy"] == 1.0


def test_save_and_load_sentiment_model(tmp_path):
    """Test saving and loading sentiment model artifact."""
    X_train = ["Great laptop", "Broken headphones"]
    y_train = ["positive", "negative"]

    model = TfidfLogisticRegressionSentiment(random_state=42)
    model.fit(X_train, y_train)

    saved_path = save_sentiment_model(
        model, output_dir=tmp_path, filename="test_model.joblib"
    )
    assert os.path.exists(saved_path)

    loaded_model = load_sentiment_model(saved_path)
    assert loaded_model.is_fitted
    assert loaded_model.classes_ == model.classes_

    pred_orig = model.predict(["Great laptop"])
    pred_loaded = loaded_model.predict(["Great laptop"])
    assert pred_orig == pred_loaded


# ---------------------------------------------------------------------------
# Regression Tests (Batch 3.1: Evaluation Integrity & Empty Input Verification)
# ---------------------------------------------------------------------------


def test_binary_model_classes_remain_binary_after_training():
    """Verify binary model classes remain strictly binary without fabricated classes."""
    X_train = [
        "This is an amazing laptop",
        "Great battery life and build",
        "Terrible customer service",
        "Worst purchase ever made",
    ]
    y_train = ["positive", "positive", "negative", "negative"]

    model = TfidfLogisticRegressionSentiment(random_state=42)
    model.fit(X_train, y_train)

    assert set(model.classes_) == {"negative", "positive"}
    assert len(model.classes_) == 2
    assert "neutral" not in model.classes_
    assert model.training_metadata["classes"] == ["negative", "positive"]


def test_binary_predictions_never_produce_neutral_for_valid_input():
    """Ensure binary models never emit 'neutral' for valid non-empty inputs."""
    X_train = [
        "Excellent product and great quality",
        "Outstanding performance, loved it",
        "Horrible build, completely defective",
        "Awful experience, never buying again",
    ]
    y_train = ["positive", "positive", "negative", "negative"]

    model = TfidfLogisticRegressionSentiment(random_state=42)
    model.fit(X_train, y_train)

    test_samples = [
        "It was okay I guess",
        "Average and standard",
        "Not sure if good or bad",
        "Great quality",
        "Broken device",
    ]

    preds = model.predict(test_samples)
    for p in preds:
        assert p in {"positive", "negative"}
        assert p != "neutral"

    for sample in test_samples:
        single_res = model.predict_single(sample)
        assert single_res["predicted_label"] in {"positive", "negative"}
        assert single_res["predicted_label"] != "neutral"
        assert single_res["is_valid_input"] is True
        assert single_res["status"] == "predicted"


def test_three_class_models_can_predict_all_classes():
    """Verify three-class model correctly supports and predicts all three classes."""
    X_train = [
        "Outstanding performance and fast charging",
        "Superb build quality, very happy",
        "Average product, does the job moderately",
        "Standard item, neither great nor bad",
        "Completely useless, defective and broken",
        "Terrible support, waste of money",
    ]
    y_train = [
        "positive",
        "positive",
        "neutral",
        "neutral",
        "negative",
        "negative",
    ]

    model = TfidfLogisticRegressionSentiment(random_state=42)
    model.fit(X_train, y_train)

    assert set(model.classes_) == {"negative", "neutral", "positive"}
    assert len(model.classes_) == 3

    # Test predictions for all three classes
    pos_res = model.predict_single("Outstanding performance and great")
    neu_res = model.predict_single("Average product and standard")
    neg_res = model.predict_single("Completely broken and defective")

    assert pos_res["predicted_label"] == "positive"
    assert neu_res["predicted_label"] == "neutral"
    assert neg_res["predicted_label"] == "negative"


def test_empty_and_whitespace_input_no_fabricated_neutral():
    """Verify empty and whitespace inputs do not fabricate neutral predictions."""
    X_train = ["Good item", "Bad item"]
    y_train = ["positive", "negative"]

    binary_model = TfidfLogisticRegressionSentiment(random_state=42)
    binary_model.fit(X_train, y_train)

    empty_inputs = ["", "   ", "\t\t\n", " \r\n "]

    for inp in empty_inputs:
        res = binary_model.predict_single(inp)
        assert res["predicted_label"] is None
        assert res["predicted_label"] != "neutral"
        assert res["is_valid_input"] is False
        assert res["status"] == "empty_input"
        assert res["confidence"] == 0.0

    # Test predict() batch with empty inputs
    list_preds = binary_model.predict(["", "   ", "Good item"])
    assert list_preds == [None, None, "positive"]

    # Test predict_proba() batch with empty inputs
    probs = binary_model.predict_proba(["", "   "])
    assert probs.shape == (2, 2)
    assert np.all(probs == 0.0)

    # Test predict_batch DataFrame
    batch_df = pd.DataFrame({"review_text": ["", "   ", "Good item"]})
    pred_df = binary_model.predict_batch(batch_df)
    assert pred_df["predicted_sentiment"].iloc[0] is None
    assert pred_df["predicted_sentiment"].iloc[1] is None
    assert pred_df["predicted_sentiment"].iloc[2] == "positive"
    assert pred_df["sentiment_confidence"].iloc[0] == 0.0
    assert bool(pred_df["is_valid_input"].iloc[0]) is False
    assert bool(pred_df["is_valid_input"].iloc[2]) is True


def test_empty_input_results_distinguishable_from_genuine_predictions():
    """Verify empty-input results are distinguishable from genuine predictions."""
    X_train = ["Great product", "Poor service"]
    y_train = ["positive", "negative"]

    model = TfidfLogisticRegressionSentiment(random_state=42)
    model.fit(X_train, y_train)

    valid_res = model.predict_single("Great product")
    empty_res = model.predict_single("   ")

    # Genuine prediction attributes
    assert valid_res["is_valid_input"] is True
    assert valid_res["status"] == "predicted"
    assert valid_res["predicted_label"] in model.classes_
    assert valid_res["confidence"] > 0.0
    assert sum(valid_res["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)

    # Empty non-prediction attributes
    assert empty_res["is_valid_input"] is False
    assert empty_res["status"] == "empty_input"
    assert empty_res["predicted_label"] is None
    assert empty_res["confidence"] == 0.0
    assert sum(empty_res["probabilities"].values()) == 0.0
    assert empty_res["is_uncertain"] is True


def test_probability_outputs_and_confidence_consistent_with_classes():
    """Verify probability outputs and confidence keys strictly match trained classes."""
    # Binary model
    bin_model = TfidfLogisticRegressionSentiment(random_state=42)
    bin_model.fit(["Superb", "Terrible"], ["positive", "negative"])

    bin_res = bin_model.predict_single("Superb")
    assert set(bin_res["probabilities"].keys()) == {"negative", "positive"}
    assert "neutral" not in bin_res["probabilities"]

    bin_empty_res = bin_model.predict_single("")
    assert set(bin_empty_res["probabilities"].keys()) == {"negative", "positive"}
    assert "neutral" not in bin_empty_res["probabilities"]
    assert bin_empty_res["probabilities"]["negative"] == 0.0
    assert bin_empty_res["probabilities"]["positive"] == 0.0

    # 3-class model
    tri_model = TfidfLogisticRegressionSentiment(random_state=42)
    tri_model.fit(
        ["Superb", "Okay", "Terrible"],
        ["positive", "neutral", "negative"],
    )

    tri_res = tri_model.predict_single("Superb")
    assert set(tri_res["probabilities"].keys()) == {"negative", "neutral", "positive"}

    tri_empty_res = tri_model.predict_single("")
    assert set(tri_empty_res["probabilities"].keys()) == {
        "negative",
        "neutral",
        "positive",
    }
    assert all(v == 0.0 for v in tri_empty_res["probabilities"].values())


def test_existing_valid_input_prediction_intact():
    """Verify valid input prediction behavior remains unchanged."""
    X_train = [
        "Highly recommended, fast delivery and wonderful quality",
        "Completely useless, defective and broken, hate it",
    ]
    y_train = ["positive", "negative"]

    model = TfidfLogisticRegressionSentiment(random_state=42)
    model.fit(X_train, y_train)

    pos_out = model.predict(["Highly recommended and wonderful"])[0]
    neg_out = model.predict(["Completely useless and defective"])[0]

    assert pos_out == "positive"
    assert neg_out == "negative"
