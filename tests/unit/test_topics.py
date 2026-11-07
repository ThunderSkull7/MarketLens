"""Unit tests for MarketLens unsupervised topic discovery."""

from __future__ import annotations

import os

import pytest

from marketlens.nlp.topics import (
    TopicDiscoveryModel,
    load_topic_model,
    save_topic_model,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def topic_sample_texts():
    """Create sample feedback with clear thematic clusters (battery, delivery)."""
    return [
        # Battery / power theme
        "Battery life lasts all day long without needing a charge.",
        "The battery drains too quickly, need to recharge constantly.",
        "Charging speed is great, battery performance is solid.",
        "Battery standby time is impressive.",
        # Delivery / shipping theme
        "Fast shipping, parcel arrived two days ahead of schedule.",
        "Delivery was delayed and the packaging box was torn.",
        "Package arrived safely, courier shipping was prompt.",
        "Quick delivery and excellent protective packaging.",
        # Audio / sound theme
        "Sound quality is crystal clear with rich bass tones.",
        "Audio volume is too low, sound muffled on calls.",
        "Headphones sound output is phenomenal, noise cancellation works.",
        "Crisp treble and balanced audio acoustics.",
    ]


# ---------------------------------------------------------------------------
# Topic Discovery Tests
# ---------------------------------------------------------------------------


def test_topic_discovery_fitting_and_summaries(topic_sample_texts):
    """Test TF-IDF + K-Means fitting and cluster summary structure."""
    model = TopicDiscoveryModel(n_clusters=3, random_state=42)
    model.fit(topic_sample_texts, top_terms_count=5, num_representative=2)

    assert model.is_fitted
    summaries = model.get_cluster_summaries()
    assert len(summaries) == 3

    total_pct = 0.0
    for s in summaries:
        assert "cluster_id" in s
        assert "label" in s
        assert "record_count" in s
        assert "record_percentage" in s
        assert "top_terms" in s
        assert "representative_records" in s
        assert len(s["top_terms"]) > 0
        assert len(s["representative_records"]) > 0
        total_pct += s["record_percentage"]

    # Total percentage across clusters should sum to ~100%
    assert round(total_pct) == 100


def test_topic_discovery_ratings_alignment(topic_sample_texts):
    """Test incorporating ratings and sentiments into cluster summaries."""
    ratings = [5.0, 1.0, 4.0, 5.0, 5.0, 2.0, 4.0, 5.0, 5.0, 2.0, 5.0, 4.0]
    sentiments = [
        "positive",
        "negative",
        "positive",
        "positive",
        "positive",
        "negative",
        "positive",
        "positive",
        "positive",
        "negative",
        "positive",
        "positive",
    ]

    model = TopicDiscoveryModel(n_clusters=3, random_state=42)
    model.fit(topic_sample_texts, ratings=ratings, sentiments=sentiments)

    summaries = model.get_cluster_summaries()
    for s in summaries:
        if s["record_count"] > 0:
            assert s["average_rating"] is not None
            assert isinstance(s["average_rating"], float)
            assert len(s["rating_distribution"]) > 0
            assert len(s["sentiment_distribution"]) > 0


def test_topic_discovery_assign_clusters(topic_sample_texts):
    """Test cluster assignment on new unseen reviews without refitting."""
    model = TopicDiscoveryModel(n_clusters=3, random_state=42)
    model.fit(topic_sample_texts)

    new_reviews = [
        "Battery holds charge for 15 hours easily.",
        "Delayed shipping courier delivered late.",
    ]
    assignments = model.assign_clusters(new_reviews)
    assert len(assignments) == 2
    assert all(isinstance(cid, int) and 0 <= cid < 3 for cid in assignments)


def test_topic_discovery_small_dataset():
    """Test graceful handling when dataset size is smaller than requested clusters."""
    small_texts = ["Battery is good", "Shipping was fast"]
    model = TopicDiscoveryModel(n_clusters=5, random_state=42)
    # n_clusters=5 with 2 samples should reduce k to 2 without failing
    model.fit(small_texts)
    assert len(model.get_cluster_summaries()) == 2


def test_topic_discovery_insufficient_samples():
    """Ensure error is raised when fewer than 2 valid text samples exist."""
    model = TopicDiscoveryModel(n_clusters=3)
    with pytest.raises(ValueError, match="Insufficient text records"):
        model.fit(["Only one text"])


def test_topic_discovery_silhouette_guards():
    """Verify silhouette score computed only when sample count and k are sufficient."""
    # Under 10 samples: silhouette score should be None (statistically guarded)
    texts_under_10 = [
        "Battery one",
        "Battery two",
        "Battery three",
        "Shipping one",
        "Shipping two",
        "Shipping three",
    ]
    model = TopicDiscoveryModel(n_clusters=2, random_state=42)
    model.fit(texts_under_10)
    assert model.silhouette_ is None

    # 12 samples: silhouette score should be computed
    texts_12 = texts_under_10 * 2
    model_12 = TopicDiscoveryModel(n_clusters=2, random_state=42)
    model_12.fit(texts_12)
    assert model_12.silhouette_ is not None
    assert isinstance(model_12.silhouette_, float)


def test_save_and_load_topic_model(topic_sample_texts, tmp_path):
    """Test serializing and loading TopicDiscoveryModel."""
    model = TopicDiscoveryModel(n_clusters=3, random_state=42)
    model.fit(topic_sample_texts)

    saved_path = save_topic_model(
        model, output_dir=tmp_path, filename="test_topics.joblib"
    )
    assert os.path.exists(saved_path)

    loaded = load_topic_model(saved_path)
    assert loaded.is_fitted
    assert loaded.n_clusters == model.n_clusters

    orig_assignments = model.assign_clusters(["Battery drained"])
    loaded_assignments = loaded.assign_clusters(["Battery drained"])
    assert orig_assignments == loaded_assignments
