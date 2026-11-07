"""
MarketLens - Natural Language Processing Subsystem.

Provides:
- Supervised sentiment classification (Majority baseline, TF-IDF + Logistic Regression).
- Unsupervised topic discovery (TF-IDF + K-Means).
- Dataset preparation, weak label derivation, and leakage prevention.
- Model evaluation and uncertainty-aware inference.
- Unified NLP interface for dashboard integration.
"""

from __future__ import annotations

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
from marketlens.nlp.topics import (
    TopicDiscoveryModel,
    load_topic_model,
    save_topic_model,
)
from marketlens.nlp.unified import NLPManager

__all__ = [
    "MajorityBaselineModel",
    "NLPManager",
    "TfidfLogisticRegressionSentiment",
    "TopicDiscoveryModel",
    "evaluate_sentiment_model",
    "load_sentiment_model",
    "load_topic_model",
    "prepare_sentiment_dataset",
    "rating_to_sentiment",
    "save_sentiment_model",
    "save_topic_model",
    "split_sentiment_dataset",
]
