"""
MarketLens - Unified NLP Interface.

Provides a consolidated, reproducible interface for sentiment classification,
topic discovery, and inference to support downstream dashboard integration
without duplicating machine learning logic.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import pandas as pd

from marketlens.nlp.dataset import prepare_sentiment_dataset, split_sentiment_dataset
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

logger = logging.getLogger("marketlens.nlp.unified")


class NLPManager:
    """
    Unified manager for sentiment classification and topic discovery.

    Decouples model training from inference, allowing pre-trained models to be
    loaded and used in Streamlit dashboards and batch jobs without retraining.
    """

    def __init__(
        self,
        sentiment_model: Optional[TfidfLogisticRegressionSentiment] = None,
        topic_model: Optional[TopicDiscoveryModel] = None,
    ) -> None:
        self.sentiment_model = sentiment_model
        self.topic_model = topic_model
        self.baseline_model: Optional[MajorityBaselineModel] = None
        self.evaluation_results: dict[str, Any] = {}

    def train_and_evaluate(
        self,
        df: pd.DataFrame,
        *,
        text_column: str = "review_text",
        rating_column: str = "rating",
        rating_scale: str = "1-5",
        test_size: float = 0.2,
        n_clusters: int = 3,
        random_state: int = 42,
        include_flagged: bool = False,
    ) -> dict[str, Any]:
        """
        Train and evaluate baseline, Logistic Regression, and K-Means topic models.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned feedback DataFrame.
        text_column : str, default "review_text"
            Review text column.
        rating_column : str, default "rating"
            Rating column.
        rating_scale : str, default "1-5"
            Rating scale.
        test_size : float, default 0.2
            Test set proportion.
        n_clusters : int, default 3
            Number of topic clusters.
        random_state : int, default 42
            Deterministic seed.
        include_flagged : bool, default False
            Whether to include flagged records.

        Returns
        -------
        dict[str, Any]
            Comprehensive training and evaluation results dictionary.
        """
        # 1. Prepare data
        prep_df = prepare_sentiment_dataset(
            df,
            text_column=text_column,
            rating_column=rating_column,
            rating_scale=rating_scale,
            include_flagged=include_flagged,
        )

        train_df, test_df = split_sentiment_dataset(
            prep_df,
            text_column=text_column,
            label_column="sentiment",
            test_size=test_size,
            random_state=random_state,
        )

        X_train = list(train_df[text_column])
        y_train = list(train_df["sentiment"])
        X_test = list(test_df[text_column]) if len(test_df) > 0 else X_train
        y_test = list(test_df["sentiment"]) if len(test_df) > 0 else y_train

        # 2. Train and evaluate Majority Baseline
        self.baseline_model = MajorityBaselineModel()
        self.baseline_model.fit(X_train, y_train)
        baseline_eval = evaluate_sentiment_model(
            self.baseline_model, X_test, y_test, model_name="majority_class_baseline"
        )

        # 3. Train and evaluate TF-IDF + Logistic Regression
        self.sentiment_model = TfidfLogisticRegressionSentiment(
            random_state=random_state
        )
        self.sentiment_model.fit(X_train, y_train)
        lr_eval = evaluate_sentiment_model(
            self.sentiment_model, X_test, y_test, model_name="tfidf_logistic_regression"
        )

        # 4. Train Topic Discovery Model
        all_texts = list(prep_df[text_column])
        all_ratings = (
            list(prep_df[rating_column]) if rating_column in prep_df.columns else None
        )
        all_sents = (
            list(prep_df["sentiment"]) if "sentiment" in prep_df.columns else None
        )

        self.topic_model = TopicDiscoveryModel(
            n_clusters=n_clusters, random_state=random_state
        )
        self.topic_model.fit(all_texts, ratings=all_ratings, sentiments=all_sents)

        self.evaluation_results = {
            "trained_at": datetime.now().isoformat(),
            "n_total_samples": len(prep_df),
            "n_train_samples": len(train_df),
            "n_test_samples": len(test_df),
            "rating_scale": rating_scale,
            "baseline_evaluation": baseline_eval,
            "logistic_regression_evaluation": lr_eval,
            "topic_clusters": self.topic_model.get_cluster_summaries(),
            "topic_metadata": self.topic_model.training_metadata,
        }

        return self.evaluation_results

    def predict_sentiment(
        self,
        data: Union[str, Sequence[str], pd.DataFrame],
        text_column: str = "review_text",
        id_column: str = "review_id",
    ) -> Any:
        """Predict sentiment for a single review, list of reviews, or DataFrame."""
        if self.sentiment_model is None:
            raise ValueError(
                "Sentiment model is not loaded. "
                "Call train_and_evaluate() or load_models()."
            )

        if isinstance(data, str):
            return self.sentiment_model.predict_single(data)

        if isinstance(data, pd.DataFrame):
            return self.sentiment_model.predict_batch(
                data, text_column=text_column, id_column=id_column
            )

        # Sequence of strings
        return [self.sentiment_model.predict_single(t) for t in data]

    def assign_topics(self, texts: Sequence[str]) -> list[int]:
        """Assign review texts to existing topic clusters."""
        if self.topic_model is None:
            raise ValueError(
                "Topic model is not loaded. Call train_and_evaluate() or load_models()."
            )
        return self.topic_model.assign_clusters(texts)

    def save_models(
        self,
        output_dir: Union[str, Path] = "models",
        report_dir: Union[str, Path] = "reports",
    ) -> dict[str, str]:
        """Persist all trained models and evaluation summaries to disk."""
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(report_dir, exist_ok=True)

        saved_paths: dict[str, str] = {}
        if self.sentiment_model is not None:
            saved_paths["sentiment_model"] = save_sentiment_model(
                self.sentiment_model, output_dir=output_dir
            )

        if self.topic_model is not None:
            saved_paths["topic_model"] = save_topic_model(
                self.topic_model, output_dir=output_dir
            )

        if self.evaluation_results:
            rep_path = os.path.join(
                report_dir,
                f"nlp_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            )
            with open(rep_path, "w", encoding="utf-8") as f:
                json.dump(self.evaluation_results, f, indent=2, ensure_ascii=False)
            saved_paths["evaluation_report"] = rep_path

        return saved_paths

    def load_models(
        self,
        models_dir: Union[str, Path] = "models",
        sentiment_filename: str = "sentiment_tfidf_logistic_regression.joblib",
        topic_filename: str = "topic_discovery_kmeans.joblib",
    ) -> None:
        """Load pre-trained sentiment and topic models for zero-training inference."""
        s_path = os.path.join(models_dir, sentiment_filename)
        if os.path.exists(s_path):
            self.sentiment_model = load_sentiment_model(s_path)
            logger.info("Loaded sentiment model from: %s", s_path)

        t_path = os.path.join(models_dir, topic_filename)
        if os.path.exists(t_path):
            self.topic_model = load_topic_model(t_path)
            logger.info("Loaded topic discovery model from: %s", t_path)
