"""
MarketLens - Supervised Sentiment Analysis Module.

Implements:
1. Majority-class baseline (DummyClassifier).
2. TF-IDF + Logistic Regression classifier with negation-aware tokenization.
3. Multi-metric evaluation (Accuracy, Precision, Recall, F1, Macro/Weighted F1).
4. Prediction interfaces for single and batch inference with uncertainty.
5. Model persistence and reloading.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from marketlens.analytics import DEFAULT_STOPWORDS
from marketlens.cleaning import normalize_text

logger = logging.getLogger("marketlens.nlp.sentiment")

# Negation tokens that carry critical polarity signals and must not be stripped
NEGATION_TOKENS = frozenset(
    {
        "no",
        "not",
        "never",
        "none",
        "neither",
        "nor",
        "cannot",
        "without",
        "hardly",
        "barely",
        "scarcely",
    }
)


def _clean_tokens(words: Sequence[str]) -> list[str]:
    """Tokenize stopwords matching TfidfVectorizer regex to avoid warnings."""
    cleaned: set[str] = set()
    for w in words:
        for t in re.findall(r"\b[a-z]{2,}\b", str(w).lower()):
            if t not in NEGATION_TOKENS:
                cleaned.add(t)
    return sorted(cleaned)


# Sentiment stopwords: Default stopwords minus negation tokens
SENTIMENT_STOPWORDS = _clean_tokens(DEFAULT_STOPWORDS - NEGATION_TOKENS)


# ---------------------------------------------------------------------------
# Majority-class baseline model
# ---------------------------------------------------------------------------


class MajorityBaselineModel:
    """
    Baseline model that always predicts the most frequent class from training data.

    Serves as an empirical benchmark to assess whether machine learning models
    provide meaningful predictive lift over trivial guessing.
    """

    def __init__(self) -> None:
        self.model = DummyClassifier(strategy="most_frequent")
        self.classes_: list[str] = []
        self.majority_class_: Optional[str] = None
        self.model_name = "majority_class_baseline"
        self.training_metadata: dict[str, Any] = {}

    def fit(self, X: Sequence[str], y: Sequence[str]) -> MajorityBaselineModel:
        """Fit the baseline model on training labels."""
        y_arr = np.asarray(y)
        self.model.fit(X, y_arr)
        self.classes_ = sorted(list(self.model.classes_))
        counts = pd.Series(y_arr).value_counts()
        self.majority_class_ = str(counts.index[0])
        self.training_metadata = {
            "model_name": self.model_name,
            "trained_at": datetime.now().isoformat(),
            "n_samples": len(y_arr),
            "classes": self.classes_,
            "majority_class": self.majority_class_,
            "class_distribution": {str(k): int(v) for k, v in counts.items()},
        }
        return self

    def predict(self, X: Sequence[str]) -> list[Optional[str]]:
        """Predict class labels for given texts. Empty inputs return None."""
        clean_texts = [normalize_text(t) for t in X]
        results: list[Optional[str]] = [None] * len(clean_texts)
        valid_indices = [i for i, t in enumerate(clean_texts) if t]
        if valid_indices:
            valid_texts = [clean_texts[i] for i in valid_indices]
            preds = self.model.predict(valid_texts)
            for idx, p in zip(valid_indices, preds):
                results[idx] = str(p)
        return results

    def predict_proba(self, X: Sequence[str]) -> np.ndarray:
        """Predict class probability matrix. Empty inputs return all zeros."""
        clean_texts = [normalize_text(t) for t in X]
        n_classes = len(self.classes_)
        results = np.zeros((len(clean_texts), n_classes), dtype=float)
        valid_indices = [i for i, t in enumerate(clean_texts) if t]
        if valid_indices:
            valid_texts = [clean_texts[i] for i in valid_indices]
            probs = self.model.predict_proba(valid_texts)
            for idx, p in zip(valid_indices, probs):
                results[idx] = p
        return results


# ---------------------------------------------------------------------------
# TF-IDF + Logistic Regression Sentiment Model
# ---------------------------------------------------------------------------


class TfidfLogisticRegressionSentiment:
    """
    Supervised sentiment classifier using TF-IDF and Logistic Regression.

    Key Design Principles:
    - Negation-preserving stopwords to retain critical negative sentiment cues.
    - Vectorizer fitted strictly on training data to prevent test set leakage.
    - Balanced class weighting to handle customer review class imbalance.
    - Explicit prediction uncertainty scoring via probability margins.
    """

    def __init__(
        self,
        *,
        ngram_range: tuple[int, int] = (1, 2),
        max_features: int = 5000,
        sublinear_tf: bool = True,
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
    ) -> None:
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state

        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            sublinear_tf=self.sublinear_tf,
            stop_words=SENTIMENT_STOPWORDS,
            token_pattern=r"(?u)\b\w+\b",
        )
        self.classifier = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state,
            class_weight="balanced",
            solver="lbfgs",
        )
        self.classes_: list[str] = []
        self.model_name = "tfidf_logistic_regression"
        self.is_fitted = False
        self.training_metadata: dict[str, Any] = {}

    def fit(
        self, X_train: Sequence[str], y_train: Sequence[str]
    ) -> TfidfLogisticRegressionSentiment:
        """
        Fit vectorizer and classifier strictly on training texts and labels.

        Parameters
        ----------
        X_train : Sequence[str]
            Training review texts.
        y_train : Sequence[str]
            Target sentiment labels.
        """
        clean_texts = [normalize_text(t) for t in X_train]
        y_arr = np.asarray(y_train)

        # Fit TF-IDF only on training data
        X_tfidf = self.vectorizer.fit_transform(clean_texts)
        self.classifier.fit(X_tfidf, y_arr)

        self.classes_ = sorted(list(self.classifier.classes_))
        self.is_fitted = True

        counts = pd.Series(y_arr).value_counts().to_dict()
        self.training_metadata = {
            "model_name": self.model_name,
            "trained_at": datetime.now().isoformat(),
            "n_samples": len(y_arr),
            "classes": self.classes_,
            "class_distribution": {str(k): int(v) for k, v in counts.items()},
            "vocabulary_size": len(self.vectorizer.vocabulary_),
            "hyperparameters": {
                "ngram_range": self.ngram_range,
                "max_features": self.max_features,
                "sublinear_tf": self.sublinear_tf,
                "C": self.C,
                "max_iter": self.max_iter,
                "random_state": self.random_state,
            },
        }
        return self

    def predict(self, X: Sequence[str]) -> list[Optional[str]]:
        """
        Predict sentiment labels for a sequence of texts.

        Empty or whitespace-only inputs return None rather than emitting
        a fabricated class label.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Call fit() first.")
        clean_texts = [normalize_text(t) for t in X]
        results: list[Optional[str]] = [None] * len(clean_texts)
        valid_indices = [i for i, t in enumerate(clean_texts) if t]

        if valid_indices:
            valid_texts = [clean_texts[i] for i in valid_indices]
            X_tfidf = self.vectorizer.transform(valid_texts)
            preds = self.classifier.predict(X_tfidf)
            for idx, p in zip(valid_indices, preds):
                results[idx] = str(p)

        return results

    def predict_proba(self, X: Sequence[str]) -> np.ndarray:
        """
        Predict class probability distribution.

        Empty or whitespace-only inputs return all-zero probability vectors
        to avoid falsely implying a genuine model prediction occurred.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Call fit() first.")
        clean_texts = [normalize_text(t) for t in X]
        n_classes = len(self.classes_)
        results = np.zeros((len(clean_texts), n_classes), dtype=float)
        valid_indices = [i for i, t in enumerate(clean_texts) if t]

        if valid_indices:
            valid_texts = [clean_texts[i] for i in valid_indices]
            X_tfidf = self.vectorizer.transform(valid_texts)
            probs = self.classifier.predict_proba(X_tfidf)
            for idx, p in zip(valid_indices, probs):
                results[idx] = p

        return results

    def predict_single(
        self,
        text: Any,
        record_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Perform inference on a single customer review with uncertainty metrics.

        Parameters
        ----------
        text : Any
            The review text string.
        record_id : Optional[str]
            Optional feedback record identifier for traceability.

        Returns
        -------
        dict[str, Any]
            Structured prediction output including confidence, class probabilities,
            and uncertainty margin. Empty or whitespace-only inputs return a
            non-prediction status with predicted_label=None.
        """
        clean_t = normalize_text(text)

        # Handle empty/missing/whitespace-only text gracefully
        if not clean_t:
            return {
                "record_id": record_id,
                "text": "",
                "predicted_label": None,
                "confidence": 0.0,
                "probabilities": {c: 0.0 for c in self.classes_},
                "margin": 0.0,
                "is_uncertain": True,
                "is_valid_input": False,
                "status": "empty_input",
                "model_name": self.model_name,
            }

        probs = self.predict_proba([clean_t])[0]
        prob_dict = {
            self.classes_[i]: round(float(probs[i]), 4)
            for i in range(len(self.classes_))
        }

        sorted_probs = sorted(probs, reverse=True)
        top1_prob = float(sorted_probs[0])
        top2_prob = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
        margin = round(top1_prob - top2_prob, 4)

        top_class = self.classes_[int(np.argmax(probs))]
        is_uncertain = bool(margin < 0.15 or top1_prob < 0.50)

        return {
            "record_id": record_id,
            "text": clean_t,
            "predicted_label": top_class,
            "confidence": round(top1_prob, 4),
            "probabilities": prob_dict,
            "margin": margin,
            "is_uncertain": is_uncertain,
            "is_valid_input": True,
            "status": "predicted",
            "model_name": self.model_name,
        }

    def predict_batch(
        self,
        df: pd.DataFrame,
        text_column: str = "review_text",
        id_column: str = "review_id",
    ) -> pd.DataFrame:
        """
        Perform vectorized batch prediction across a DataFrame.

        Appends predicted_sentiment, sentiment_confidence, sentiment_margin,
        sentiment_probabilities, is_uncertain, and is_valid_input columns.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Call fit() first.")

        out_df = df.copy()
        texts = out_df[text_column].fillna("").astype(str)
        clean_texts = [normalize_text(t) for t in texts]

        preds = self.predict(clean_texts)
        probs = self.predict_proba(clean_texts)

        confs: list[float] = []
        margins: list[float] = []
        uncertains: list[bool] = []
        prob_dicts: list[dict[str, float]] = []

        for clean_t, p_row in zip(clean_texts, probs):
            p_dict = {
                self.classes_[i]: round(float(p_row[i]), 4)
                for i in range(len(self.classes_))
            }
            prob_dicts.append(p_dict)
            if not clean_t:
                confs.append(0.0)
                margins.append(0.0)
                uncertains.append(True)
            else:
                sorted_p = sorted(p_row, reverse=True)
                t1 = float(sorted_p[0])
                t2 = float(sorted_p[1]) if len(sorted_p) > 1 else 0.0
                m = round(t1 - t2, 4)
                confs.append(round(t1, 4))
                margins.append(m)
                uncertains.append(bool(m < 0.15 or t1 < 0.50))

        out_df["predicted_sentiment"] = preds
        out_df["sentiment_confidence"] = confs
        out_df["sentiment_margin"] = margins
        out_df["sentiment_probabilities"] = prob_dicts
        out_df["is_uncertain"] = uncertains
        out_df["is_valid_input"] = [bool(t) for t in clean_texts]

        return out_df


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------


def evaluate_sentiment_model(
    model: Union[MajorityBaselineModel, TfidfLogisticRegressionSentiment],
    X_test: Sequence[str],
    y_test: Sequence[str],
    model_name: Optional[str] = None,
) -> dict[str, Any]:
    """
    Evaluate a sentiment classification model across multi-class performance metrics.

    Parameters
    ----------
    model : MajorityBaselineModel or TfidfLogisticRegressionSentiment
        The fitted classifier.
    X_test : Sequence[str]
        Held-out evaluation texts.
    y_test : Sequence[str]
        True sentiment labels.
    model_name : Optional[str]
        Descriptive model identifier.

    Returns
    -------
    dict[str, Any]
        Structured evaluation metrics including accuracy, class-level metrics,
        macro-F1, weighted-F1, confusion matrix, and sample distributions.
    """
    y_true = [str(y) for y in y_test]
    y_pred = model.predict(X_test)
    classes = model.classes_

    # Zero-division safe classification report
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=classes,
        output_dict=True,
        zero_division=0,
    )

    acc = float(report_dict.get("accuracy", 0.0))
    macro_f1 = float(
        f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)
    )
    weighted_f1 = float(
        f1_score(y_true, y_pred, labels=classes, average="weighted", zero_division=0)
    )

    cm = confusion_matrix(y_true, y_pred, labels=classes)

    precision_by_class: dict[str, float] = {}
    recall_by_class: dict[str, float] = {}
    f1_by_class: dict[str, float] = {}

    for c in classes:
        if c in report_dict:
            precision_by_class[c] = round(float(report_dict[c]["precision"]), 4)
            recall_by_class[c] = round(float(report_dict[c]["recall"]), 4)
            f1_by_class[c] = round(float(report_dict[c]["f1-score"]), 4)
        else:
            precision_by_class[c] = 0.0
            recall_by_class[c] = 0.0
            f1_by_class[c] = 0.0

    class_counts = pd.Series(y_true).value_counts().to_dict()

    return {
        "model_name": model_name or getattr(model, "model_name", "sentiment_model"),
        "evaluated_at": datetime.now().isoformat(),
        "n_samples": len(y_true),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "precision_by_class": precision_by_class,
        "recall_by_class": recall_by_class,
        "f1_by_class": f1_by_class,
        "confusion_matrix": {
            "labels": classes,
            "matrix": cm.tolist(),
        },
        "classification_report": report_dict,
        "class_distribution": {str(k): int(v) for k, v in class_counts.items()},
        "notes": (
            "Evaluation on held-out split. Weak labels derived from ratings "
            "do not represent independent human-verified accuracy."
        ),
    }


# ---------------------------------------------------------------------------
# Model persistence and loading
# ---------------------------------------------------------------------------


def save_sentiment_model(
    model: TfidfLogisticRegressionSentiment,
    output_dir: Union[str, Path] = "models",
    filename: Optional[str] = None,
) -> str:
    """
    Persist fitted sentiment model and vectorizer to disk using joblib.

    Parameters
    ----------
    model : TfidfLogisticRegressionSentiment
        The fitted sentiment classifier.
    output_dir : Union[str, Path], default "models"
        Directory to store the serialized model.
    filename : Optional[str]
        Custom filename override.

    Returns
    -------
    str
        Full path to the saved model file.
    """
    os.makedirs(output_dir, exist_ok=True)
    if filename is None:
        filename = "sentiment_tfidf_logistic_regression.joblib"

    filepath = os.path.join(output_dir, filename)
    payload = {
        "model_name": model.model_name,
        "vectorizer": model.vectorizer,
        "classifier": model.classifier,
        "classes": model.classes_,
        "hyperparameters": {
            "ngram_range": model.ngram_range,
            "max_features": model.max_features,
            "sublinear_tf": model.sublinear_tf,
            "C": model.C,
            "max_iter": model.max_iter,
            "random_state": model.random_state,
        },
        "training_metadata": model.training_metadata,
        "saved_at": datetime.now().isoformat(),
    }
    joblib.dump(payload, filepath)
    logger.info("Saved sentiment model artifact to: %s", filepath)
    return filepath


def load_sentiment_model(
    filepath: Union[str, Path],
    expected_sha256: Optional[str] = None,
) -> TfidfLogisticRegressionSentiment:
    """
    Load a previously persisted sentiment model artifact for inference.

    SECURITY NOTICE:
    ----------------
    joblib/pickle deserialization executes arbitrary code during loading.
    Only trusted model artifacts created within MarketLens in the application's
    designated models directory should ever be loaded. Never load untrusted,
    third-party, or user-uploaded pickle files. MarketLens does not expose any
    interface for user model uploads.
    Post-deserialization schema/attribute validation verifies structure and
    compatibility; it DOES NOT protect against malicious bytecode payloads
    embedded in an untrusted artifact.
    If expected_sha256 is supplied, the file's SHA-256 digest is cryptographically
    verified before deserialization begins.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to the saved .joblib model file.
    expected_sha256 : Optional[str]
        Optional expected SHA-256 hexadecimal checksum.

    Returns
    -------
    TfidfLogisticRegressionSentiment
        Initialized and fitted model instance.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Sentiment model file not found at: {filepath}")

    if expected_sha256 is not None:
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash.lower() != expected_sha256.lower().strip():
            raise ValueError(
                f"Model artifact SHA-256 mismatch for {filepath}. "
                f"Expected {expected_sha256}, got {actual_hash}."
            )

    try:
        payload = joblib.load(path)
        if not isinstance(payload, dict) or "hyperparameters" not in payload:
            raise ValueError(
                f"Incompatible or corrupted sentiment model artifact at: {filepath}"
            )
        model = TfidfLogisticRegressionSentiment(
            ngram_range=payload["hyperparameters"]["ngram_range"],
            max_features=payload["hyperparameters"]["max_features"],
            sublinear_tf=payload["hyperparameters"]["sublinear_tf"],
            C=payload["hyperparameters"]["C"],
            max_iter=payload["hyperparameters"]["max_iter"],
            random_state=payload["hyperparameters"]["random_state"],
        )
        model.vectorizer = payload["vectorizer"]
        model.classifier = payload["classifier"]
        model.classes_ = payload["classes"]
        model.training_metadata = payload.get("training_metadata", {})
        model.is_fitted = True
        return model
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ValueError) and (
            "Incompatible or corrupted" in str(exc) or "SHA-256 mismatch" in str(exc)
        ):
            raise
        raise ValueError(
            f"Incompatible or corrupted sentiment model artifact at {filepath}: {exc}"
        ) from exc
