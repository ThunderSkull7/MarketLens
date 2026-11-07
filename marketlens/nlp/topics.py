"""
MarketLens - Unsupervised Topic & Theme Discovery Module.

Implements:
1. TF-IDF + K-Means clustering for customer feedback theme discovery.
2. Cluster summarisation with top representative terms and centroid-proximate reviews.
3. Automated descriptive label generation.
4. Silhouette score computation with data sufficiency guards.
5. Persistent topic discovery artifacts and inference assignment interfaces.
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
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances_argmin_min, silhouette_score

from marketlens.analytics import DEFAULT_STOPWORDS
from marketlens.cleaning import normalize_text

logger = logging.getLogger("marketlens.nlp.topics")


def _clean_tokens(words: Sequence[str]) -> list[str]:
    """Tokenize stopwords matching TfidfVectorizer regex to avoid warnings."""
    cleaned: set[str] = set()
    for w in words:
        for t in re.findall(r"\b[a-z]{2,}\b", str(w).lower()):
            cleaned.add(t)
    return sorted(cleaned)


# Domain-specific terms that are generic across reviews
_RAW_TOPIC_STOPWORDS = DEFAULT_STOPWORDS.union(
    {
        "product",
        "item",
        "review",
        "bought",
        "buy",
        "ordered",
        "order",
        "good",
        "great",
        "bad",
        "okay",
        "ok",
        "really",
        "just",
        "like",
    }
)
TOPIC_STOPWORDS = _clean_tokens(_RAW_TOPIC_STOPWORDS)


class TopicDiscoveryModel:
    """
    Unsupervised theme discovery using TF-IDF vectorization and K-Means clustering.

    Important Scope & Integrity Principle:
    A topic cluster represents lexical and semantic similarity between reviews,
    NOT customer sentiment. A topic (e.g. "battery life") can contain both positive
    and negative feedback and must not be automatically equated to sentiment classes.
    """

    def __init__(
        self,
        *,
        n_clusters: int = 3,
        ngram_range: tuple[int, int] = (1, 2),
        max_features: int = 2000,
        random_state: int = 42,
    ) -> None:
        self.n_clusters = n_clusters
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.random_state = random_state

        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            stop_words=TOPIC_STOPWORDS,
            token_pattern=r"(?u)\b[a-zA-Z]{2,}\b",
        )
        self.kmeans: Optional[KMeans] = None
        self.is_fitted = False
        self.cluster_summaries: list[dict[str, Any]] = []
        self.silhouette_: Optional[float] = None
        self.training_metadata: dict[str, Any] = {}

    def fit(
        self,
        texts: Sequence[str],
        *,
        ratings: Optional[Sequence[Any]] = None,
        sentiments: Optional[Sequence[Any]] = None,
        top_terms_count: int = 8,
        num_representative: int = 3,
    ) -> TopicDiscoveryModel:
        """
        Fit the TF-IDF vectorizer and K-Means clustering on review texts.

        Parameters
        ----------
        texts : Sequence[str]
            Customer review texts.
        ratings : Optional[Sequence[Any]]
            Corresponding numerical ratings, if available.
        sentiments : Optional[Sequence[Any]]
            Corresponding sentiment labels, if available.
        top_terms_count : int, default 8
            Number of top keywords to extract per cluster centroid.
        num_representative : int, default 3
            Number of exemplar reviews closest to the centroid to extract.
        """
        raw_texts = list(texts)
        clean_texts: list[str] = []
        valid_indices: list[int] = []

        for idx, t in enumerate(raw_texts):
            norm = normalize_text(t)
            if norm:
                clean_texts.append(norm)
                valid_indices.append(idx)

        n_samples = len(clean_texts)
        if n_samples < 2:
            raise ValueError(
                f"Insufficient text records ({n_samples} found; minimum 2 required)."
            )

        # Adjust k if sample count is smaller than requested clusters
        effective_k = min(self.n_clusters, n_samples)
        if effective_k < self.n_clusters:
            logger.warning(
                "Requested n_clusters=%d exceeds sample count %d; reducing k to %d.",
                self.n_clusters,
                n_samples,
                effective_k,
            )

        # 1. TF-IDF Transformation
        X_tfidf = self.vectorizer.fit_transform(clean_texts)

        # 2. K-Means Clustering
        self.kmeans = KMeans(
            n_clusters=effective_k,
            random_state=self.random_state,
            n_init=10,
        )
        cluster_labels = self.kmeans.fit_predict(X_tfidf)
        self.is_fitted = True

        # 3. Silhouette Score (calculated only when statistically valid)
        if 2 <= effective_k < n_samples and n_samples >= 10:
            try:
                self.silhouette_ = round(
                    float(silhouette_score(X_tfidf, cluster_labels)), 4
                )
            except Exception as e:
                logger.warning("Could not compute silhouette score: %s", e)
                self.silhouette_ = None
        else:
            self.silhouette_ = None

        # 4. Extract cluster centroids and summaries
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        centroids = self.kmeans.cluster_centers_

        # Align ratings and sentiments if provided
        aligned_ratings = (
            [ratings[i] for i in valid_indices] if ratings is not None else None
        )
        aligned_sentiments = (
            [sentiments[i] for i in valid_indices] if sentiments is not None else None
        )

        summaries: list[dict[str, Any]] = []

        for cid in range(effective_k):
            mask = cluster_labels == cid
            count = int(mask.sum())
            pct = round((count / n_samples) * 100.0, 2)

            # Top terms from centroid weights
            if len(feature_names) > 0:
                top_indices = centroids[cid].argsort()[::-1][:top_terms_count]
                top_terms = [
                    str(feature_names[i]) for i in top_indices if centroids[cid][i] > 0
                ]
            else:
                top_terms = []

            # Descriptive label derived from top 3 terms
            label = ", ".join(top_terms[:3]) if top_terms else f"Cluster {cid}"

            # Representative records closest to centroid
            rep_records: list[str] = []
            if count > 0:
                cluster_tfidf = X_tfidf[mask]
                cluster_texts = [clean_texts[i] for i in range(n_samples) if mask[i]]
                center = centroids[cid].reshape(1, -1)

                closest_indices, _ = pairwise_distances_argmin_min(
                    center, cluster_tfidf
                )
                closest_idx = int(closest_indices[0])

                # Get up to num_representative distinct texts
                rep_records.append(cluster_texts[closest_idx])
                for t in cluster_texts:
                    if t not in rep_records and len(rep_records) < num_representative:
                        rep_records.append(t)

            # Optional rating statistics
            avg_rating: Optional[float] = None
            rating_dist: dict[str, int] = {}
            if aligned_ratings is not None and count > 0:
                raw_ratings = pd.Series(
                    [aligned_ratings[i] for i in range(n_samples) if mask[i]]
                )
                c_ratings = pd.to_numeric(raw_ratings, errors="coerce").dropna()
                if len(c_ratings) > 0:
                    avg_rating = round(float(c_ratings.mean()), 2)
                    rating_dist = {
                        str(k): int(v) for k, v in c_ratings.value_counts().items()
                    }

            # Optional sentiment distribution
            sentiment_dist: dict[str, int] = {}
            if aligned_sentiments is not None and count > 0:
                c_sents = [
                    str(aligned_sentiments[i]) for i in range(n_samples) if mask[i]
                ]
                sentiment_dist = {
                    str(k): int(v) for k, v in pd.Series(c_sents).value_counts().items()
                }

            summaries.append(
                {
                    "cluster_id": cid,
                    "label": label,
                    "record_count": count,
                    "record_percentage": pct,
                    "top_terms": top_terms,
                    "representative_records": rep_records,
                    "average_rating": avg_rating,
                    "rating_distribution": rating_dist,
                    "sentiment_distribution": sentiment_dist,
                }
            )

        self.cluster_summaries = summaries
        self.training_metadata = {
            "n_samples": n_samples,
            "n_clusters": effective_k,
            "silhouette_score": self.silhouette_,
            "vocabulary_size": len(feature_names),
            "fitted_at": datetime.now().isoformat(),
            "disclaimer": (
                "Clusters reflect lexical and textual similarity; they do not "
                "automatically represent sentiment categories or business complaint "
                "types."
            ),
        }
        return self

    def assign_clusters(self, texts: Sequence[str]) -> list[int]:
        """
        Assign new texts to fitted clusters without refitting.

        Parameters
        ----------
        texts : Sequence[str]
            Sequence of reviews to cluster.

        Returns
        -------
        list[int]
            Cluster IDs for each input text.
        """
        if not self.is_fitted or self.kmeans is None:
            raise ValueError("TopicDiscoveryModel is not fitted. Call fit() first.")

        clean_texts = [normalize_text(t) for t in texts]
        X_tfidf = self.vectorizer.transform(clean_texts)
        assignments = self.kmeans.predict(X_tfidf)
        return [int(a) for a in assignments]

    def get_cluster_summaries(self) -> list[dict[str, Any]]:
        """Return structured summaries for all discovered topic clusters."""
        return self.cluster_summaries


# ---------------------------------------------------------------------------
# Topic model persistence and loading
# ---------------------------------------------------------------------------


def save_topic_model(
    model: TopicDiscoveryModel,
    output_dir: Union[str, Path] = "models",
    filename: Optional[str] = None,
) -> str:
    """
    Persist fitted TopicDiscoveryModel to disk using joblib.

    Parameters
    ----------
    model : TopicDiscoveryModel
        The fitted topic discovery model.
    output_dir : Union[str, Path], default "models"
        Target directory for serialization.
    filename : Optional[str]
        Filename override.

    Returns
    -------
    str
        Path to the saved model file.
    """
    os.makedirs(output_dir, exist_ok=True)
    if filename is None:
        filename = "topic_discovery_kmeans.joblib"

    filepath = os.path.join(output_dir, filename)
    payload = {
        "n_clusters": model.n_clusters,
        "ngram_range": model.ngram_range,
        "max_features": model.max_features,
        "random_state": model.random_state,
        "vectorizer": model.vectorizer,
        "kmeans": model.kmeans,
        "cluster_summaries": model.cluster_summaries,
        "silhouette_score": model.silhouette_,
        "training_metadata": model.training_metadata,
        "saved_at": datetime.now().isoformat(),
    }
    joblib.dump(payload, filepath)
    logger.info("Saved topic discovery model artifact to: %s", filepath)
    return filepath


def load_topic_model(
    filepath: Union[str, Path],
    expected_sha256: Optional[str] = None,
) -> TopicDiscoveryModel:
    """
    Load a previously fitted TopicDiscoveryModel artifact.

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
        Path to the saved .joblib file.
    expected_sha256 : Optional[str]
        Optional expected SHA-256 hexadecimal checksum.

    Returns
    -------
    TopicDiscoveryModel
        Ready-to-use fitted topic model.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Topic model file not found at: {filepath}")

    if expected_sha256 is not None:
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash.lower() != expected_sha256.lower().strip():
            raise ValueError(
                f"Model artifact SHA-256 mismatch for {filepath}. "
                f"Expected {expected_sha256}, got {actual_hash}."
            )

    try:
        payload = joblib.load(path)
        if not isinstance(payload, dict) or "n_clusters" not in payload:
            raise ValueError(
                f"Incompatible or corrupted topic model artifact at: {filepath}"
            )
        model = TopicDiscoveryModel(
            n_clusters=payload["n_clusters"],
            ngram_range=payload["ngram_range"],
            max_features=payload["max_features"],
            random_state=payload["random_state"],
        )
        model.vectorizer = payload["vectorizer"]
        model.kmeans = payload["kmeans"]
        model.cluster_summaries = payload.get("cluster_summaries", [])
        model.silhouette_ = payload.get("silhouette_score")
        model.training_metadata = payload.get("training_metadata", {})
        model.is_fitted = True
        return model
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ValueError) and (
            "Incompatible or corrupted" in str(exc) or "SHA-256 mismatch" in str(exc)
        ):
            raise
        raise ValueError(
            f"Incompatible or corrupted topic model artifact at {filepath}: {exc}"
        ) from exc
