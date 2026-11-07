"""
MarketLens Dashboard - Sentiment Analysis Page.

Integrates NLPManager and sentiment classification models. Faithfully presents
binary vs. three-class architectures, confidence metrics, and baseline comparisons
without fabricating neutral predictions on binary classifiers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from marketlens.dashboard.components import (
    paginate_dataframe,
    render_bar_chart,
    render_empty_state,
    render_kpi_card,
)
from marketlens.dashboard.state import (
    KEY_NLP_MANAGER,
    KEY_NLP_RESULTS,
    get_active_dataframe,
    load_demo_dataset,
)
from marketlens.nlp.unified import NLPManager


def _get_or_create_nlp_manager() -> NLPManager:
    """Retrieve existing NLPManager from session state or instantiate a new one."""
    manager: Optional[NLPManager] = st.session_state.get(KEY_NLP_MANAGER)
    if manager is None:
        manager = NLPManager()
        st.session_state[KEY_NLP_MANAGER] = manager
    return manager


def render_sentiment_page() -> None:
    """Render the Sentiment Analysis page."""
    st.title("🎭 Sentiment Intelligence & Modeling")
    st.markdown(
        "Supervised sentiment classification, prediction uncertainty inspection, "
        "and comparative baseline evaluation."
    )

    df = get_active_dataframe()
    if df is None or df.empty:
        render_empty_state(
            title="No Dataset Loaded",
            message=(
                "Please upload a feedback dataset or load the demo "
                "dataset to analyze sentiment."
            ),
            icon="🎭",
            action_button_label="Load Demo Dataset",
            on_action=lambda: (load_demo_dataset(), st.rerun()),
        )
        return

    manager = _get_or_create_nlp_manager()
    nlp_results = st.session_state.get(KEY_NLP_RESULTS)

    # 1. Model Lifecycle Management Bar
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 2])

    with col_ctrl1:
        if st.button(
            "🚀 Train Sentiment Model", type="primary", use_container_width=True
        ):
            if "review_text" not in df.columns or "rating" not in df.columns:
                st.error("Dataset must contain 'review_text' and 'rating' columns.")
            else:
                with st.spinner(
                    "Training TF-IDF Logistic Regression & baseline models..."
                ):
                    try:
                        results = manager.train_and_evaluate(
                            df,
                            text_column="review_text",
                            rating_column="rating",
                        )
                        st.session_state[KEY_NLP_RESULTS] = results
                        st.session_state[KEY_NLP_MANAGER] = manager
                        st.success("Model trained successfully!")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Training failed: {exc}")

    with col_ctrl2:
        saved_models_exist = Path(
            "models/sentiment_tfidf_logistic_regression.joblib"
        ).exists()
        if st.button(
            "📂 Load Saved Model",
            disabled=not saved_models_exist,
            use_container_width=True,
            help="Load pre-trained model from disk without retraining",
        ):
            try:
                manager.load_models()
                st.session_state[KEY_NLP_MANAGER] = manager
                st.success("Loaded model from models/ directory!")
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to load model: {exc}")

    with col_ctrl3:
        if manager.sentiment_model is not None:
            if st.button("💾 Persist Model to Disk", use_container_width=True):
                try:
                    paths = manager.save_models()
                    st.success(
                        f"Model saved to {paths.get('sentiment_model', 'models/')}"
                    )
                except Exception as exc:
                    st.error(f"Failed to save model: {exc}")

    st.divider()

    # 2. Check model readiness
    model = manager.sentiment_model
    if model is None or not model.is_fitted:
        st.info(
            "No active sentiment model is fitted. Click **Train Sentiment Model** "
            "above to train on the active dataset or load a pre-trained model."
        )
        # Display rating-derived weak labels if available in the dataset
        if "rating" in df.columns:
            st.subheader("📊 Rating-Derived Weak Sentiment Labels")
            st.caption(
                "Notice: These are heuristic labels mapped directly from "
                "ratings (1-2: neg, 3: neu, 4-5: pos), not ML predictions."
            )
            from marketlens.nlp.dataset import rating_to_sentiment

            weak_labels = [rating_to_sentiment(r) for r in df["rating"].dropna()]
            weak_counts = pd.Series(weak_labels).value_counts().to_dict()
            plot_df = pd.DataFrame(
                [{"Class": k.title(), "Count": v} for k, v in weak_counts.items()]
            )
            render_bar_chart(
                plot_df,
                x_col="Class",
                y_col="Count",
                title="Rating Heuristic Distribution",
            )
        return

    # 3. Model Architecture & Trained Classes Metadata
    trained_classes = list(model.classes_)
    is_binary = len(trained_classes) == 2

    st.subheader("⚙️ Active Model Specifications")
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        render_kpi_card("Model Architecture", "TF-IDF + LogReg", model.model_name)
    with mc2:
        class_type = (
            "Binary (2-class)" if is_binary else f"Multi-class ({len(trained_classes)})"
        )
        render_kpi_card("Classification Scope", class_type, ", ".join(trained_classes))
    with mc3:
        n_features = (
            len(model.vectorizer.get_feature_names_out())
            if hasattr(model, "vectorizer")
            else "N/A"
        )
        render_kpi_card("Vocabulary Size", n_features, "Max n-gram features")
    with mc4:
        c_param = getattr(model, "C", 1.0)
        render_kpi_card("Regularization C", c_param, "L2 Inverse Penalty")

    # Epistemological Integrity Callout
    if is_binary:
        st.warning(
            "🛡️ **Binary Model Integrity Guard:** Trained on binary classes "
            f"(`{trained_classes}`). Neutral classifications are never "
            "fabricated or emitted."
        )

    st.divider()

    # 4. Evaluation Performance vs. Baseline
    if nlp_results and "logistic_regression_evaluation" in nlp_results:
        st.subheader("📈 Benchmark vs. Majority Baseline")
        lr_eval = nlp_results["logistic_regression_evaluation"]
        base_eval = nlp_results.get("baseline_evaluation", {})

        ec1, ec2, ec3, ec4 = st.columns(4)
        with ec1:
            render_kpi_card("Model Accuracy", f"{lr_eval.get('accuracy', 0.0):.1%}")
        with ec2:
            render_kpi_card("Model Macro-F1", f"{lr_eval.get('macro_f1', 0.0):.3f}")
        with ec3:
            render_kpi_card(
                "Baseline Macro-F1",
                f"{base_eval.get('macro_f1', 0.0):.3f}" if base_eval else "N/A",
                "Majority class baseline",
            )
        with ec4:
            delta_f1 = (
                lr_eval.get("macro_f1", 0.0) - base_eval.get("macro_f1", 0.0)
                if base_eval
                else None
            )
            render_kpi_card(
                "F1 Uplift",
                f"{delta_f1:+.3f}" if delta_f1 is not None else "N/A",
                delta=f"{delta_f1:+.3f}" if delta_f1 is not None else None,
            )

        # Per-class metrics
        class_metrics = lr_eval.get("class_metrics", {})
        if class_metrics:
            cm_rows = []
            for cls_name, metrics in class_metrics.items():
                cm_rows.append(
                    {
                        "Class": cls_name,
                        "Precision": f"{metrics.get('precision', 0.0):.3f}",
                        "Recall": f"{metrics.get('recall', 0.0):.3f}",
                        "F1-Score": f"{metrics.get('f1', 0.0):.3f}",
                        "Support": metrics.get("support", 0),
                    }
                )
            st.dataframe(pd.DataFrame(cm_rows), use_container_width=True)

        st.divider()

    # 5. Batch Predictions on Active Dataset
    st.subheader("🔍 Dataset Batch Predictions & Uncertainty")
    if "review_text" in df.columns:
        scored_df = model.predict_batch(
            df, text_column="review_text", id_column="review_id"
        )

        # Uncertainty metric
        uncertain_count = int(scored_df["is_uncertain"].sum())
        uncertain_pct = (
            (uncertain_count / len(scored_df)) * 100 if len(scored_df) > 0 else 0.0
        )

        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            render_kpi_card("Total Scored", len(scored_df))
        with sc2:
            render_kpi_card(
                "Mean Confidence", f"{scored_df['sentiment_confidence'].mean():.1%}"
            )
        with sc3:
            render_kpi_card(
                "Uncertain Predictions",
                f"{uncertain_count} ({uncertain_pct:.1f}%)",
                "Margin < 0.15 or confidence < 50%",
            )

        # Class breakdown chart
        valid_preds = scored_df["predicted_sentiment"].dropna()
        if not valid_preds.empty:
            pred_counts = valid_preds.value_counts().to_dict()
            p_chart_df = pd.DataFrame(
                [
                    {"Sentiment": str(k).title(), "Count": v}
                    for k, v in pred_counts.items()
                ]
            )
            render_bar_chart(
                p_chart_df,
                x_col="Sentiment",
                y_col="Count",
                title="Predicted Sentiment Distribution",
            )

        # Table of predictions
        preview_cols = [
            c
            for c in [
                "review_id",
                "review_text",
                "rating",
                "predicted_sentiment",
                "sentiment_confidence",
                "sentiment_margin",
                "is_uncertain",
            ]
            if c in scored_df.columns
        ]
        slice_df, _, _ = paginate_dataframe(
            scored_df[preview_cols], page_size=10, key_prefix="sent_prev"
        )
        st.dataframe(slice_df, use_container_width=True)

    st.divider()

    # 6. Interactive Single Review Inference
    st.subheader("🧪 Interactive Sentiment Inference")
    st.caption(
        "Test individual customer review strings in real time against "
        "the active classifier."
    )

    sample_input = st.text_area(
        "Enter customer review text:",
        value="",
        placeholder="e.g. Great product, but customer support was unhelpful.",
        key="single_review_input",
    )

    if st.button("Predict Sentiment", type="secondary"):
        pred_res = model.predict_single(sample_input)

        if not pred_res.get("is_valid_input", True):
            st.warning(
                "⚠️ **Empty / Whitespace Input:** Non-prediction emitted. "
                "Empty text is never assigned a fabricated sentiment label."
            )
        else:
            pred_label = pred_res.get("predicted_label")
            conf = pred_res.get("confidence", 0.0)
            margin = pred_res.get("margin", 0.0)
            is_unc = pred_res.get("is_uncertain", False)

            ic1, ic2, ic3 = st.columns(3)
            with ic1:
                render_kpi_card("Predicted Sentiment", str(pred_label).upper())
            with ic2:
                render_kpi_card("Confidence", f"{conf:.1%}")
            with ic3:
                render_kpi_card(
                    "Prediction Margin",
                    f"{margin:.4f}",
                    "Low confidence warning" if is_unc else "Decisive prediction",
                )

            # Probabilities distribution
            probs = pred_res.get("probabilities", {})
            if probs:
                prob_df = pd.DataFrame(
                    [{"Class": k.title(), "Probability": v} for k, v in probs.items()]
                )
                render_bar_chart(
                    prob_df,
                    x_col="Class",
                    y_col="Probability",
                    title="Model Class Probabilities",
                )
