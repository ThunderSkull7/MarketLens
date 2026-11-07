"""
MarketLens - Intelligence Engine.

Orchestrates context preparation, prompt dispatch, bounded retries,
safe JSON response parsing, evidence validation, cost estimation,
and report generation.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime
from typing import Any, Optional

import pandas as pd
from pydantic import ValidationError

from marketlens.config import MarketLensSettings, load_settings
from marketlens.exceptions import (
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from marketlens.intelligence.context import prepare_intelligence_context
from marketlens.intelligence.provider import (
    LLMProvider,
    LLMResponse,
    create_llm_provider,
    estimate_token_cost,
)
from marketlens.intelligence.schemas import (
    ComparisonInsight,
    ExecutiveSummary,
    IntelligenceReport,
    LLMExecutionMetadata,
    MarketingRecommendation,
    SentimentInsight,
    TopicInsight,
)
from marketlens.intelligence.validator import sanitize_and_validate_evidence

logger = logging.getLogger("marketlens.intelligence.engine")

_SYSTEM_PROMPT = """You are MarketLens Intelligence Engine, an expert marketing analyst.
Analyze customer feedback analytics, sentiment models, and topic clusters to produce
actionable, evidence-grounded marketing intelligence in valid JSON.

CRITICAL EVIDENCE RULES:
1. Every observation or recommendation MUST cite valid evidence IDs from context.
2. DO NOT INVENT OR FABRICATE EVIDENCE IDs under any circumstances.
3. Every recommendation MUST include a non-empty 'supporting_observation'.
4. Do NOT infer competitor market share from internal feedback volumes.
5. Distinguish classical ML predictions and weak labels from human ground truth.
6. Output ONLY a valid JSON object matching the requested schema without extra prose.
"""


def _extract_json_payload(raw_text: str) -> dict[str, Any]:
    """Extract and parse JSON from raw LLM output, stripping markdown code blocks."""
    text = raw_text.strip()
    if text.startswith("```"):
        # Match ```json ... ``` or ``` ... ```
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            text = match.group(1).strip()
    return json.loads(text)


class IntelligenceEngine:
    """
    Main intelligence service coordinating LLM inference and evidence grounding.

    Can be executed completely independently of the ingestion and cleaning pipelines.
    Gracefully handles missing providers without interrupting platform operations.
    """

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        settings: Optional[MarketLensSettings] = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.provider = provider or create_llm_provider(self.settings)

    def is_available(self) -> bool:
        """Check if an active, configured LLM provider is available."""
        return self.provider is not None and self.provider.is_available()

    def generate_report(
        self,
        df: pd.DataFrame,
        *,
        analytics_summary: Optional[dict[str, Any]] = None,
        nlp_results: Optional[dict[str, Any]] = None,
        max_reviews: Optional[int] = None,
        max_review_chars: Optional[int] = None,
        id_column: str = "review_id",
        text_column: str = "review_text",
        rating_column: str = "rating",
        sentiment_column: str = "sentiment",
        product_column: str = "product_id",
    ) -> IntelligenceReport:
        """
        Generate a structured, evidence-grounded intelligence report.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned customer reviews DataFrame.
        analytics_summary : Optional[dict[str, Any]]
            Output of analytics (overview, rating_analysis, temporal_analysis).
        nlp_results : Optional[dict[str, Any]]
            Output of NLPManager (topic_clusters, baseline/logistic eval).
        max_reviews : Optional[int]
            Override for maximum reviews sent in context.
        max_review_chars : Optional[int]
            Override for character limit per review excerpt.
        id_column : str
            Review ID column name.
        text_column : str
            Review text column name.
        rating_column : str
            Rating column name.
        sentiment_column : str
            Sentiment label column name.
        product_column : str
            Product/service column name.

        Returns
        -------
        IntelligenceReport
            Structured report with execution status, insights, and recommendations.
        """
        now_iso = datetime.now().isoformat()
        dataset_scope = {
            "total_records": len(df),
            "columns": list(df.columns) if not df.empty else [],
        }

        # 1. Check provider availability
        if not self.is_available():
            logger.info("LLM provider unavailable; returning structured report.")
            return IntelligenceReport(
                generated_at=now_iso,
                status="unavailable",
                dataset_scope=dataset_scope,
                execution_metadata=LLMExecutionMetadata(
                    provider=self.provider.get_provider_name()
                    if self.provider
                    else None,
                    model=self.provider.get_model_name() if self.provider else None,
                    executed_at=now_iso,
                    status="unavailable",
                    error_message="LLM provider is not configured or unavailable",
                ),
            )

        # 2. Check for empty dataset
        if df.empty:
            logger.warning(
                "Empty dataset passed to IntelligenceEngine; cannot generate report."
            )
            return IntelligenceReport(
                generated_at=now_iso,
                status="failed",
                dataset_scope=dataset_scope,
                execution_metadata=LLMExecutionMetadata(
                    provider=self.provider.get_provider_name(),
                    model=self.provider.get_model_name(),
                    executed_at=now_iso,
                    status="failed",
                    error_message=(
                        "Input dataset is empty; insufficient evidence "
                        "to generate intelligence report."
                    ),
                ),
            )

        # 3. Context Preparation & Data Minimization
        limit_revs = max_reviews or self.settings.llm_max_context_reviews
        limit_chars = max_review_chars or self.settings.llm_max_review_chars

        context = prepare_intelligence_context(
            df,
            analytics_summary=analytics_summary,
            nlp_results=nlp_results,
            max_reviews=limit_revs,
            max_review_chars=limit_chars,
            id_column=id_column,
            text_column=text_column,
            rating_column=rating_column,
            sentiment_column=sentiment_column,
            product_column=product_column,
        )

        user_prompt = f"""
{context.prompt_context_text}

### Instructions:
Generate an evidence-grounded marketing report as a single JSON object with keys:
- "executive_summary": overview, positive_signals, negative_signals,
  recurring_issues, business_interpretation, evidence_references
- "sentiment_intelligence": dominant_sentiment, positive_themes, negative_themes,
  distribution_summary, model_limitations, evidence_references
- "topic_insights": list of objects each having [cluster_id, topic_label,
  interpretation, customer_concerns, business_relevance, is_emerging,
  temporal_evidence, evidence_references]
- "comparison_insights": list of objects each having [entity_type,
  comparison_name, findings, sample_size_adequate, limitations, evidence_references]
- "recommendations": list of objects each having [title, description,
  supporting_observation, evidence_ids, impact_category, priority, confidence,
  suggested_action]

Allowed Evidence IDs you can cite:
{json.dumps(context.allowed_evidence_ids)}
"""

        # 4. Dispatch with Bounded Retries & Backoff
        max_retries = max(0, self.settings.llm_max_retries)
        backoff_factor = max(1.0, self.settings.llm_retry_backoff_factor)

        response: Optional[LLMResponse] = None
        last_error: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            try:
                response = self.provider.generate(
                    prompt=user_prompt,
                    system_prompt=_SYSTEM_PROMPT,
                    temperature=self.settings.llm_temperature,
                    max_tokens=self.settings.llm_max_tokens,
                )
                break
            except LLMAuthenticationError as auth_err:
                logger.error("LLM authentication failure: %s", auth_err)
                last_error = auth_err
                break  # Do not retry credentials failure
            except (LLMTimeoutError, LLMRateLimitError, Exception) as transient_err:
                last_error = transient_err
                logger.warning(
                    "LLM call attempt %d/%d failed: %s",
                    attempt + 1,
                    max_retries + 1,
                    transient_err,
                )
                if attempt < max_retries:
                    sleep_time = min(5.0, backoff_factor * (attempt + 1))
                    time.sleep(sleep_time)

        if response is None:
            err_msg = str(last_error) if last_error else "Provider returned no response"
            return IntelligenceReport(
                generated_at=now_iso,
                status="failed",
                dataset_scope=dataset_scope,
                allowed_evidence_ids=context.allowed_evidence_ids,
                execution_metadata=LLMExecutionMetadata(
                    provider=self.provider.get_provider_name(),
                    model=self.provider.get_model_name(),
                    executed_at=now_iso,
                    status="failed",
                    error_message=err_msg,
                ),
            )

        # 5. Parse JSON Output
        try:
            raw_dict = _extract_json_payload(response.content)
        except Exception as json_err:
            logger.error("Failed to parse JSON response from LLM: %s", json_err)
            return IntelligenceReport(
                generated_at=now_iso,
                status="failed",
                dataset_scope=dataset_scope,
                allowed_evidence_ids=context.allowed_evidence_ids,
                execution_metadata=LLMExecutionMetadata(
                    provider=self.provider.get_provider_name(),
                    model=self.provider.get_model_name(),
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    total_tokens=response.total_tokens,
                    latency_ms=response.latency_ms,
                    executed_at=now_iso,
                    status="validation_error",
                    error_message=f"Malformed JSON response: {json_err}",
                ),
            )

        # 6. Sanitize and Validate Evidence
        sanitized_dict, val_summary = sanitize_and_validate_evidence(
            raw_dict,
            allowed_evidence_ids=context.allowed_evidence_ids,
            review_evidence_map=context.review_evidence_map,
            metric_evidence_map=context.metric_evidence_map,
        )

        # 7. Construct & Validate Pydantic Models
        try:
            exec_summary = (
                ExecutiveSummary.model_validate(sanitized_dict["executive_summary"])
                if "executive_summary" in sanitized_dict
                else None
            )
            sent_intel = (
                SentimentInsight.model_validate(
                    sanitized_dict["sentiment_intelligence"]
                )
                if "sentiment_intelligence" in sanitized_dict
                else None
            )
            topics = [
                TopicInsight.model_validate(t)
                for t in sanitized_dict.get("topic_insights", [])
            ]
            comparisons = [
                ComparisonInsight.model_validate(c)
                for c in sanitized_dict.get("comparison_insights", [])
            ]
            recommendations = [
                MarketingRecommendation.model_validate(r)
                for r in sanitized_dict.get("recommendations", [])
            ]

        except ValidationError as val_err:
            logger.error("Schema validation failed on LLM output: %s", val_err)
            return IntelligenceReport(
                generated_at=now_iso,
                status="failed",
                dataset_scope=dataset_scope,
                allowed_evidence_ids=context.allowed_evidence_ids,
                execution_metadata=LLMExecutionMetadata(
                    provider=self.provider.get_provider_name(),
                    model=self.provider.get_model_name(),
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    total_tokens=response.total_tokens,
                    latency_ms=response.latency_ms,
                    executed_at=now_iso,
                    status="validation_error",
                    error_message=f"Pydantic schema validation error: {val_err}",
                ),
            )

        # 8. Calculate Cost Metadata
        cost_usd = estimate_token_cost(
            self.provider.get_model_name(),
            response.input_tokens,
            response.output_tokens,
        )

        metadata = LLMExecutionMetadata(
            provider=self.provider.get_provider_name(),
            model=self.provider.get_model_name(),
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            total_tokens=response.total_tokens,
            estimated_cost_usd=cost_usd,
            latency_ms=response.latency_ms,
            executed_at=now_iso,
            status="success",
        )

        return IntelligenceReport(
            generated_at=now_iso,
            status="completed",
            dataset_scope=dataset_scope,
            executive_summary=exec_summary,
            sentiment_intelligence=sent_intel,
            topic_insights=topics,
            comparison_insights=comparisons,
            recommendations=recommendations,
            allowed_evidence_ids=context.allowed_evidence_ids,
            execution_metadata=metadata,
        )
