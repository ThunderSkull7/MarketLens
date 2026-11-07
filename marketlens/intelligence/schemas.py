"""
MarketLens - LLM Intelligence Layer Pydantic Schemas.

Defines validated schemas for executive summaries, sentiment intelligence,
topic interpretation, comparative insights, actionable recommendations,
evidence references, execution metadata, and complete intelligence reports.
"""

from __future__ import annotations

from typing import Any, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class EvidenceReference(BaseModel):
    """
    Reference linking an intelligence statement or recommendation to evidence.

    Distinguishes individual-review customer feedback from aggregate metrics.
    """

    model_config = ConfigDict(extra="ignore")

    evidence_id: str = Field(
        ...,
        description="Unique identifier of the evidence (review ID or metric key)",
    )
    evidence_type: Literal["review", "metric", "aggregate"] = Field(
        default="review",
        description="Type: 'review' for text, 'metric' or 'aggregate' for stats",
    )
    description: Optional[str] = Field(
        default=None,
        description="Brief description of what this evidence demonstrates",
    )
    source_field: Optional[str] = Field(
        default=None,
        description="Underlying field or KPI name (e.g. 'review_text')",
    )


class ExecutiveSummary(BaseModel):
    """Executive summary of customer feedback patterns and interpretation."""

    model_config = ConfigDict(extra="ignore")

    overview: str = Field(
        ...,
        description="Concise synthesis of overall customer feedback patterns",
    )
    positive_signals: list[str] = Field(
        default_factory=list,
        description="Key positive aspects repeatedly highlighted by customers",
    )
    negative_signals: list[str] = Field(
        default_factory=list,
        description="Major friction points, complaints, or negative patterns",
    )
    recurring_issues: list[str] = Field(
        default_factory=list,
        description="Recurring operational, product, or service issues",
    )
    business_interpretation: str = Field(
        ...,
        description="High-level business implications and executive takeaway",
    )
    evidence_references: list[EvidenceReference] = Field(
        default_factory=list,
        description="Traceable evidence references supporting this summary",
    )


class SentimentInsight(BaseModel):
    """Analysis and interpretation of sentiment distributions and polarity."""

    model_config = ConfigDict(extra="ignore")

    dominant_sentiment: str = Field(
        ...,
        description="Dominant sentiment tendency ('positive', 'mixed', etc.)",
    )
    positive_themes: list[str] = Field(
        default_factory=list,
        description="Key drivers of positive customer sentiment",
    )
    negative_themes: list[str] = Field(
        default_factory=list,
        description="Key drivers of negative customer sentiment",
    )
    distribution_summary: str = Field(
        ...,
        description="Interpretation of sentiment distribution across ratings",
    )
    model_limitations: str = Field(
        ...,
        description="Limitations of underlying sentiment model and weak labels",
    )
    evidence_references: list[EvidenceReference] = Field(
        default_factory=list,
        description="Evidence IDs grounding the sentiment findings",
    )


class TopicInsight(BaseModel):
    """Business interpretation of an unsupervised topic cluster."""

    model_config = ConfigDict(extra="ignore")

    cluster_id: int = Field(
        ...,
        description="Numeric identifier of the topic cluster",
    )
    topic_label: str = Field(
        ...,
        description="Descriptive business title for the discovered topic",
    )
    interpretation: str = Field(
        ...,
        description="Interpretation of customer feedback in this cluster",
    )
    customer_concerns: list[str] = Field(
        default_factory=list,
        description="Specific concerns or remarks prevalent in this topic",
    )
    business_relevance: str = Field(
        ...,
        description="Relevance to marketing, product, or support teams",
    )
    is_emerging: bool = Field(
        default=False,
        description="Whether temporal evidence supports this as emerging",
    )
    temporal_evidence: Optional[str] = Field(
        default=None,
        description="Supporting temporal observation if marked as emerging",
    )
    evidence_references: list[EvidenceReference] = Field(
        default_factory=list,
        description="Representative review IDs grounding this topic insight",
    )


class ComparisonInsight(BaseModel):
    """Comparative analysis across products, channels, or feedback sources."""

    model_config = ConfigDict(extra="ignore")

    entity_type: str = Field(
        default="product",
        description="Entity being compared ('product', 'channel', 'source')",
    )
    comparison_name: str = Field(
        ...,
        description="Title of the comparative analysis",
    )
    findings: list[str] = Field(
        default_factory=list,
        description="Observed empirical differences across entities",
    )
    sample_size_adequate: bool = Field(
        default=True,
        description="Whether comparative sample sizes meet statistical thresholds",
    )
    limitations: str = Field(
        default=(
            "Sample size must be considered. Does not reflect external market share."
        ),
        description="Statistical boundaries; competitor market share is never inferred",
    )
    evidence_references: list[EvidenceReference] = Field(
        default_factory=list,
        description="Supporting evidence references",
    )


class MarketingRecommendation(BaseModel):
    """Evidence-grounded, actionable marketing or operational recommendation."""

    model_config = ConfigDict(extra="ignore")

    recommendation_id: str = Field(
        default_factory=lambda: f"REC-{uuid4().hex[:6].upper()}",
        description="Unique identifier for the recommendation",
    )
    title: str = Field(
        ...,
        description="Concise recommendation headline",
    )
    description: str = Field(
        ...,
        description="Detailed description of the recommendation",
    )
    supporting_observation: str = Field(
        ...,
        description="Empirical analytical observation motivating recommendation",
    )
    evidence_ids: list[str] = Field(
        default_factory=list,
        description="Allowed evidence IDs grounding this recommendation",
    )
    impact_category: str = Field(
        ...,
        description="Business area impacted ('product_quality', 'messaging', etc.)",
    )
    priority: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="Implementation urgency",
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence score in recommendation's basis (0.0 to 1.0)",
    )
    suggested_action: str = Field(
        ...,
        description="Specific, concrete action suggested for teams",
    )


class LLMExecutionMetadata(BaseModel):
    """Traceable execution metadata for an LLM intelligence generation run."""

    model_config = ConfigDict(extra="ignore")

    provider: Optional[str] = Field(
        default=None,
        description="LLM provider name ('mock', 'openai', etc.)",
    )
    model: Optional[str] = Field(
        default=None,
        description="Underlying model identifier",
    )
    input_tokens: Optional[int] = Field(
        default=None,
        description="Number of prompt/input tokens processed",
    )
    output_tokens: Optional[int] = Field(
        default=None,
        description="Number of completion/output tokens generated",
    )
    total_tokens: Optional[int] = Field(
        default=None,
        description="Total tokens consumed",
    )
    estimated_cost_usd: Optional[float] = Field(
        default=None,
        description="Estimated API cost in USD based on documented pricing",
    )
    latency_ms: Optional[float] = Field(
        default=None,
        description="Execution latency in milliseconds",
    )
    executed_at: str = Field(
        ...,
        description="ISO 8601 timestamp of execution",
    )
    status: Literal["success", "unavailable", "failed", "validation_error"] = Field(
        default="success",
        description="Status of the intelligence generation execution",
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Sanitized error description if failed or unavailable",
    )


_DEFAULT_DISCLAIMER = (
    "LLM-generated insights represent algorithmic interpretations of customer "
    "feedback data and must be validated against operational evidence. Classical "
    "sentiment and topic predictions are statistical models and should not be "
    "construed as verified objective facts."
)


class IntelligenceReport(BaseModel):
    """
    Consolidated, validated intelligence report combining analytical insights.

    Can be serialized to JSON, stored, and loaded independently of the ML pipeline.
    """

    model_config = ConfigDict(extra="ignore")

    report_id: str = Field(
        default_factory=lambda: f"INTEL-{uuid4().hex[:8].upper()}",
        description="Unique report identifier",
    )
    generated_at: str = Field(
        ...,
        description="ISO 8601 generation timestamp",
    )
    status: Literal["completed", "unavailable", "partial", "failed"] = Field(
        default="completed",
        description="Overall report generation status",
    )
    dataset_scope: dict[str, Any] = Field(
        default_factory=dict,
        description="Dataset scope metadata (counts, spans, coverage)",
    )
    executive_summary: Optional[ExecutiveSummary] = Field(
        default=None,
        description="Executive summary section",
    )
    sentiment_intelligence: Optional[SentimentInsight] = Field(
        default=None,
        description="Sentiment intelligence section",
    )
    topic_insights: list[TopicInsight] = Field(
        default_factory=list,
        description="Cluster-by-cluster topic interpretation",
    )
    comparison_insights: list[ComparisonInsight] = Field(
        default_factory=list,
        description="Comparative insights across products or channels",
    )
    recommendations: list[MarketingRecommendation] = Field(
        default_factory=list,
        description="Evidence-grounded marketing recommendations",
    )
    allowed_evidence_ids: list[str] = Field(
        default_factory=list,
        description="Validated universe of evidence IDs available during analysis",
    )
    execution_metadata: LLMExecutionMetadata = Field(
        ...,
        description="Operational and cost metadata",
    )
    limitations_disclaimer: str = Field(
        default=_DEFAULT_DISCLAIMER,
        description="Standard disclaimer regarding algorithmic intelligence",
    )
