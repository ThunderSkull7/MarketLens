"""
MarketLens - LLM Intelligence Layer Package.

Provides provider-neutral LLM adapters, evidence-grounded marketing insights,
actionable recommendations, context preparation, and structured report persistence.
"""

from __future__ import annotations

from marketlens.intelligence.context import (
    IntelligenceContext,
    prepare_intelligence_context,
)
from marketlens.intelligence.engine import IntelligenceEngine
from marketlens.intelligence.persistence import (
    load_intelligence_report,
    save_intelligence_report,
)
from marketlens.intelligence.provider import (
    LLMProvider,
    LLMResponse,
    MockLLMProvider,
    OpenAICompatibleProvider,
    create_llm_provider,
    estimate_token_cost,
)
from marketlens.intelligence.schemas import (
    ComparisonInsight,
    EvidenceReference,
    ExecutiveSummary,
    IntelligenceReport,
    LLMExecutionMetadata,
    MarketingRecommendation,
    SentimentInsight,
    TopicInsight,
)

__all__ = [
    "ComparisonInsight",
    "EvidenceReference",
    "ExecutiveSummary",
    "IntelligenceContext",
    "IntelligenceEngine",
    "IntelligenceReport",
    "LLMExecutionMetadata",
    "LLMProvider",
    "LLMResponse",
    "MarketingRecommendation",
    "MockLLMProvider",
    "OpenAICompatibleProvider",
    "SentimentInsight",
    "TopicInsight",
    "create_llm_provider",
    "estimate_token_cost",
    "load_intelligence_report",
    "prepare_intelligence_context",
    "save_intelligence_report",
]
