"""
MarketLens - Evidence Validator & Hallucination Guard.

Strictly validates model-generated evidence references and recommendations
against the authoritative allowed evidence ID set established during context
preparation. Discards fabricated IDs and ungrounded claims.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("marketlens.intelligence.validator")


class EvidenceValidationSummary:
    """Record of evidence validation outcomes and detected anomalies."""

    def __init__(self) -> None:
        self.total_evidence_references = 0
        self.valid_evidence_references = 0
        self.rejected_evidence_ids: list[str] = []
        self.dropped_recommendations_count = 0


def _classify_evidence_type(
    evidence_id: str,
    review_ids: set[str],
    metric_ids: set[str],
) -> str:
    """Classify evidence reference as 'review', 'metric', or 'aggregate'."""
    if evidence_id in review_ids:
        return "review"
    if evidence_id in metric_ids or evidence_id.startswith("METRIC:"):
        return "metric"
    return "aggregate"


def sanitize_and_validate_evidence(
    data: dict[str, Any],
    *,
    allowed_evidence_ids: list[str],
    review_evidence_map: dict[str, dict[str, Any]],
    metric_evidence_map: dict[str, Any],
) -> tuple[dict[str, Any], EvidenceValidationSummary]:
    """
    Sanitize and validate raw LLM output against the allowed evidence universe.

    Rules enforced:
    1. Every evidence ID must strictly exist in `allowed_evidence_ids`.
    2. Hallucinated / invented IDs are stripped immediately.
    3. Recommendations without a supporting observation are dropped.
    4. Evidence references are explicitly tagged as 'review' vs. 'metric'.
    5. Disclaimers for market-share inference boundaries are guaranteed.

    Parameters
    ----------
    data : dict[str, Any]
        Raw parsed JSON dictionary from the LLM provider.
    allowed_evidence_ids : list[str]
        Authoritative list of IDs passed in the prompt context.
    review_evidence_map : dict[str, dict[str, Any]]
        Map of review IDs to review metadata.
    metric_evidence_map : dict[str, Any]
        Map of metric IDs to statistical metadata.

    Returns
    -------
    tuple[dict[str, Any], EvidenceValidationSummary]
        Sanitized data dictionary and validation summary.
    """
    allowed_set = set(allowed_evidence_ids)
    review_set = set(review_evidence_map.keys())
    metric_set = set(metric_evidence_map.keys())

    summary = EvidenceValidationSummary()
    sanitized = dict(data)

    def _filter_evidence_refs(raw_refs: list[Any]) -> list[dict[str, Any]]:
        clean_refs: list[dict[str, Any]] = []
        for ref in raw_refs:
            summary.total_evidence_references += 1
            if isinstance(ref, str):
                eid = ref.strip()
                desc = None
                src = None
            elif isinstance(ref, dict):
                eid = str(ref.get("evidence_id", "")).strip()
                desc = ref.get("description")
                src = ref.get("source_field")
            else:
                continue

            if not eid:
                continue

            if eid in allowed_set:
                summary.valid_evidence_references += 1
                ev_type = _classify_evidence_type(eid, review_set, metric_set)
                clean_refs.append(
                    {
                        "evidence_id": eid,
                        "evidence_type": ev_type,
                        "description": desc,
                        "source_field": src,
                    }
                )
            else:
                summary.rejected_evidence_ids.append(eid)
                logger.warning(
                    "Rejected fabricated or ungrounded evidence ID from LLM: '%s'",
                    eid,
                )
        return clean_refs

    # 1. Executive Summary validation
    if "executive_summary" in sanitized and isinstance(
        sanitized["executive_summary"], dict
    ):
        es = sanitized["executive_summary"]
        raw_refs = es.get("evidence_references", [])
        es["evidence_references"] = _filter_evidence_refs(raw_refs)

    # 2. Sentiment Intelligence validation
    if "sentiment_intelligence" in sanitized and isinstance(
        sanitized["sentiment_intelligence"], dict
    ):
        si = sanitized["sentiment_intelligence"]
        raw_refs = si.get("evidence_references", [])
        si["evidence_references"] = _filter_evidence_refs(raw_refs)

    # 3. Topic Insights validation
    if "topic_insights" in sanitized and isinstance(sanitized["topic_insights"], list):
        clean_topics: list[dict[str, Any]] = []
        for ti in sanitized["topic_insights"]:
            if not isinstance(ti, dict):
                continue
            ti_copy = dict(ti)
            raw_refs = ti_copy.get("evidence_references", [])
            ti_copy["evidence_references"] = _filter_evidence_refs(raw_refs)
            clean_topics.append(ti_copy)
        sanitized["topic_insights"] = clean_topics

    # 4. Comparison Insights validation
    if "comparison_insights" in sanitized and isinstance(
        sanitized["comparison_insights"], list
    ):
        clean_comparisons: list[dict[str, Any]] = []
        for ci in sanitized["comparison_insights"]:
            if not isinstance(ci, dict):
                continue
            ci_copy = dict(ci)
            raw_refs = ci_copy.get("evidence_references", [])
            ci_copy["evidence_references"] = _filter_evidence_refs(raw_refs)
            # Ensure market-share boundary disclaimer is preserved
            base_lim = str(ci_copy.get("limitations", ""))
            if "market share" not in base_lim.lower():
                disclaimer = (
                    "Note: Customer feedback volume does not infer "
                    "competitor market share."
                )
                ci_copy["limitations"] = f"{base_lim} {disclaimer}".strip()
            clean_comparisons.append(ci_copy)
        sanitized["comparison_insights"] = clean_comparisons

    # 5. Marketing Recommendations validation (Mandatory strict grounding)
    if "recommendations" in sanitized and isinstance(
        sanitized["recommendations"], list
    ):
        valid_recs: list[dict[str, Any]] = []
        for rec in sanitized["recommendations"]:
            if not isinstance(rec, dict):
                continue
            obs = str(rec.get("supporting_observation", "")).strip()
            # Must not generate recommendations without supporting observations
            if not obs:
                summary.dropped_recommendations_count += 1
                logger.warning(
                    "Dropping recommendation without supporting observation: '%s'",
                    rec.get("title"),
                )
                continue

            raw_eids = rec.get("evidence_ids", [])
            valid_eids: list[str] = []
            for eid_item in raw_eids:
                eid_str = str(eid_item).strip()
                summary.total_evidence_references += 1
                if eid_str in allowed_set:
                    valid_eids.append(eid_str)
                    summary.valid_evidence_references += 1
                else:
                    summary.rejected_evidence_ids.append(eid_str)
                    logger.warning(
                        "Removed invalid evidence ID '%s' from recommendation '%s'",
                        eid_str,
                        rec.get("title"),
                    )

            # Keep recommendation with valid grounded evidence IDs
            rec_copy = dict(rec)
            rec_copy["evidence_ids"] = valid_eids
            valid_recs.append(rec_copy)

        sanitized["recommendations"] = valid_recs

    return sanitized, summary
