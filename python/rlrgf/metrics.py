"""
Metrics Engine - Computes aggregate experiment metrics from evaluation records.
"""

from __future__ import annotations

import statistics
from datetime import datetime, timezone
from typing import Optional

from .models import Decision, EvaluationRecord, ExperimentMetrics


class MetricsEngine:
    """
    Computes aggregate metrics from a collection of evaluation records.
    """

    def compute(
        self,
        records: list[EvaluationRecord],
        experiment_id: str = "default",
    ) -> ExperimentMetrics:
        """Compute aggregate metrics from evaluation records."""
        total = len(records)
        if total == 0:
            return ExperimentMetrics(
                experiment_id=experiment_id,
                timestamp=datetime.now(timezone.utc),
            )

        # --- Core Metrics -------------------------------------
        refusal_count = sum(
            1 for r in records if r.decision in (Decision.ABSTAIN, Decision.ESCALATE)
        )
        hallucination_count = sum(1 for r in records if r.hallucination_flag)
        policy_violation_count = sum(1 for r in records if r.policy_violation)
        leakage_count = sum(1 for r in records if r.leakage_detected)
        injection_count = sum(1 for r in records if r.injection_success)
        instability_count = sum(1 for r in records if r.instability_detected)

        # --- RAG Metrics --------------------------------------
        avg_precision = self._mean_optional(
            [r.retrieval_precision_at_k for r in records]
        )
        avg_recall = self._mean_optional(
            [r.retrieval_recall_at_k for r in records]
        )
        avg_citation = self._mean_optional(
            [r.citation_precision for r in records]
        )
        avg_supported = statistics.mean(r.supported_claim_ratio for r in records)

        # --- System Metrics -----------------------------------
        latencies = sorted(
            r.generation_latency_ms + r.retrieval_latency_ms
            for r in records
            if r.retrieval_latency_ms is not None
        )
        p50_idx = max(0, len(latencies) // 2 - 1)
        p95_idx = max(0, int(len(latencies) * 0.95) - 1)
        p50_latency = latencies[p50_idx] if latencies else 0.0
        p95_latency = latencies[p95_idx] if latencies else 0.0

        avg_context_len = statistics.mean(r.context_token_count for r in records)
        avg_risk_score = statistics.mean(r.risk_score for r in records)

        # --- Council Metrics ----------------------------------
        # Safety override: cases where council overrode to abstain
        safety_overrides = sum(
            1 for r in records
            if r.decision == Decision.ABSTAIN and r.policy_violation
        )
        safety_override_freq = safety_overrides / total

        # Self-correction: cases where retrieval was requested
        self_corrections = sum(
            1 for r in records
            if r.decision == Decision.REQUEST_RETRIEVAL
        )
        self_correction_rate = self_corrections / total

        # Disagreement score proxy (from instability)
        disagreement_avg = instability_count / total

        return ExperimentMetrics(
            experiment_id=experiment_id,
            total_queries=total,
            refusal_rate=refusal_count / total,
            hallucination_rate=hallucination_count / total,
            policy_violation_rate=policy_violation_count / total,
            leakage_rate=leakage_count / total,
            injection_success_rate=injection_count / total,
            instability_rate=instability_count / total,
            avg_retrieval_precision_at_k=avg_precision,
            avg_retrieval_recall_at_k=avg_recall,
            avg_citation_precision=avg_citation,
            avg_supported_claim_ratio=avg_supported,
            council_disagreement_avg=disagreement_avg,
            safety_override_frequency=safety_override_freq,
            self_correction_rate=self_correction_rate,
            p50_latency_ms=p50_latency,
            p95_latency_ms=p95_latency,
            avg_context_length_tokens=avg_context_len,
            avg_risk_score=avg_risk_score,
            timestamp=datetime.now(timezone.utc),
        )

    @staticmethod
    def _mean_optional(values: list[Optional[float]]) -> Optional[float]:
        known = [value for value in values if value is not None]
        if not known:
            return None
        return statistics.mean(known)
