from __future__ import annotations

import re

from .execution_state import (
    AgentExecutionState,
    TaskClassification,
    VerifierResult,
    VerifierVerdict,
)


def _looks_like_code(answer: str) -> bool:
    text = (answer or "").strip()
    if not text:
        return False
    return any(
        (
            "```" in text,
            bool(re.search(r"(?m)^\s*def\s+\w+\s*\(", text)),
            bool(re.search(r"(?m)^\s*(from|import)\s+\w+", text)),
            bool(re.search(r"(?m)^\s*class\s+\w+\s*[:(]", text)),
        )
    )


class AnswerVerifier:
    def verify(self, state: AgentExecutionState) -> VerifierResult:
        answer = state.candidate_answer.strip()
        if not answer:
            return VerifierResult(
                verdict=VerifierVerdict.INSUFFICIENT_EVIDENCE,
                rationale="No candidate answer was produced.",
                evidence_coverage=state.evidence_coverage,
                hallucination_risk=state.hallucination_risk,
            )

        if state.task_classification == TaskClassification.CODING and not _looks_like_code(answer):
            return VerifierResult(
                verdict=VerifierVerdict.REVISE,
                rationale="Prompt requires code, but the answer does not contain code-like content.",
                evidence_coverage=state.evidence_coverage,
                hallucination_risk=state.hallucination_risk,
            )

        if state.disagreement_score is not None and state.disagreement_score >= 0.55:
            return VerifierResult(
                verdict=VerifierVerdict.CONFLICTING_EVIDENCE,
                rationale="Disagreement remains above the conservative escalation threshold.",
                evidence_coverage=state.evidence_coverage,
                conflict_score=state.disagreement_score,
                hallucination_risk=state.hallucination_risk,
            )

        if state.hallucination_risk is not None and state.hallucination_risk >= 0.7:
            return VerifierResult(
                verdict=VerifierVerdict.HIGH_HALLUCINATION_RISK,
                rationale="Hallucination risk remains above the acceptable limit.",
                evidence_coverage=state.evidence_coverage,
                hallucination_risk=state.hallucination_risk,
            )

        if state.retrieval_context and state.evidence_coverage < 0.25:
            return VerifierResult(
                verdict=VerifierVerdict.INSUFFICIENT_EVIDENCE,
                rationale="Retrieved context is still too weak to verify the answer.",
                evidence_coverage=state.evidence_coverage,
                hallucination_risk=state.hallucination_risk,
            )

        return VerifierResult(
            verdict=VerifierVerdict.PASS,
            rationale="Answer satisfied the current conservative verification checks.",
            evidence_coverage=state.evidence_coverage,
            conflict_score=state.disagreement_score or 0.0,
            hallucination_risk=state.hallucination_risk,
        )
