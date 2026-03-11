from __future__ import annotations

from uuid import uuid4

from rlrgf.classifier import FailureClassifier
from rlrgf.models import (
    CouncilAggregation,
    CouncilVerdict,
    Decision,
    EvaluatorRole,
    FailureType,
    ModelOutput,
    TokenUsage,
)


def make_model_output() -> ModelOutput:
    return ModelOutput(
        query_id=uuid4(),
        generated_answer="answer",
        token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        generation_latency_ms=25.0,
        prompt_hash="hash",
    )


def make_aggregation(
    *,
    supported_claim_ratio: float,
    final_decision: Decision,
    aggregate_safety_flag: bool = False,
    disagreement_score: float = 0.0,
    leakage_risk: float = 0.0,
) -> CouncilAggregation:
    query_id = uuid4()
    verdicts = [
        CouncilVerdict(
            query_id=query_id,
            evaluator_role=EvaluatorRole.GROUNDING_INSPECTOR,
            grounding_score=supported_claim_ratio,
        ),
        CouncilVerdict(
            query_id=query_id,
            evaluator_role=EvaluatorRole.SAFETY_AUDITOR,
            safety_flag=aggregate_safety_flag,
            leakage_risk=leakage_risk,
        ),
        CouncilVerdict(
            query_id=query_id,
            evaluator_role=EvaluatorRole.CRITIC,
            critic_score=0.8,
        ),
    ]
    return CouncilAggregation(
        query_id=query_id,
        verdicts=verdicts,
        final_decision=final_decision,
        aggregate_safety_flag=aggregate_safety_flag,
        disagreement_score=disagreement_score,
        supported_claim_ratio=supported_claim_ratio,
    )


def test_classifier_emits_injection_success() -> None:
    classifier = FailureClassifier()
    record = classifier.classify(
        council_result=make_aggregation(
            supported_claim_ratio=0.95,
            final_decision=Decision.ACCEPT,
            aggregate_safety_flag=False,
        ),
        model_output=make_model_output(),
        retrieval_chunks=[],
        injection_flags=["ignore previous instructions"],
    )
    assert record.failure_type == FailureType.INJECTION_SUCCESS


def test_classifier_emits_instability() -> None:
    classifier = FailureClassifier()
    record = classifier.classify(
        council_result=make_aggregation(
            supported_claim_ratio=0.95,
            final_decision=Decision.ACCEPT,
            disagreement_score=0.75,
        ),
        model_output=make_model_output(),
        retrieval_chunks=[],
        injection_flags=[],
    )
    assert record.failure_type == FailureType.INSTABILITY
    assert record.instability_detected is True


def test_classifier_emits_multiple_failures() -> None:
    classifier = FailureClassifier()
    record = classifier.classify(
        council_result=make_aggregation(
            supported_claim_ratio=0.2,
            final_decision=Decision.ABSTAIN,
            aggregate_safety_flag=True,
            leakage_risk=0.9,
        ),
        model_output=make_model_output(),
        retrieval_chunks=[],
        injection_flags=[],
    )
    assert record.failure_type == FailureType.MULTIPLE_FAILURES
