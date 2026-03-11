from __future__ import annotations

from uuid import uuid4

import pytest

from rlrgf.council import CouncilAggregator, Critic, GroundingInspector, SafetyAuditor
from rlrgf.models import EvaluatorRole, RetrievedChunkRef


@pytest.mark.asyncio
async def test_critic_verdict_uses_critic_role_and_score() -> None:
    verdict = await Critic().evaluate(
        query_id=uuid4(),
        model_output="answer",
        context=[],
        model_name="phi-3-mini",
        sequence_id=1,
    )

    assert verdict.evaluator_role == EvaluatorRole.CRITIC
    assert verdict.critic_score > 0.0
    assert verdict.grounding_score == 0.0


@pytest.mark.asyncio
async def test_critic_signal_survives_aggregation() -> None:
    query_id = uuid4()
    chunk = RetrievedChunkRef(
        chunk_id=uuid4(),
        doc_id=uuid4(),
        similarity_score=0.9,
        chunk_text="evidence",
        chunk_index=0,
    )

    grounding = await GroundingInspector().evaluate(
        query_id=query_id,
        answer="answer",
        chunks=[chunk],
        model_name="phi-3-mini",
        sequence_id=1,
    )
    safety = await SafetyAuditor().evaluate(
        query_id=query_id,
        model_output="answer",
        query="question",
        injection_flags=[],
        model_name="phi-3-mini",
    )
    critic = await Critic().evaluate(
        query_id=query_id,
        model_output="answer",
        context=[chunk],
        model_name="phi-3-mini",
        sequence_id=1,
    )

    aggregation = CouncilAggregator().aggregate(query_id, [grounding, safety, critic])

    assert any(v.evaluator_role == EvaluatorRole.CRITIC for v in aggregation.verdicts)
    assert aggregation.disagreement_score >= 0.0
