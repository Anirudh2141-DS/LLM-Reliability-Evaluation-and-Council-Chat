from __future__ import annotations

from uuid import uuid4

import pytest

from rlrgf.models import RetrievedChunkRef
from rlrgf.pipeline import EvaluationPipeline, PipelineConfig, TestCase


@pytest.mark.asyncio
async def test_retrieval_metrics_are_derived_from_retrieved_docs(tmp_path) -> None:
    pipeline = EvaluationPipeline(PipelineConfig(output_dir=str(tmp_path)))
    query_id = uuid4()
    relevant_doc_id = uuid4()
    distractor_doc_id = uuid4()

    test_case = TestCase(
        query_id=query_id,
        query="What is RLRGF?",
        expected_doc_ids={relevant_doc_id},
    )
    chunks = [
        RetrievedChunkRef(
            chunk_id=uuid4(),
            doc_id=relevant_doc_id,
            similarity_score=0.9,
            chunk_text="RLRGF is a reliability framework.",
            chunk_index=0,
        ),
        RetrievedChunkRef(
            chunk_id=uuid4(),
            doc_id=distractor_doc_id,
            similarity_score=0.7,
            chunk_text="Distractor document.",
            chunk_index=1,
        ),
    ]

    record = await pipeline.evaluate_test_case(
        test_case=test_case,
        chunks=chunks,
        evaluator_model="phi-3-mini",
        sequence_id=1,
        retrieval_latency_ms=12.5,
    )

    assert record.retrieval_latency_ms == pytest.approx(12.5)
    assert record.retrieval_precision_at_k == pytest.approx(0.5)
    assert record.retrieval_recall_at_k == pytest.approx(1.0)
    assert record.citation_precision is None


@pytest.mark.asyncio
async def test_model_version_matches_actual_evaluator_model(tmp_path) -> None:
    pipeline = EvaluationPipeline(PipelineConfig(output_dir=str(tmp_path)))
    test_case = TestCase(query="question")
    chunks: list[RetrievedChunkRef] = []

    record_a = await pipeline.evaluate_test_case(
        test_case=test_case,
        chunks=chunks,
        evaluator_model="llama-3-8b",
        sequence_id=1,
        retrieval_latency_ms=5.0,
    )
    record_b = await pipeline.evaluate_test_case(
        test_case=test_case,
        chunks=chunks,
        evaluator_model="qwen-2-7b",
        sequence_id=1,
        retrieval_latency_ms=5.0,
    )

    assert record_a.model_version == "llama-3-8b"
    assert record_b.model_version == "qwen-2-7b"
