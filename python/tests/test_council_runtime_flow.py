from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel

from rlrgf.council_runtime import CouncilRuntime
from rlrgf.council_runtime_config import CouncilRuntimeConfig
from rlrgf.council_runtime_inference_adapter import RemoteInferenceResult
from rlrgf.council_runtime_schemas import (
    CouncilMode,
    CouncilRequest,
    CouncilSeat,
    FailureFlag,
    FinalSynthesisPayload,
    InitialAnswerPayload,
    PeerCritiquePayload,
    RevisedAnswerPayload,
)


class _FakeAdapter:
    def __init__(self, *, fail_synthesis: bool = False) -> None:
        self.fail_synthesis = fail_synthesis
        self.calls: list[tuple[str, str]] = []

    def call_json(
        self,
        *,
        model_id: str,
        messages: list[dict[str, str]],
        schema_model: type[BaseModel],
        timeout_s: float,
        temperature: float = 0.2,
        max_tokens: int = 900,
    ) -> tuple[Optional[BaseModel], RemoteInferenceResult, Optional[str]]:
        _ = (messages, timeout_s, temperature, max_tokens)
        name = schema_model.__name__
        self.calls.append((name, model_id))

        if schema_model is FinalSynthesisPayload and self.fail_synthesis:
            return (
                None,
                RemoteInferenceResult(
                    status="timeout",
                    text="",
                    latency_ms=20.0,
                    model_id=model_id,
                    error="Request timed out.",
                ),
                "Request timed out.",
            )

        if schema_model is InitialAnswerPayload:
            confidence = 0.2 if model_id in {"model-a", "model-c"} else 0.8
            payload = InitialAnswerPayload(
                answer=f"answer from {model_id}",
                confidence=confidence,
                grounding_confidence=confidence,
                key_points=["point-a", "point-b"],
                uncertainty_notes=[],
                cited_risks=[],
                citations=[],
            )
            return payload, self._ok_result(model_id), None

        if schema_model is PeerCritiquePayload:
            payload = PeerCritiquePayload(critiques=[], confidence=0.7)
            return payload, self._ok_result(model_id), None

        if schema_model is RevisedAnswerPayload:
            payload = RevisedAnswerPayload(
                revised_answer=f"revised answer from {model_id}",
                confidence=0.7,
                grounding_confidence=0.7,
                change_summary="minor edits",
            )
            return payload, self._ok_result(model_id), None

        if schema_model is FinalSynthesisPayload:
            payload = FinalSynthesisPayload(
                final_answer="final synthesized answer",
                reasoning_summary="combined strongest points",
                winner_seat_ids=["seat-a"],
                strongest_contributors=["seat-a"],
                uncertainty_notes=[],
                cited_risk_notes=[],
                confidence=0.75,
            )
            return payload, self._ok_result(model_id), None

        raise AssertionError(f"Unexpected schema: {name}")

    def _ok_result(self, model_id: str) -> RemoteInferenceResult:
        return RemoteInferenceResult(
            status="ok",
            text='{"ok": true}',
            latency_ms=12.0,
            model_id=model_id,
        )


def _runtime_config(cache_path: Path) -> CouncilRuntimeConfig:
    return CouncilRuntimeConfig(
        base_url="http://unit-test.local",
        api_key="test-key",
        model_timeout_s=5.0,
        fast_quorum=2,
        full_quorum=3,
        escalation_confidence_threshold=0.55,
        escalation_disagreement_threshold=0.45,
        chair_seat_id="seat-a",
        backup_chair_seat_id="seat-c",
        cache_path=str(cache_path),
        seats=[
            CouncilSeat(
                seat_id="seat-a",
                role_title="Seat A",
                model_id="model-a",
                enabled_in_fast_mode=True,
                can_chair=True,
            ),
            CouncilSeat(
                seat_id="seat-b",
                role_title="Seat B",
                model_id="model-b",
                enabled_in_fast_mode=False,
                can_chair=True,
            ),
            CouncilSeat(
                seat_id="seat-c",
                role_title="Seat C",
                model_id="model-c",
                enabled_in_fast_mode=True,
                can_chair=True,
            ),
        ],
    )


def test_fast_mode_escalates_to_full_and_adds_missing_initial_answers(
    tmp_path: Path,
) -> None:
    adapter = _FakeAdapter()
    runtime = CouncilRuntime(config=_runtime_config(tmp_path / "cache.json"), adapter=adapter)

    trace = runtime.run(
        CouncilRequest(
            query="How should we secure a RAG app?",
            mode=CouncilMode.FAST_COUNCIL,
            demo_mode=False,
        )
    )

    assert trace.escalated_to_full is True
    assert {answer.seat_id for answer in trace.initial_answers} == {
        "seat-a",
        "seat-b",
        "seat-c",
    }
    assert len([call for call in adapter.calls if call[0] == "InitialAnswerPayload"]) == 3


def test_synthesis_failure_triggers_fallback(tmp_path: Path) -> None:
    adapter = _FakeAdapter(fail_synthesis=True)
    runtime = CouncilRuntime(config=_runtime_config(tmp_path / "cache.json"), adapter=adapter)

    trace = runtime.run(
        CouncilRequest(
            query="What release gates should block deployment?",
            mode=CouncilMode.FULL_COUNCIL,
            demo_mode=False,
        )
    )

    assert trace.final_synthesis is not None
    assert trace.final_synthesis.fallback_used is True
    assert any(failure.flag == FailureFlag.SYNTHESIS_FAILURE for failure in trace.failures)


def test_demo_mode_cache_returns_cached_trace_without_extra_calls(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.json"
    adapter = _FakeAdapter()
    runtime = CouncilRuntime(config=_runtime_config(cache_path), adapter=adapter)

    first = runtime.run(
        CouncilRequest(
            query="How do we control prompt injection?",
            mode=CouncilMode.FAST_COUNCIL,
            demo_mode=True,
        )
    )
    first_call_count = len(adapter.calls)
    second = runtime.run(
        CouncilRequest(
            query="How do we control prompt injection?",
            mode=CouncilMode.FAST_COUNCIL,
            demo_mode=True,
        )
    )

    assert first.cached is False
    assert second.cached is True
    assert len(adapter.calls) == first_call_count
