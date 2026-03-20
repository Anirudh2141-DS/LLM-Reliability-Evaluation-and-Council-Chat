from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from rlrgf.council_runtime import CouncilRuntime, normalize_user_visible_answer_text
from rlrgf.council_runtime_config import CouncilRuntimeConfig
from rlrgf.council_runtime_inference_adapter import RemoteInferenceResult
from rlrgf.council_runtime_schemas import (
    AvailabilityStatus,
    CacheStatus,
    CouncilMode,
    CouncilRequest,
    CouncilRunTrace,
    CouncilSeat,
    CouncilStage,
    ExecutionMode,
    FailureFlag,
    FinalSynthesisPayload,
    InitialAnswerPayload,
    ModelScoreCard,
    PeerCritiquePayload,
    RevisedAnswerPayload,
)


@dataclass
class _Outcome:
    payload: Optional[BaseModel] = None
    status: str = "ok"
    error: Optional[str] = None
    parse_error: Optional[str] = None
    text: str = '{"ok": true}'
    latency_ms: float = 12.0


class _ScriptedAdapter:
    def __init__(
        self,
        *,
        outcomes: Optional[dict[tuple[str, str], _Outcome | list[_Outcome]]] = None,
        initial_conf_by_model: Optional[dict[str, float]] = None,
        initial_answer_by_model: Optional[dict[str, str]] = None,
    ) -> None:
        self._outcomes = outcomes or {}
        self._initial_conf_by_model = initial_conf_by_model or {}
        self._initial_answer_by_model = initial_answer_by_model or {}
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
        schema_name = schema_model.__name__
        self.calls.append((schema_name, model_id))

        outcome = self._outcomes.get((schema_name, model_id)) or self._outcomes.get(
            (schema_name, "*")
        )
        if isinstance(outcome, list):
            outcome = outcome.pop(0) if outcome else None
        if outcome is not None:
            result = RemoteInferenceResult(
                status=outcome.status,
                text=outcome.text if outcome.status == "ok" else "",
                latency_ms=outcome.latency_ms,
                model_id=model_id,
                error=outcome.error,
            )
            if outcome.payload is not None:
                return outcome.payload, result, None
            return None, result, outcome.parse_error

        return self._default(schema_model=schema_model, model_id=model_id)

    def _default(
        self, *, schema_model: type[BaseModel], model_id: str
    ) -> tuple[Optional[BaseModel], RemoteInferenceResult, Optional[str]]:
        if schema_model is InitialAnswerPayload:
            confidence = self._initial_conf_by_model.get(model_id, 0.82)
            answer = self._initial_answer_by_model.get(
                model_id,
                f"baseline answer from {model_id}",
            )
            payload = InitialAnswerPayload(
                answer=answer,
                confidence=confidence,
                grounding_confidence=confidence,
                key_points=["a", "b"],
                uncertainty_notes=[],
                cited_risks=[],
                citations=[],
            )
            return payload, self._ok(model_id), None

        if schema_model is PeerCritiquePayload:
            payload = PeerCritiquePayload(critiques=[], confidence=0.7)
            return payload, self._ok(model_id), None

        if schema_model is RevisedAnswerPayload:
            payload = RevisedAnswerPayload(
                revised_answer=f"revised answer from {model_id}",
                confidence=0.75,
                grounding_confidence=0.75,
                change_summary="updated wording",
            )
            return payload, self._ok(model_id), None

        if schema_model is FinalSynthesisPayload:
            payload = FinalSynthesisPayload(
                final_answer="synthesized final answer",
                reasoning_summary="combined strongest content",
                winner_seat_ids=["seat-a"],
                strongest_contributors=["seat-a"],
                uncertainty_notes=[],
                cited_risk_notes=[],
                confidence=0.78,
            )
            return payload, self._ok(model_id), None

        raise AssertionError(f"Unsupported schema in test adapter: {schema_model.__name__}")

    def _ok(self, model_id: str) -> RemoteInferenceResult:
        return RemoteInferenceResult(
            status="ok",
            text='{"ok": true}',
            latency_ms=11.0,
            model_id=model_id,
        )


class _HttpErrorAdapter(_ScriptedAdapter):
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
        _ = (messages, timeout_s, temperature, max_tokens, schema_model)
        self.calls.append((schema_model.__name__, model_id))
        result = RemoteInferenceResult(
            status="error",
            text="",
            latency_ms=15.0,
            model_id=model_id,
            error=(
                f"HTTP 400: {{\"error\":{{\"message\":\"The requested model '{model_id}' "
                "is not supported by any provider you have enabled.\"}}}}"
            ),
            http_status=400,
        )
        return None, result, result.error


def _runtime_config(cache_path: Path) -> CouncilRuntimeConfig:
    return CouncilRuntimeConfig(
        base_url="http://unit-test.local",
        api_key="test-key",
        model_timeout_s=5.0,
        fast_quorum=2,
        full_quorum=3,
        benchmark_enable_pairwise_critique=True,
        escalation_disagreement_threshold=0.45,
        escalation_confidence_threshold=0.55,
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
            CouncilSeat(
                seat_id="seat-d",
                role_title="Seat D",
                model_id="model-d",
                enabled_in_fast_mode=False,
                can_chair=True,
            ),
            CouncilSeat(
                seat_id="seat-e",
                role_title="Seat E",
                model_id="model-e",
                enabled_in_fast_mode=True,
                can_chair=False,
            ),
        ],
    )


def _count_calls(adapter: _ScriptedAdapter, schema_name: str) -> int:
    return sum(1 for call_schema, _ in adapter.calls if call_schema == schema_name)


def _run(
    tmp_path: Path,
    *,
    mode: CouncilMode,
    adapter: _ScriptedAdapter,
    query: str = "How should we secure a RAG system?",
    demo_mode: bool = False,
    force_live_rerun: bool = False,
    execution_mode: ExecutionMode = ExecutionMode.BENCHMARK,
) -> CouncilRunTrace:
    runtime = CouncilRuntime(config=_runtime_config(tmp_path / "cache.json"), adapter=adapter)
    return runtime.run(
        CouncilRequest(
            query=query,
            mode=mode,
            execution_mode=execution_mode,
            demo_mode=demo_mode,
            force_live_rerun=force_live_rerun,
        )
    )


def test_one_model_timeout_full_mode_quorum_survives(tmp_path: Path) -> None:
    adapter = _ScriptedAdapter(
        outcomes={
            ("InitialAnswerPayload", "model-b"): _Outcome(
                status="timeout",
                error="Request timed out.",
            )
        }
    )
    trace = _run(tmp_path, mode=CouncilMode.FULL_COUNCIL, adapter=adapter)

    assert trace.quorum_success is True
    assert any(
        failure.flag == FailureFlag.TIMEOUT and failure.seat_id == "seat-b"
        for failure in trace.failures
    )


def test_two_model_failures_in_fast_mode_still_returns_structured_trace(
    tmp_path: Path,
) -> None:
    adapter = _ScriptedAdapter(
        outcomes={
            ("InitialAnswerPayload", "model-a"): _Outcome(
                status="timeout",
                error="Request timed out.",
            ),
            ("InitialAnswerPayload", "model-c"): _Outcome(
                status="unavailable",
                error="model unavailable",
            ),
        }
    )
    trace = _run(tmp_path, mode=CouncilMode.FAST_COUNCIL, adapter=adapter)

    assert trace.final_synthesis is not None
    assert CouncilRunTrace.model_validate(trace.model_dump(mode="json"))
    assert len(trace.failures) >= 2


def test_critique_json_malformed_is_recorded(tmp_path: Path) -> None:
    adapter = _ScriptedAdapter(
        outcomes={
            ("PeerCritiquePayload", "model-e"): _Outcome(
                status="ok",
                parse_error="Malformed JSON output.",
            )
        }
    )
    trace = _run(tmp_path, mode=CouncilMode.FULL_COUNCIL, adapter=adapter)

    assert any(
        failure.stage == CouncilStage.PEER_CRITIQUE
        and failure.flag == FailureFlag.MALFORMED_JSON
        for failure in trace.failures
    )


def test_initial_json_malformed_is_recorded(tmp_path: Path) -> None:
    adapter = _ScriptedAdapter(
        outcomes={
            ("InitialAnswerPayload", "model-a"): _Outcome(
                status="ok",
                parse_error="Malformed JSON output.",
            )
        }
    )
    trace = _run(tmp_path, mode=CouncilMode.FULL_COUNCIL, adapter=adapter)

    assert any(
        failure.stage == CouncilStage.INITIAL_ANSWER
        and failure.flag == FailureFlag.MALFORMED_JSON
        for failure in trace.failures
    )


def test_empty_model_response_is_classified_as_empty_response(tmp_path: Path) -> None:
    adapter = _ScriptedAdapter(
        outcomes={
            ("InitialAnswerPayload", "model-a"): _Outcome(
                status="ok",
                parse_error="Empty response.",
            )
        }
    )
    trace = _run(tmp_path, mode=CouncilMode.FULL_COUNCIL, adapter=adapter)

    assert any(
        failure.stage == CouncilStage.INITIAL_ANSWER
        and failure.flag == FailureFlag.EMPTY_RESPONSE
        for failure in trace.failures
    )


def test_http_400_model_support_error_is_classified_as_unavailable_model(
    tmp_path: Path,
) -> None:
    adapter = _HttpErrorAdapter()
    trace = _run(tmp_path, mode=CouncilMode.FULL_COUNCIL, adapter=adapter)

    assert any(
        failure.stage == CouncilStage.INITIAL_ANSWER
        and failure.flag == FailureFlag.UNAVAILABLE_MODEL
        for failure in trace.failures
    )


def test_synthesis_json_malformed_falls_back(tmp_path: Path) -> None:
    adapter = _ScriptedAdapter(
        outcomes={
            ("FinalSynthesisPayload", "model-a"): _Outcome(
                status="ok",
                parse_error="Malformed JSON output.",
            ),
            ("FinalSynthesisPayload", "model-c"): _Outcome(
                status="ok",
                parse_error="Malformed JSON output.",
            ),
        }
    )
    trace = _run(tmp_path, mode=CouncilMode.FULL_COUNCIL, adapter=adapter)

    assert trace.final_synthesis is not None
    assert trace.final_synthesis.fallback_used is True
    assert any(f.flag == FailureFlag.SYNTHESIS_FAILURE for f in trace.failures)


def test_normalize_user_visible_answer_text_rejects_internal_schema_blobs() -> None:
    @dataclass
    class _AnswerWrapper:
        final_answer: str

    assert normalize_user_visible_answer_text(_AnswerWrapper("clean answer")) == "clean answer"
    assert (
        normalize_user_visible_answer_text(
            {
                "final_answer": "extracted answer",
                "confidence": 0.9,
                "winner_seat_ids": ["seat-a"],
            }
        )
        == "extracted answer"
    )
    assert (
        normalize_user_visible_answer_text(
            '{"title":"FinalSynthesisPayload","type":"object","properties":{"final_answer":{"type":"string"}}}'
        )
        == ""
    )
    assert (
        normalize_user_visible_answer_text(
            "FinalSynthesisPayload(final_answer='leak', confidence=0.8)"
        )
        == ""
    )


def test_interactive_numeric_conflict_prefers_direct_answer(tmp_path: Path) -> None:
    adapter = _ScriptedAdapter(
        initial_answer_by_model={"model-a": "9 sheep are left."},
        outcomes={
            ("FinalSynthesisPayload", "model-a"): _Outcome(
                payload=FinalSynthesisPayload(
                    final_answer="8 sheep are left.",
                    reasoning_summary="All but 9 die means 8 remain.",
                    winner_seat_ids=["seat-a"],
                    strongest_contributors=["seat-a"],
                    uncertainty_notes=[],
                    cited_risk_notes=[],
                    confidence=0.91,
                )
            )
        },
    )
    trace = _run(
        tmp_path,
        mode=CouncilMode.FAST_COUNCIL,
        adapter=adapter,
        query="A farmer has 17 sheep and all but 9 die. How many are left?",
        execution_mode=ExecutionMode.INTERACTIVE,
    )

    assert trace.final_synthesis is not None
    assert trace.final_synthesis.final_answer == "9 sheep are left."
    assert trace.final_synthesis.fallback_used is False
    assert [entry.stage for entry in trace.transcript] == [
        CouncilStage.INITIAL_ANSWER,
        CouncilStage.SYNTHESIS,
    ]


def test_interactive_invalid_synthesis_retries_once_then_succeeds(tmp_path: Path) -> None:
    adapter = _ScriptedAdapter(
        initial_answer_by_model={"model-a": "def top_k(nums, k): return nums[:k]"},
        outcomes={
            ("FinalSynthesisPayload", "model-a"): [
                _Outcome(
                    payload=FinalSynthesisPayload(
                        final_answer='{"title":"FinalSynthesisPayload","type":"object","properties":{"final_answer":{"type":"string"}}}',
                        reasoning_summary="schema leak",
                        winner_seat_ids=["seat-a"],
                        strongest_contributors=["seat-a"],
                        uncertainty_notes=[],
                        cited_risk_notes=[],
                        confidence=0.5,
                    )
                ),
                _Outcome(
                    payload=FinalSynthesisPayload(
                        final_answer=(
                            "Use a frequency map plus bucket sort to return the top k "
                            "elements in O(n) average time."
                        ),
                        reasoning_summary="Returned a user-facing explanation only.",
                        winner_seat_ids=["seat-a"],
                        strongest_contributors=["seat-a"],
                        uncertainty_notes=[],
                        cited_risk_notes=[],
                        confidence=0.8,
                    )
                ),
            ]
        },
    )
    trace = _run(
        tmp_path,
        mode=CouncilMode.FAST_COUNCIL,
        adapter=adapter,
        query="Write a Python function that returns the top k most frequent elements.",
        execution_mode=ExecutionMode.INTERACTIVE,
    )

    assert trace.final_synthesis is not None
    assert _count_calls(adapter, "FinalSynthesisPayload") == 2
    assert trace.final_synthesis.fallback_used is False
    assert "FinalSynthesisPayload" not in trace.final_synthesis.final_answer
    assert trace.final_synthesis.final_answer.startswith("Use a frequency map")


def test_interactive_invalid_synthesis_falls_back_to_best_initial_answer(
    tmp_path: Path,
) -> None:
    adapter = _ScriptedAdapter(
        initial_answer_by_model={"model-a": "def top_k(nums, k): return nums[:k]"},
        outcomes={
            ("FinalSynthesisPayload", "model-a"): [
                _Outcome(
                    payload=FinalSynthesisPayload(
                        final_answer='{"title":"FinalSynthesisPayload","type":"object","properties":{"final_answer":{"type":"string"}}}',
                        reasoning_summary="schema leak",
                        winner_seat_ids=["seat-a"],
                        strongest_contributors=["seat-a"],
                        uncertainty_notes=[],
                        cited_risk_notes=[],
                        confidence=0.5,
                    )
                ),
                _Outcome(
                    payload=FinalSynthesisPayload(
                        final_answer="FinalSynthesisPayload(final_answer='leak')",
                        reasoning_summary="repr leak",
                        winner_seat_ids=["seat-a"],
                        strongest_contributors=["seat-a"],
                        uncertainty_notes=[],
                        cited_risk_notes=[],
                        confidence=0.5,
                    )
                ),
            ]
        },
    )
    trace = _run(
        tmp_path,
        mode=CouncilMode.FAST_COUNCIL,
        adapter=adapter,
        query="Write a Python function that returns the top k most frequent elements.",
        execution_mode=ExecutionMode.INTERACTIVE,
    )

    assert trace.final_synthesis is not None
    assert _count_calls(adapter, "FinalSynthesisPayload") == 2
    assert trace.final_synthesis.fallback_used is True
    assert trace.final_synthesis.final_answer == "def top_k(nums, k): return nums[:k]"
    assert "FinalSynthesisPayload" not in trace.final_synthesis.final_answer
    assert any(
        failure.stage == CouncilStage.SYNTHESIS
        and failure.flag == FailureFlag.SYNTHESIS_FAILURE
        for failure in trace.failures
    )


def test_escalation_from_three_fast_to_five_full_triggers(tmp_path: Path) -> None:
    adapter = _ScriptedAdapter(
        initial_conf_by_model={"model-a": 0.2, "model-c": 0.2, "model-e": 0.2}
    )
    trace = _run(tmp_path, mode=CouncilMode.FAST_COUNCIL, adapter=adapter)

    assert trace.escalated_to_full is True
    assert _count_calls(adapter, "InitialAnswerPayload") == 5


def test_benchmark_all_seats_unavailable_returns_explicit_unavailable_message(
    tmp_path: Path,
) -> None:
    adapter = _ScriptedAdapter(
        outcomes={
            ("InitialAnswerPayload", "*"): _Outcome(
                status="unavailable",
                error="HTTP 402: Provider credits exhausted.",
            )
        }
    )
    trace = _run(
        tmp_path,
        mode=CouncilMode.FULL_COUNCIL,
        adapter=adapter,
        query="Design a production-ready LLM evaluation system.",
        execution_mode=ExecutionMode.BENCHMARK,
    )

    assert trace.final_synthesis is not None
    assert trace.final_synthesis.fallback_used is True
    assert "Benchmark execution could not complete" in trace.final_synthesis.final_answer
    assert trace.observability.number_of_models_requested == 5
    assert trace.observability.number_of_models_succeeded == 0
    assert trace.observability.number_of_models_failed == 5
    assert trace.observability.critique_enabled is False
    assert not any(
        entry.stage == CouncilStage.PEER_CRITIQUE for entry in trace.transcript
    )
    assert any(
        failure.flag == FailureFlag.UNAVAILABLE_MODEL for failure in trace.failures
    )
    assert not any(
        failure.flag == FailureFlag.MALFORMED_JSON for failure in trace.failures
    )


def test_escalation_is_skipped_when_fast_run_is_confident_and_consistent(
    tmp_path: Path,
) -> None:
    adapter = _ScriptedAdapter(
        initial_conf_by_model={"model-a": 0.9, "model-c": 0.9, "model-e": 0.9},
        initial_answer_by_model={
            "model-a": "shared stable answer",
            "model-c": "shared stable answer",
            "model-e": "shared stable answer",
        },
    )
    trace = _run(tmp_path, mode=CouncilMode.FAST_COUNCIL, adapter=adapter)

    assert trace.escalated_to_full is False
    assert _count_calls(adapter, "InitialAnswerPayload") == 3


def test_escalation_with_extra_models_unavailable_is_handled(tmp_path: Path) -> None:
    adapter = _ScriptedAdapter(
        initial_conf_by_model={"model-a": 0.2, "model-c": 0.2, "model-e": 0.2},
        outcomes={
            ("InitialAnswerPayload", "model-a"): _Outcome(
                status="timeout",
                error="Request timed out.",
            ),
            ("InitialAnswerPayload", "model-b"): _Outcome(
                status="unavailable",
                error="model unavailable",
            ),
            ("InitialAnswerPayload", "model-d"): _Outcome(
                status="timeout",
                error="Request timed out.",
            ),
        },
    )
    trace = _run(tmp_path, mode=CouncilMode.FAST_COUNCIL, adapter=adapter)

    assert trace.escalated_to_full is True
    assert trace.quorum_success is False
    assert any(f.seat_id == "seat-b" for f in trace.failures)
    assert any(f.seat_id == "seat-d" for f in trace.failures)


def test_chair_failure_uses_backup_chair(tmp_path: Path) -> None:
    adapter = _ScriptedAdapter(
        outcomes={
            ("FinalSynthesisPayload", "model-a"): _Outcome(
                status="timeout",
                error="Request timed out.",
            )
        }
    )
    trace = _run(tmp_path, mode=CouncilMode.FULL_COUNCIL, adapter=adapter)

    assert trace.final_synthesis is not None
    assert trace.final_synthesis.fallback_used is False
    assert trace.final_synthesis.chair_seat_id == "seat-c"
    assert any(
        f.stage == CouncilStage.SYNTHESIS and f.seat_id == "seat-a"
        for f in trace.failures
    )


def test_cache_hit_returns_consistent_transcript(tmp_path: Path) -> None:
    adapter = _ScriptedAdapter()
    first = _run(
        tmp_path,
        mode=CouncilMode.FAST_COUNCIL,
        adapter=adapter,
        demo_mode=True,
        query="cache me",
    )
    second = _run(
        tmp_path,
        mode=CouncilMode.FAST_COUNCIL,
        adapter=adapter,
        demo_mode=True,
        query="cache me",
    )

    assert first.cached is False
    assert second.cached is True
    assert second.observability.cache_status == CacheStatus.HIT
    assert len(first.transcript) == len(second.transcript)


def test_force_live_rerun_bypasses_cache(tmp_path: Path) -> None:
    adapter = _ScriptedAdapter()
    _run(
        tmp_path,
        mode=CouncilMode.FAST_COUNCIL,
        adapter=adapter,
        demo_mode=True,
        query="rerun me",
    )
    first_count = len(adapter.calls)
    rerun = _run(
        tmp_path,
        mode=CouncilMode.FAST_COUNCIL,
        adapter=adapter,
        demo_mode=True,
        force_live_rerun=True,
        query="rerun me",
    )

    assert rerun.cached is False
    assert rerun.observability.cache_status == CacheStatus.BYPASS
    assert len(adapter.calls) > first_count


def test_stale_cache_entry_is_ignored(tmp_path: Path) -> None:
    adapter = _ScriptedAdapter()
    cache_path = tmp_path / "cache.json"
    runtime = CouncilRuntime(config=_runtime_config(cache_path), adapter=adapter)
    request = CouncilRequest(
        query="stale cache check",
        mode=CouncilMode.FAST_COUNCIL,
        demo_mode=True,
    )
    cache_key = runtime._cache_key(request)
    cache_path.write_text(
        json.dumps(
            {
                cache_key: {
                    "schema_version": 1,
                    "trace": {"cached": True},
                }
            }
        ),
        encoding="utf-8",
    )

    trace = runtime.run(request)

    assert trace.cached is False
    assert trace.observability.cache_status == CacheStatus.MISS
    assert len(adapter.calls) > 0


def test_transcript_round_ordering_is_stable(tmp_path: Path) -> None:
    adapter = _ScriptedAdapter()
    trace = _run(tmp_path, mode=CouncilMode.FULL_COUNCIL, adapter=adapter)
    rank = {
        CouncilStage.INITIAL_ANSWER: 0,
        CouncilStage.PEER_CRITIQUE: 1,
        CouncilStage.REVISION: 2,
        CouncilStage.SYNTHESIS: 3,
    }
    sequence = [rank[item.stage] for item in trace.transcript]

    assert sequence == sorted(sequence)


def test_scorecards_remain_valid_when_one_model_partially_fails(tmp_path: Path) -> None:
    adapter = _ScriptedAdapter(
        outcomes={
            ("RevisedAnswerPayload", "model-c"): _Outcome(
                status="timeout",
                error="Request timed out.",
            )
        }
    )
    trace = _run(tmp_path, mode=CouncilMode.FULL_COUNCIL, adapter=adapter)

    assert len(trace.scorecards) == 5
    assert all(ModelScoreCard.model_validate(card) for card in trace.scorecards)
    seat_c = next(card for card in trace.scorecards if card.seat_id == "seat-c")
    assert seat_c.availability_status in {
        AvailabilityStatus.READY,
        AvailabilityStatus.DEGRADED,
        AvailabilityStatus.UNAVAILABLE,
    }


def test_contradiction_flag_emitted_for_divergent_revision(tmp_path: Path) -> None:
    adapter = _ScriptedAdapter(
        initial_answer_by_model={"model-a": "alpha alpha alpha stable response"},
        outcomes={
            ("RevisedAnswerPayload", "model-a"): _Outcome(
                payload=RevisedAnswerPayload(
                    revised_answer="zebra quantum nebula contradiction",
                    confidence=0.7,
                    grounding_confidence=0.7,
                    change_summary="major rewrite",
                )
            )
        },
    )
    trace = _run(tmp_path, mode=CouncilMode.FULL_COUNCIL, adapter=adapter)
    seat_a = next(card for card in trace.scorecards if card.seat_id == "seat-a")

    assert FailureFlag.CONTRADICTION in seat_a.failure_flags


def test_duplicate_failure_events_are_deduplicated(tmp_path: Path) -> None:
    adapter = _ScriptedAdapter()
    runtime = CouncilRuntime(config=_runtime_config(tmp_path / "cache.json"), adapter=adapter)
    trace = CouncilRunTrace(request=CouncilRequest(query="dedupe me"))
    seat = runtime.config.seats[0]
    result = RemoteInferenceResult(
        status="timeout",
        text="",
        latency_ms=10.0,
        model_id=seat.model_id,
        error="Request timed out.",
    )

    runtime._record_failure(
        trace=trace,
        stage=CouncilStage.INITIAL_ANSWER,
        seat=seat,
        result=result,
        parse_error=None,
    )
    runtime._record_failure(
        trace=trace,
        stage=CouncilStage.INITIAL_ANSWER,
        seat=seat,
        result=result,
        parse_error=None,
    )

    assert len(trace.failures) == 1


def test_final_output_remains_structured_when_intermediate_rounds_degrade(
    tmp_path: Path,
) -> None:
    adapter = _ScriptedAdapter(
        outcomes={
            ("PeerCritiquePayload", "model-c"): _Outcome(
                status="ok",
                parse_error="Malformed JSON output.",
            ),
            ("RevisedAnswerPayload", "model-e"): _Outcome(
                status="timeout",
                error="Request timed out.",
            ),
            ("FinalSynthesisPayload", "model-a"): _Outcome(
                status="timeout",
                error="Request timed out.",
            ),
            ("FinalSynthesisPayload", "model-c"): _Outcome(
                status="timeout",
                error="Request timed out.",
            ),
        }
    )
    trace = _run(tmp_path, mode=CouncilMode.FULL_COUNCIL, adapter=adapter)

    assert trace.final_synthesis is not None
    assert CouncilRunTrace.model_validate(trace.model_dump(mode="json"))


def test_scorecard_builder_failure_still_returns_valid_trace(tmp_path: Path, monkeypatch) -> None:
    adapter = _ScriptedAdapter()
    import rlrgf.council_runtime as runtime_module

    def _boom(**kwargs):
        _ = kwargs
        raise RuntimeError("scorecard explode")

    monkeypatch.setattr(runtime_module, "build_scorecards", _boom)
    trace = _run(tmp_path, mode=CouncilMode.FULL_COUNCIL, adapter=adapter)

    assert len(trace.scorecards) == 5
    assert any(
        f.stage == CouncilStage.RUNTIME and f.flag == FailureFlag.RUNTIME_ERROR
        for f in trace.failures
    )
