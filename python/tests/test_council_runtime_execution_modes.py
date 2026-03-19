from __future__ import annotations

import asyncio
from pathlib import Path
import time
from typing import Optional

from pydantic import BaseModel

from rlrgf.council_runtime import CouncilRuntime
from rlrgf.council_runtime_config import CouncilRuntimeConfig
from rlrgf.council_runtime_inference_adapter import RemoteInferenceResult
from rlrgf.council_runtime_schemas import (
    CouncilMode,
    CouncilRequest,
    CouncilSeat,
    CouncilStage,
    ExecutionMode,
    FinalSynthesisPayload,
    InitialAnswerPayload,
    PeerCritiquePayload,
    RevisedAnswerPayload,
)


class _RecordingAdapter:
    def __init__(
        self,
        *,
        initial_delay_s: float = 0.0,
        fail_initial_models: Optional[set[str]] = None,
    ) -> None:
        self.initial_delay_s = initial_delay_s
        self.fail_initial_models = fail_initial_models or set()
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

        if schema_model is InitialAnswerPayload:
            if self.initial_delay_s > 0:
                time.sleep(self.initial_delay_s)
            if model_id in self.fail_initial_models:
                raise RuntimeError("simulated initial failure")
            payload = InitialAnswerPayload(
                answer=f"answer from {model_id}",
                confidence=0.75,
                grounding_confidence=0.75,
                key_points=[],
                uncertainty_notes=[],
                cited_risks=[],
                citations=[],
            )
            return payload, self._ok(model_id), None

        if schema_model is PeerCritiquePayload:
            payload = PeerCritiquePayload(critiques=[], confidence=0.6)
            return payload, self._ok(model_id), None

        if schema_model is RevisedAnswerPayload:
            payload = RevisedAnswerPayload(
                revised_answer=f"revised from {model_id}",
                confidence=0.7,
                grounding_confidence=0.7,
                change_summary="minor",
            )
            return payload, self._ok(model_id), None

        if schema_model is FinalSynthesisPayload:
            payload = FinalSynthesisPayload(
                final_answer="final answer",
                reasoning_summary="summary",
                winner_seat_ids=["seat-a"],
                strongest_contributors=["seat-a"],
                uncertainty_notes=[],
                cited_risk_notes=[],
                confidence=0.7,
            )
            return payload, self._ok(model_id), None

        raise AssertionError(f"Unexpected schema model: {name}")

    def _ok(self, model_id: str) -> RemoteInferenceResult:
        return RemoteInferenceResult(
            status="ok",
            text='{"ok": true}',
            latency_ms=10.0,
            model_id=model_id,
        )


class _AsyncRecordingAdapter:
    def __init__(
        self,
        *,
        delays_s: Optional[dict[str, float]] = None,
        failures: Optional[set[tuple[str, str]]] = None,
    ) -> None:
        self.delays_s = delays_s or {}
        self.failures = failures or set()
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
        _ = (model_id, messages, schema_model, timeout_s, temperature, max_tokens)
        raise AssertionError("sync adapter path should not be used")

    async def call_json_async(
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
        delay_s = self.delays_s.get(name, 0.0)
        if delay_s > 0:
            await asyncio.sleep(delay_s)
        if (name, model_id) in self.failures:
            raise RuntimeError(f"simulated async failure for {name}:{model_id}")

        if schema_model is InitialAnswerPayload:
            payload = InitialAnswerPayload(
                answer=f"answer from {model_id}",
                confidence=0.75,
                grounding_confidence=0.75,
                key_points=[],
                uncertainty_notes=[],
                cited_risks=[],
                citations=[],
            )
            return payload, self._ok(model_id), None

        if schema_model is PeerCritiquePayload:
            payload = PeerCritiquePayload(critiques=[], confidence=0.6)
            return payload, self._ok(model_id), None

        if schema_model is RevisedAnswerPayload:
            payload = RevisedAnswerPayload(
                revised_answer=f"revised from {model_id}",
                confidence=0.7,
                grounding_confidence=0.7,
                change_summary="minor",
            )
            return payload, self._ok(model_id), None

        if schema_model is FinalSynthesisPayload:
            payload = FinalSynthesisPayload(
                final_answer="final answer",
                reasoning_summary="summary",
                winner_seat_ids=["seat-a"],
                strongest_contributors=["seat-a"],
                uncertainty_notes=[],
                cited_risk_notes=[],
                confidence=0.7,
            )
            return payload, self._ok(model_id), None

        raise AssertionError(f"Unexpected schema model: {name}")

    def _ok(self, model_id: str) -> RemoteInferenceResult:
        return RemoteInferenceResult(
            status="ok",
            text='{"ok": true}',
            latency_ms=10.0,
            model_id=model_id,
        )


def _config(tmp_path: Path) -> CouncilRuntimeConfig:
    return CouncilRuntimeConfig(
        base_url="http://unit-test.local",
        api_key="test-key",
        model_timeout_s=5.0,
        fast_quorum=2,
        full_quorum=3,
        chair_seat_id="seat-a",
        backup_chair_seat_id="seat-b",
        cache_path=str(tmp_path / "cache.json"),
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
                enabled_in_fast_mode=True,
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


def test_interactive_mode_uses_single_model_by_default(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cfg.interactive_enable_secondary_review = False
    adapter = _RecordingAdapter()
    runtime = CouncilRuntime(config=cfg, adapter=adapter)

    trace = runtime.run(
        CouncilRequest(
            query="interactive check",
            mode=CouncilMode.FULL_COUNCIL,
            execution_mode=ExecutionMode.INTERACTIVE,
            demo_mode=False,
        )
    )

    assert trace.observability.execution_mode == ExecutionMode.INTERACTIVE
    assert len(trace.active_seats) == 1
    assert trace.escalated_to_full is False
    assert trace.peer_critiques == []
    assert trace.quorum_success is True


def test_benchmark_mode_parallel_initial_fanout_handles_partial_failure(
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path)
    cfg.benchmark_enable_pairwise_critique = False
    cfg.benchmark_enable_summary_review = False
    adapter = _RecordingAdapter(initial_delay_s=0.12, fail_initial_models={"model-b"})
    runtime = CouncilRuntime(config=cfg, adapter=adapter)

    start = time.perf_counter()
    trace = runtime.run(
        CouncilRequest(
            query="benchmark parallel check",
            mode=CouncilMode.FULL_COUNCIL,
            execution_mode=ExecutionMode.BENCHMARK,
            demo_mode=False,
        )
    )
    elapsed = time.perf_counter() - start

    # Three initial calls at 0.12s each should complete close to single-call latency in parallel fan-out.
    assert elapsed < 0.30
    assert trace.final_synthesis is not None
    assert trace.observability.number_of_models_failed >= 1
    assert trace.observability.number_of_models_succeeded >= 1


def test_critique_is_gated_by_benchmark_flags(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cfg.benchmark_enable_pairwise_critique = False
    cfg.benchmark_enable_summary_review = False
    adapter = _RecordingAdapter()
    runtime = CouncilRuntime(config=cfg, adapter=adapter)

    trace = runtime.run(
        CouncilRequest(
            query="critique off",
            mode=CouncilMode.FULL_COUNCIL,
            execution_mode=ExecutionMode.BENCHMARK,
            demo_mode=False,
        )
    )

    assert trace.observability.critique_enabled is False
    assert trace.peer_critiques == []
    assert not any(name == "PeerCritiquePayload" for name, _ in adapter.calls)


def test_mode_observability_contains_requested_telemetry_fields(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cfg.benchmark_enable_pairwise_critique = True
    adapter = _RecordingAdapter()
    runtime = CouncilRuntime(config=cfg, adapter=adapter)

    trace = runtime.run(
        CouncilRequest(
            query="telemetry check",
            mode=CouncilMode.FAST_COUNCIL,
            execution_mode=ExecutionMode.BENCHMARK,
            demo_mode=False,
        )
    )

    obs = trace.observability
    assert obs.execution_mode == ExecutionMode.BENCHMARK
    assert obs.backend_type in {"mock", "remote"}
    assert obs.number_of_models_requested >= 1
    assert obs.number_of_models_succeeded >= 1
    assert obs.number_of_models_failed >= 0
    assert obs.request_wall_time_ms >= 0.0
    assert obs.total_latency_ms >= 0.0
    assert isinstance(obs.per_model_latency_ms, dict)
    assert obs.stage_latency_ms.get("initial_round", 0.0) >= 0.0
    assert obs.stage_latency_ms.get("aggregation", 0.0) >= 0.0


def test_async_adapter_parallelizes_pairwise_critique_and_revision(
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path)
    cfg.benchmark_enable_pairwise_critique = True
    cfg.benchmark_enable_summary_review = False
    adapter = _AsyncRecordingAdapter(
        delays_s={
            "InitialAnswerPayload": 0.12,
            "PeerCritiquePayload": 0.12,
            "RevisedAnswerPayload": 0.12,
            "FinalSynthesisPayload": 0.01,
        }
    )
    runtime = CouncilRuntime(config=cfg, adapter=adapter)

    start = time.perf_counter()
    trace = runtime.run(
        CouncilRequest(
            query="async benchmark path",
            mode=CouncilMode.FULL_COUNCIL,
            execution_mode=ExecutionMode.BENCHMARK,
            demo_mode=False,
        )
    )
    elapsed = time.perf_counter() - start

    assert elapsed < 0.55
    assert trace.final_synthesis is not None
    assert trace.observability.stage_latency_ms.get("critique", 0.0) < 250.0
    assert trace.observability.stage_latency_ms.get("revision", 0.0) < 250.0
    assert sum(1 for name, _ in adapter.calls if name == "PeerCritiquePayload") == 3
    assert sum(1 for name, _ in adapter.calls if name == "RevisedAnswerPayload") == 3


def test_async_adapter_partial_critique_failure_does_not_abort_benchmark(
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path)
    cfg.benchmark_enable_pairwise_critique = True
    cfg.benchmark_enable_summary_review = False
    adapter = _AsyncRecordingAdapter(
        failures={("PeerCritiquePayload", "model-b")}
    )
    runtime = CouncilRuntime(config=cfg, adapter=adapter)

    trace = runtime.run(
        CouncilRequest(
            query="async critique failure",
            mode=CouncilMode.FULL_COUNCIL,
            execution_mode=ExecutionMode.BENCHMARK,
            demo_mode=False,
        )
    )

    assert trace.final_synthesis is not None
    assert trace.quorum_success is True
    assert len(trace.peer_critiques) == 2
    assert any(failure.stage == CouncilStage.PEER_CRITIQUE for failure in trace.failures)


def test_critique_flag_is_false_when_benchmark_stage_is_skipped(
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path)
    cfg.benchmark_enable_pairwise_critique = False
    cfg.benchmark_enable_summary_review = True
    adapter = _RecordingAdapter(fail_initial_models={"model-b", "model-c"})
    runtime = CouncilRuntime(config=cfg, adapter=adapter)

    trace = runtime.run(
        CouncilRequest(
            query="insufficient survivors for critique",
            mode=CouncilMode.FULL_COUNCIL,
            execution_mode=ExecutionMode.BENCHMARK,
            demo_mode=False,
        )
    )

    assert trace.final_synthesis is not None
    assert trace.observability.critique_enabled is False
    assert trace.observability.stage_latency_ms.get("critique", 0.0) == 0.0
