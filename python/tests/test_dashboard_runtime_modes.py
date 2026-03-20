from __future__ import annotations

import pandas as pd

from rlrgf import dashboard
from rlrgf.council_runtime_config import CouncilRuntimeConfig
from rlrgf.council_runtime_schemas import (
    CouncilMode,
    CouncilRequest,
    CouncilRunTrace,
    CouncilSeat,
    CouncilStage,
    ExecutionMode,
    FailureEvent,
    FailureFlag,
    FinalSynthesis,
)


def _fake_state(inference_mode: str) -> dashboard.DashboardRuntimeState:
    return dashboard.DashboardRuntimeState(
        inference_mode=inference_mode,
        selection_reason="test",
        escalated=False,
        escalation_reason="",
        active_models=[],
        model_count=0,
        per_model=[],
        agreement_summary="",
        final_response_metadata={"final_answer": "ok", "quorum_success": True},
        degraded_quorum=False,
        fallback_used=False,
        total_latency_ms=0.0,
        observability={},
        round_stats=[],
    )


def test_benchmark_mode_forces_full_council_sequence(monkeypatch) -> None:
    calls: list[tuple[str, bool]] = []
    trace = CouncilRunTrace(request=CouncilRequest(query="benchmark route"))

    monkeypatch.setattr(
        dashboard,
        "_select_initial_mode",
        lambda prompt, route: (dashboard.INFERENCE_MODE_SOLO, "test"),
    )
    monkeypatch.setattr(
        dashboard,
        "_run_single_mode",
        lambda prompt, requested_mode, *, benchmark_mode: (
            calls.append((requested_mode, benchmark_mode)) or trace
        ),
    )
    monkeypatch.setattr(
        dashboard,
        "_dashboard_state_from_trace",
        lambda *args, **kwargs: _fake_state(dashboard.INFERENCE_MODE_FULL),
    )
    monkeypatch.setattr(dashboard, "_quality_gate", lambda *args, **kwargs: (True, ""))

    state, _ = dashboard.run_adaptive_council(
        "benchmark route",
        {},
        benchmark_mode=True,
    )

    assert state.inference_mode == dashboard.INFERENCE_MODE_FULL
    assert calls == [(dashboard.INFERENCE_MODE_FULL, True)]


def test_interactive_mode_stays_on_lightweight_sequence(monkeypatch) -> None:
    calls: list[tuple[str, bool]] = []
    trace = CouncilRunTrace(request=CouncilRequest(query="interactive route"))

    monkeypatch.setattr(
        dashboard,
        "_select_initial_mode",
        lambda prompt, route: (dashboard.INFERENCE_MODE_PARTIAL, "test"),
    )
    monkeypatch.setattr(
        dashboard,
        "_run_single_mode",
        lambda prompt, requested_mode, *, benchmark_mode: (
            calls.append((requested_mode, benchmark_mode)) or trace
        ),
    )
    monkeypatch.setattr(
        dashboard,
        "_dashboard_state_from_trace",
        lambda *args, **kwargs: _fake_state(dashboard.INFERENCE_MODE_PARTIAL),
    )
    monkeypatch.setattr(dashboard, "_quality_gate", lambda *args, **kwargs: (True, ""))

    state, _ = dashboard.run_adaptive_council(
        "interactive route",
        {},
        benchmark_mode=False,
    )

    assert state.inference_mode == dashboard.INFERENCE_MODE_PARTIAL
    assert calls == [(dashboard.INFERENCE_MODE_PARTIAL, False)]


def test_dashboard_state_marks_benchmark_unavailable_from_provider_collapse() -> None:
    seats = [
        CouncilSeat(
            seat_id=f"seat-{suffix}",
            role_title=f"Seat {suffix.upper()}",
            model_id=f"model-{suffix}",
            enabled_in_fast_mode=suffix in {"a", "c", "e"},
            can_chair=True,
        )
        for suffix in ("a", "b", "c", "d", "e")
    ]
    trace = CouncilRunTrace(
        request=CouncilRequest(
            query="benchmark unavailable",
            mode=CouncilMode.FULL_COUNCIL,
            execution_mode=ExecutionMode.BENCHMARK,
        ),
        active_seats=seats,
        final_synthesis=FinalSynthesis(
            chair_seat_id="seat-a",
            chair_model_id="model-a",
            final_answer=(
                "Benchmark execution could not complete because no council model seats "
                "were available from the configured provider. Check provider availability "
                "or credits and retry."
            ),
            reasoning_summary="No provider-backed council seats were available.",
            winner_seat_ids=[],
            strongest_contributors=[],
            uncertainty_notes=[],
            cited_risk_notes=[],
            confidence=0.0,
            fallback_used=True,
        ),
        failures=[
            FailureEvent(
                stage=CouncilStage.INITIAL_ANSWER,
                flag=FailureFlag.UNAVAILABLE_MODEL,
                detail="HTTP 402: Provider credits exhausted.",
                seat_id=seat.seat_id,
                model_id=seat.model_id,
            )
            for seat in seats
        ],
        quorum_success=False,
    )
    trace.observability.execution_mode = ExecutionMode.BENCHMARK
    trace.observability.requested_mode = CouncilMode.FULL_COUNCIL
    trace.observability.effective_mode = CouncilMode.FULL_COUNCIL
    trace.observability.backend_type = "remote"
    trace.observability.number_of_models_requested = 5
    trace.observability.number_of_models_succeeded = 0
    trace.observability.number_of_models_failed = 5
    trace.observability.critique_enabled = False
    trace.observability.quorum_success = False
    trace.observability.stage_latency_ms = {"initial_round": 321.0}

    state = dashboard._dashboard_state_from_trace(
        trace,
        requested_mode=dashboard.INFERENCE_MODE_FULL,
        selection_reason="test",
        escalated=False,
        escalation_reason="",
    )

    assert state.final_response_metadata["benchmark_unavailable"] is True
    assert state.final_response_metadata["critique_enabled"] is False
    assert state.final_response_metadata["number_of_models_requested"] == 5
    assert state.final_response_metadata["number_of_models_succeeded"] == 0
    assert state.final_response_metadata["number_of_models_failed"] == 5
    assert "no council seats were available" in state.agreement_summary.lower()
    assert state.per_model[0].diagnostic_summary == (
        "Model was unavailable from the configured provider."
    )


def test_dashboard_run_single_mode_stays_on_council_runtime(monkeypatch) -> None:
    seen: dict[str, object] = {}

    class _StubStreamlit:
        def __init__(self) -> None:
            self.session_state: dict[str, object] = {}

    class _FakeCouncilRuntime:
        def __init__(self, *, config) -> None:
            seen["runtime"] = "council"
            seen["config"] = config

        def run(self, request: CouncilRequest) -> CouncilRunTrace:
            seen["request"] = request
            return CouncilRunTrace(request=request)

    monkeypatch.setattr(dashboard, "st", _StubStreamlit())
    monkeypatch.setattr(dashboard, "load_runtime_config", lambda: CouncilRuntimeConfig())
    monkeypatch.setattr(dashboard, "CouncilRuntime", _FakeCouncilRuntime)

    trace = dashboard._run_single_mode(
        "interactive route",
        dashboard.INFERENCE_MODE_SOLO,
        benchmark_mode=False,
    )

    assert isinstance(trace, CouncilRunTrace)
    assert seen["runtime"] == "council"
    assert seen["request"].execution_mode == ExecutionMode.INTERACTIVE



def test_prepare_evaluation_dashboard_data_summarizes_models_and_categories() -> None:
    df = pd.DataFrame(
        [
            {
                "evaluator_model": "model-a",
                "query_id": "q1",
                "category": "normal",
                "supported_claim_ratio": 0.95,
                "risk_score": 0.10,
                "generation_latency_ms": 120.0,
                "retrieval_latency_ms": 30.0,
                "decision": "accept",
                "failure_type": None,
                "policy_violation": False,
                "instability_detected": False,
            },
            {
                "evaluator_model": "model-b",
                "query_id": "q1",
                "category": "ambiguous",
                "supported_claim_ratio": 0.40,
                "risk_score": 0.60,
                "generation_latency_ms": 260.0,
                "retrieval_latency_ms": 40.0,
                "decision": "escalate",
                "failure_type": "hallucination",
                "policy_violation": True,
                "instability_detected": True,
            },
            {
                "evaluator_model": "model-a",
                "query_id": "q2",
                "category": "hallucination_bait",
                "supported_claim_ratio": 0.88,
                "risk_score": 0.12,
                "generation_latency_ms": 140.0,
                "retrieval_latency_ms": 20.0,
                "decision": "accept",
                "failure_type": None,
                "policy_violation": False,
                "instability_detected": False,
            },
        ]
    )

    prepared = dashboard.prepare_evaluation_dashboard_data(df)

    assert prepared["overview"]["model_count"] == 2
    assert prepared["overview"]["prompt_count"] == 2
    assert prepared["overview"]["best_model"] == "model-a"
    assert set(prepared["category_summary"]["prompt_category"].tolist()) == {
        "Normal",
        "Tradeoff / Ambiguity",
        "Hallucination Bait",
    }
    assert prepared["leaderboard"].iloc[0]["evaluator_model"] == "model-a"
    assert "suite_score" in prepared["aggregate_metrics"]["metric"].tolist()
