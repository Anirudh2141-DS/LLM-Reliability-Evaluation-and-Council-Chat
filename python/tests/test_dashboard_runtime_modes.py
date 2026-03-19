from __future__ import annotations

from rlrgf import dashboard
from rlrgf.council_runtime_schemas import CouncilRequest, CouncilRunTrace


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
