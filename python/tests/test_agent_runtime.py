from __future__ import annotations

import pytest

from rlrgf.agent_runtime import AgentRuntime
from rlrgf.council_runtime_config import CouncilRuntimeConfig
from rlrgf.council_runtime_schemas import (
    CacheStatus,
    CouncilMode,
    CouncilRequest,
    CouncilRunTrace,
    CouncilStage,
    ExecutionMode,
    FailureEvent,
    FailureFlag,
    FinalSynthesis,
)
from rlrgf.execution_state import (
    AgentAction,
    AgentExecutionState,
    TaskClassification,
    VerifierVerdict,
)
from rlrgf.planner import DeterministicPlanner, classify_task
from rlrgf.policies import RetryPolicy, StopPolicy
from rlrgf.run_council_runtime import _resolve_prompt_inputs, _runtime_banner, build_parser
from rlrgf.verifier import AnswerVerifier


def _make_trace(
    request: CouncilRequest,
    *,
    answer: str,
    disagreement: float = 0.1,
    confidence: float = 0.8,
    fallback_used: bool = False,
    failure_count: int = 0,
    quorum_success: bool = True,
) -> CouncilRunTrace:
    trace = CouncilRunTrace(request=request, quorum_success=quorum_success)
    trace.disagreement_score = disagreement
    trace.council_confidence = confidence
    trace.final_synthesis = FinalSynthesis(
        chair_seat_id="seat-a",
        chair_model_id="model-a",
        final_answer=answer,
        reasoning_summary="synthetic trace for tests",
        winner_seat_ids=["seat-a"],
        strongest_contributors=["seat-a"],
        uncertainty_notes=[],
        cited_risk_notes=[],
        confidence=confidence,
        fallback_used=fallback_used,
    )
    trace.observability.active_model_ids = ["model-a"]
    trace.observability.execution_mode = request.execution_mode
    trace.observability.requested_mode = request.mode
    trace.observability.effective_mode = request.mode
    trace.observability.cache_status = CacheStatus.BYPASS
    trace.failures = [
        FailureEvent(
            stage=CouncilStage.RUNTIME,
            flag=FailureFlag.RUNTIME_ERROR,
            detail="synthetic test failure",
        )
        for _ in range(failure_count)
    ]
    return trace


def test_classify_task_and_planner_choose_valid_action() -> None:
    planner = DeterministicPlanner()

    coding_state = AgentExecutionState(
        prompt="Write a Python function to reverse a list.",
        task_classification=classify_task("Write a Python function to reverse a list."),
    )
    retrieval_state = AgentExecutionState(
        prompt="Retrieve evidence for this grounded answer.",
        task_classification=TaskClassification.RETRIEVAL,
    )
    direct_state = AgentExecutionState(
        prompt="What is 2 + 2?",
        task_classification=TaskClassification.DIRECT,
    )

    assert coding_state.task_classification == TaskClassification.CODING
    assert planner.choose_next_action(coding_state) == AgentAction.GENERATE_INITIAL
    assert planner.choose_next_action(retrieval_state) == AgentAction.RETRIEVE_CONTEXT
    assert planner.choose_next_action(direct_state) == AgentAction.DIRECT_ANSWER


def test_verifier_returns_structured_revise_for_code_without_code() -> None:
    state = AgentExecutionState(
        prompt="Write a Python function that adds two numbers.",
        task_classification=TaskClassification.CODING,
        candidate_answer="Use Python to add the numbers together.",
    )

    result = AnswerVerifier().verify(state)

    assert result.verdict == VerifierVerdict.REVISE
    assert "does not contain code-like content" in result.rationale


def test_policies_enforce_bounds() -> None:
    state = AgentExecutionState(prompt="test", iteration_count=3, max_iterations=3)

    retry_policy = RetryPolicy(max_iterations=3, retrieval_retry_limit=1)
    stop_policy = StopPolicy(max_iterations=3)

    assert retry_policy.can_retry(state) is False
    assert stop_policy.should_stop(state) is True


def test_agent_runtime_halts_safely_and_records_steps() -> None:
    seen: list[CouncilRequest] = []

    def _executor(request: CouncilRequest) -> CouncilRunTrace:
        seen.append(request)
        return _make_trace(
            request,
            answer="final delegated answer",
            disagreement=0.12,
            confidence=0.84,
        )

    runtime = AgentRuntime(
        config=CouncilRuntimeConfig(agentic_enabled=True, execution_strategy="agentic"),
        executor=_executor,
    )

    state = runtime.run(
        "What release gates should block deployment?",
        mode=CouncilMode.FAST_COUNCIL,
        execution_mode=ExecutionMode.INTERACTIVE,
    )

    assert state.final_answer == "final delegated answer"
    assert state.stop_reason == "verification_pass"
    assert state.iteration_count <= runtime.config.max_iterations
    assert state.selected_execution_path[:2] == [
        AgentAction.DIRECT_ANSWER,
        AgentAction.VERIFY_ANSWER,
    ]
    assert state.telemetry[-1].action == AgentAction.STOP
    assert len(seen) == 1


def test_agent_runtime_retries_retrieval_within_budget() -> None:
    retrieval_attempts: list[int] = []

    def _retriever(prompt: str, attempt: int) -> list[str]:
        _ = prompt
        retrieval_attempts.append(attempt)
        if attempt == 1:
            return []
        return ["Evidence snippet"]

    def _executor(request: CouncilRequest) -> CouncilRunTrace:
        return _make_trace(
            request,
            answer="grounded delegated answer",
            disagreement=0.1,
            confidence=0.8,
        )

    runtime = AgentRuntime(
        config=CouncilRuntimeConfig(
            agentic_enabled=True,
            execution_strategy="agentic",
            max_iterations=3,
            retrieval_retry_limit=2,
        ),
        executor=_executor,
        retriever=_retriever,
    )

    state = runtime.run(
        "Retrieve evidence before answering this grounded question.",
        mode=CouncilMode.FAST_COUNCIL,
        execution_mode=ExecutionMode.INTERACTIVE,
    )

    assert state.retrieval_attempts == 1
    assert retrieval_attempts == [1]
    assert state.telemetry[0].action == AgentAction.RETRIEVE_CONTEXT


def test_agent_runtime_can_refine_after_verifier_request() -> None:
    calls = {"count": 0}

    def _executor(request: CouncilRequest) -> CouncilRunTrace:
        calls["count"] += 1
        if calls["count"] == 1:
            return _make_trace(
                request,
                answer="Use Python to count the items.",
                disagreement=0.18,
                confidence=0.74,
            )
        return _make_trace(
            request,
            answer="def top_k(nums, k):\n    return nums[:k]",
            disagreement=0.08,
            confidence=0.82,
        )

    runtime = AgentRuntime(
        config=CouncilRuntimeConfig(
            agentic_enabled=True,
            execution_strategy="agentic",
            max_iterations=3,
        ),
        executor=_executor,
    )

    state = runtime.run(
        "Write a Python function that returns the top k most frequent elements.",
        mode=CouncilMode.FAST_COUNCIL,
        execution_mode=ExecutionMode.INTERACTIVE,
    )

    assert calls["count"] == 2
    assert state.final_answer.startswith("def top_k")
    assert AgentAction.REFINE_ANSWER in state.selected_execution_path


def test_cli_defaults_to_standard_execution_strategy() -> None:
    args = build_parser().parse_args(["What is 2 + 2?"])

    assert args.execution_strategy == "standard"


def test_cli_accepts_builtin_prompt_set_without_query() -> None:
    args = build_parser().parse_args(["--prompt-set", "quick5"])
    prompts = _resolve_prompt_inputs(query="", prompt_set=args.prompt_set)

    assert len(prompts) == 5
    assert prompts[0]["category"] == "factual"
    assert prompts[2]["category"] == "numeric_direct"


def test_cli_rejects_query_and_prompt_set_together() -> None:
    try:
        _resolve_prompt_inputs(query="hello", prompt_set="quick5")
    except SystemExit as exc:
        assert "either a single query or --prompt-set" in str(exc)
    else:
        raise AssertionError("expected SystemExit when both query and prompt-set are provided")


def test_cli_parser_rejects_invalid_prompt_set() -> None:
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["--prompt-set", "unknown-pack"])

    assert excinfo.value.code == 2


def test_runtime_banner_format_is_stable() -> None:
    banner = _runtime_banner(
        adapter_name="MockCouncilInferenceAdapter",
        remote_requests=False,
        token_found=True,
        token_source="file",
        base_url="",
    )

    assert "adapter=MockCouncilInferenceAdapter" in banner
    assert "remote_requests=false" in banner
    assert "token_source=file" in banner
    assert "base_url=<none>" in banner
