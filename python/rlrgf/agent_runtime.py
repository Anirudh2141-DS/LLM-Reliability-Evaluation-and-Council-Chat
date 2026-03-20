from __future__ import annotations

from typing import Callable, Optional

from .council_runtime import CouncilRuntime
from .council_runtime_config import CouncilRuntimeConfig, load_runtime_config
from .council_runtime_schemas import CouncilMode, CouncilRequest, CouncilRunTrace, ExecutionMode
from .execution_state import (
    AgentAction,
    AgentExecutionState,
    ExecutionStepRecord,
    ExecutionStrategy,
)
from .planner import DeterministicPlanner, classify_task
from .policies import (
    RetryPolicy,
    StopPolicy,
    VerificationPolicy,
)
from .refiner import SafeRefiner
from .verifier import AnswerVerifier


CouncilExecutor = Callable[[CouncilRequest], CouncilRunTrace]
Retriever = Callable[[str, int], list[str]]


def _summarize_evidence(items: list[str]) -> str:
    cleaned = [item.strip() for item in items if item and item.strip()]
    if not cleaned:
        return ""
    return " | ".join(cleaned[:2])


def _estimate_trace_risk(trace: CouncilRunTrace) -> float:
    risk = (1.0 - float(trace.council_confidence)) * 0.45
    risk += float(trace.disagreement_score) * 0.35
    if trace.final_synthesis is not None and trace.final_synthesis.fallback_used:
        risk += 0.15
    risk += min(0.2, 0.05 * len(trace.failures))
    return max(0.0, min(1.0, round(risk, 2)))


class AgentRuntime:
    def __init__(
        self,
        *,
        config: Optional[CouncilRuntimeConfig] = None,
        planner: Optional[DeterministicPlanner] = None,
        verifier: Optional[AnswerVerifier] = None,
        refiner: Optional[SafeRefiner] = None,
        executor: Optional[CouncilExecutor] = None,
        retriever: Optional[Retriever] = None,
    ) -> None:
        self.config = config or load_runtime_config()
        self.planner = planner or DeterministicPlanner()
        self.verifier = verifier or AnswerVerifier()
        self.refiner = refiner or SafeRefiner()
        self._executor = executor
        self._retriever = retriever

    def run(
        self,
        prompt: str,
        *,
        mode: CouncilMode = CouncilMode.FAST_COUNCIL,
        execution_mode: ExecutionMode = ExecutionMode.INTERACTIVE,
    ) -> AgentExecutionState:
        state = AgentExecutionState(
            prompt=prompt,
            task_classification=classify_task(prompt),
            max_iterations=self.config.max_iterations,
            execution_strategy=ExecutionStrategy.AGENTIC,
        )
        retry_policy = RetryPolicy(
            max_iterations=self.config.max_iterations,
            retrieval_retry_limit=self.config.retrieval_retry_limit,
        )
        stop_policy = StopPolicy(max_iterations=self.config.max_iterations)
        verification_policy = VerificationPolicy(enabled=self.config.verification_enabled)

        while True:
            action = self.planner.choose_next_action(state)
            if action == AgentAction.STOP or stop_policy.should_stop(state):
                if not state.stop_reason:
                    state.stop_reason = "planner_stop"
                if not state.final_answer and state.candidate_answer:
                    state.final_answer = state.candidate_answer
                state.telemetry.append(
                    ExecutionStepRecord(
                        iteration=state.iteration_count,
                        action=AgentAction.STOP,
                        detail=state.stop_reason,
                        retrieval_attempts=state.retrieval_attempts,
                    )
                )
                break

            if action == AgentAction.RETRIEVE_CONTEXT:
                if not retry_policy.can_retry_retrieval(state):
                    state.stop_reason = "retrieval_budget_exhausted"
                    continue
                state.iteration_count += 1
                docs = self._retrieve_context(prompt, state.retrieval_attempts + 1)
                state.retrieval_attempts += 1
                state.retrieval_context = docs
                state.evidence_summary = _summarize_evidence(docs)
                state.evidence_coverage = 0.0 if not docs else min(1.0, 0.35 + 0.2 * len(docs))
                self._record_step(
                    state,
                    action,
                    "Retrieved supporting context." if docs else "No retriever configured or no context found.",
                )
                continue

            if not retry_policy.can_retry(state):
                state.stop_reason = "iteration_budget_exhausted"
                continue

            state.iteration_count += 1
            if action in {
                AgentAction.DIRECT_ANSWER,
                AgentAction.GENERATE_INITIAL,
                AgentAction.ESCALATE_DISAGREEMENT,
            }:
                runtime_mode = (
                    CouncilMode.FULL_COUNCIL
                    if action == AgentAction.ESCALATE_DISAGREEMENT
                    else mode
                )
                trace = self._execute_council(
                    query=prompt,
                    mode=runtime_mode,
                    execution_mode=execution_mode,
                )
                self._update_state_from_trace(state, trace)
                self._record_step(
                    state,
                    action,
                    f"Executed {runtime_mode.value} council run.",
                    metadata={
                        "quorum_success": trace.quorum_success,
                        "failure_count": len(trace.failures),
                    },
                )
                continue

            if action == AgentAction.VERIFY_ANSWER:
                if not verification_policy.should_verify(state):
                    state.stop_reason = "verification_disabled"
                    continue
                result = self.verifier.verify(state)
                state.verifier_results.append(result)
                if result.hallucination_risk is not None:
                    state.hallucination_risk = result.hallucination_risk
                if result.evidence_coverage is not None:
                    state.evidence_coverage = result.evidence_coverage
                self._record_step(
                    state,
                    action,
                    result.rationale,
                    verifier_verdict=result.verdict,
                )
                if result.verdict.value == "pass":
                    state.final_answer = state.candidate_answer
                    state.stop_reason = "verification_pass"
                continue

            if action == AgentAction.REFINE_ANSWER:
                guidance = (
                    state.verifier_results[-1].rationale
                    if state.verifier_results
                    else "Revise conservatively."
                )
                refined_prompt = self.refiner.build_refinement_prompt(state, guidance)
                trace = self._execute_council(
                    query=refined_prompt,
                    mode=mode,
                    execution_mode=execution_mode,
                )
                self._update_state_from_trace(state, trace)
                self._record_step(
                    state,
                    action,
                    "Refined candidate answer via delegated council execution.",
                    metadata={"failure_count": len(trace.failures)},
                )
                continue

            state.stop_reason = f"unsupported_action:{action.value}"

        return state

    def _execute_council(
        self,
        *,
        query: str,
        mode: CouncilMode,
        execution_mode: ExecutionMode,
    ) -> CouncilRunTrace:
        request = CouncilRequest(
            query=query,
            mode=mode,
            execution_mode=execution_mode,
            enable_revision_round=self.config.enable_revision_round,
            demo_mode=False,
            force_live_rerun=True,
        )
        if self._executor is not None:
            return self._executor(request)
        runtime = CouncilRuntime(config=self.config)
        return runtime.run(request)

    def _retrieve_context(self, prompt: str, attempt: int) -> list[str]:
        if self._retriever is None:
            return []
        return [item for item in self._retriever(prompt, attempt) if item]

    def _update_state_from_trace(
        self,
        state: AgentExecutionState,
        trace: CouncilRunTrace,
    ) -> None:
        state.active_models = list(trace.observability.active_model_ids)
        state.disagreement_score = float(trace.disagreement_score)
        state.confidence_signal = float(trace.council_confidence)
        state.hallucination_risk = _estimate_trace_risk(trace)
        if trace.final_synthesis is not None:
            state.candidate_answer = trace.final_synthesis.final_answer
            if trace.quorum_success:
                state.final_answer = trace.final_synthesis.final_answer

    def _record_step(
        self,
        state: AgentExecutionState,
        action: AgentAction,
        detail: str,
        *,
        verifier_verdict=None,
        metadata: Optional[dict[str, object]] = None,
    ) -> None:
        state.selected_execution_path.append(action)
        state.telemetry.append(
            ExecutionStepRecord(
                iteration=state.iteration_count,
                action=action,
                detail=detail,
                retrieval_attempts=state.retrieval_attempts,
                verifier_verdict=verifier_verdict,
                metadata=metadata or {},
            )
        )
