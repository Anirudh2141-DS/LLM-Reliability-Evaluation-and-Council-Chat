from __future__ import annotations

from .execution_state import (
    AgentAction,
    AgentExecutionState,
    TaskClassification,
    VerifierVerdict,
)


_CODING_TERMS = ("code", "function", "python", "implement", "class", "script")
_RETRIEVAL_TERMS = ("retrieve", "citation", "source", "evidence", "grounded")
_ANALYTICAL_TERMS = ("design", "architecture", "plan", "tradeoff", "compare")
_SAFETY_TERMS = ("policy", "safety", "risk", "compliance", "security")


def classify_task(prompt: str) -> TaskClassification:
    normalized = " ".join((prompt or "").strip().lower().split())
    if not normalized:
        return TaskClassification.UNKNOWN
    if any(term in normalized for term in _CODING_TERMS):
        return TaskClassification.CODING
    if any(term in normalized for term in _RETRIEVAL_TERMS):
        return TaskClassification.RETRIEVAL
    if any(term in normalized for term in _SAFETY_TERMS):
        return TaskClassification.SAFETY
    if any(term in normalized for term in _ANALYTICAL_TERMS):
        return TaskClassification.ANALYTICAL
    return TaskClassification.DIRECT


class DeterministicPlanner:
    def choose_next_action(self, state: AgentExecutionState) -> AgentAction:
        if state.stop_reason:
            return AgentAction.STOP
        if state.iteration_count >= state.max_iterations:
            return AgentAction.STOP
        if not state.selected_execution_path:
            if state.task_classification == TaskClassification.DIRECT:
                return AgentAction.DIRECT_ANSWER
            if (
                state.task_classification in {TaskClassification.RETRIEVAL, TaskClassification.SAFETY}
                and not state.retrieval_context
            ):
                return AgentAction.RETRIEVE_CONTEXT
            return AgentAction.GENERATE_INITIAL
        if state.selected_execution_path[-1] in {
            AgentAction.DIRECT_ANSWER,
            AgentAction.GENERATE_INITIAL,
            AgentAction.REFINE_ANSWER,
            AgentAction.ESCALATE_DISAGREEMENT,
        }:
            return AgentAction.VERIFY_ANSWER
        if not state.candidate_answer.strip():
            return AgentAction.GENERATE_INITIAL
        if not state.verifier_results:
            return AgentAction.VERIFY_ANSWER

        verdict = state.verifier_results[-1].verdict
        if verdict == VerifierVerdict.PASS:
            return AgentAction.STOP
        if verdict == VerifierVerdict.INSUFFICIENT_EVIDENCE and not state.retrieval_context:
            return AgentAction.RETRIEVE_CONTEXT
        if verdict == VerifierVerdict.CONFLICTING_EVIDENCE and (
            state.disagreement_score is not None and state.disagreement_score >= 0.55
        ):
            if state.iteration_count < state.max_iterations:
                return AgentAction.ESCALATE_DISAGREEMENT
            return AgentAction.STOP
        if verdict in {
            VerifierVerdict.REVISE,
            VerifierVerdict.HIGH_HALLUCINATION_RISK,
            VerifierVerdict.CONFLICTING_EVIDENCE,
        }:
            if state.iteration_count < state.max_iterations:
                return AgentAction.REFINE_ANSWER
            return AgentAction.STOP
        return AgentAction.STOP
