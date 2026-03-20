from __future__ import annotations

from dataclasses import dataclass

from .execution_state import AgentExecutionState, VerifierVerdict


@dataclass(frozen=True)
class RetryPolicy:
    max_iterations: int
    retrieval_retry_limit: int

    def can_retry(self, state: AgentExecutionState) -> bool:
        return state.iteration_count < self.max_iterations

    def can_retry_retrieval(self, state: AgentExecutionState) -> bool:
        return state.retrieval_attempts < self.retrieval_retry_limit


@dataclass(frozen=True)
class StopPolicy:
    max_iterations: int

    def should_stop(self, state: AgentExecutionState) -> bool:
        if state.stop_reason:
            return True
        if state.iteration_count >= self.max_iterations:
            return True
        if state.verifier_results and state.verifier_results[-1].verdict == VerifierVerdict.PASS:
            return True
        return False


@dataclass(frozen=True)
class VerificationPolicy:
    enabled: bool

    def should_verify(self, state: AgentExecutionState) -> bool:
        return self.enabled and bool(state.candidate_answer.strip())


@dataclass(frozen=True)
class DisagreementEscalationPolicy:
    threshold: float

    def should_escalate(self, state: AgentExecutionState) -> bool:
        if state.disagreement_score is None:
            return False
        return state.disagreement_score >= self.threshold
