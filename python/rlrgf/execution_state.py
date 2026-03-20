from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class ExecutionStrategy(str, Enum):
    STANDARD = "standard"
    AGENTIC = "agentic"


class TaskClassification(str, Enum):
    DIRECT = "direct"
    CODING = "coding"
    RETRIEVAL = "retrieval"
    ANALYTICAL = "analytical"
    SAFETY = "safety"
    UNKNOWN = "unknown"


class AgentAction(str, Enum):
    DIRECT_ANSWER = "direct_answer"
    RETRIEVE_CONTEXT = "retrieve_context"
    GENERATE_INITIAL = "generate_initial"
    VERIFY_ANSWER = "verify_answer"
    REFINE_ANSWER = "refine_answer"
    ESCALATE_DISAGREEMENT = "escalate_disagreement"
    STOP = "stop"


class VerifierVerdict(str, Enum):
    PASS = "pass"
    REVISE = "revise"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    HIGH_HALLUCINATION_RISK = "high_hallucination_risk"


class VerifierResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: VerifierVerdict
    rationale: str = ""
    evidence_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    conflict_score: float = Field(default=0.0, ge=0.0, le=1.0)
    hallucination_risk: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExecutionStepRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    iteration: int = Field(default=0, ge=0)
    action: AgentAction
    detail: str = ""
    retrieval_attempts: int = Field(default=0, ge=0)
    verifier_verdict: Optional[VerifierVerdict] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentExecutionState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)
    task_classification: TaskClassification = TaskClassification.UNKNOWN
    selected_execution_path: list[AgentAction] = Field(default_factory=list)
    active_models: list[str] = Field(default_factory=list)
    retrieval_attempts: int = Field(default=0, ge=0)
    iteration_count: int = Field(default=0, ge=0)
    evidence_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_summary: str = ""
    disagreement_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    hallucination_risk: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    confidence_signal: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    verifier_results: list[VerifierResult] = Field(default_factory=list)
    stop_reason: str = ""
    final_answer: str = ""
    candidate_answer: str = ""
    telemetry: list[ExecutionStepRecord] = Field(default_factory=list)
    retrieval_context: list[str] = Field(default_factory=list)
    max_iterations: int = Field(default=3, ge=1)
    execution_strategy: ExecutionStrategy = ExecutionStrategy.STANDARD
