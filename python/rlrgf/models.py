"""
RLRGF Data Models - Pydantic schemas for structured evaluation data.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


class Decision(str, Enum):
    ACCEPT = "accept"
    ABSTAIN = "abstain"
    ESCALATE = "escalate"
    ASK_CLARIFICATION = "ask_clarification"
    REQUEST_RETRIEVAL = "request_retrieval"


class EvaluatorRole(str, Enum):
    GENERATOR = "generator"
    GROUNDING_INSPECTOR = "grounding_inspector"
    SAFETY_AUDITOR = "safety_auditor"
    CRITIC = "critic"


class FailureType(str, Enum):
    HALLUCINATION = "hallucination"
    POLICY_VIOLATION = "policy_violation"
    LEAKAGE = "leakage"
    INJECTION_SUCCESS = "injection_success"
    INSTABILITY = "instability"
    MULTIPLE_FAILURES = "multiple_failures"


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class Citation(BaseModel):
    chunk_id: UUID
    source_text: str
    confidence: float


class ModelOutput(BaseModel):
    query_id: UUID
    generated_answer: str
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    citations: List[Citation] = Field(default_factory=list)
    generation_latency_ms: float = 0.0
    prompt_hash: str = ""


class RetrievedChunkRef(BaseModel):
    chunk_id: UUID
    doc_id: UUID
    similarity_score: float
    chunk_text: str
    chunk_index: int


class ClaimAssessment(BaseModel):
    claim_text: str
    supported: bool
    supporting_chunk_id: Optional[UUID] = None
    confidence: float


class CouncilVerdict(BaseModel):
    query_id: UUID
    evaluator_role: EvaluatorRole
    model_name: str = "default-model"
    grounding_score: float = 0.0
    confidence_score: float = 1.0
    safety_flag: bool = False
    leakage_risk: float = 0.0
    critic_score: float = 1.0
    claims: List[ClaimAssessment] = Field(default_factory=list)
    reasoning: str = ""


class CouncilAggregation(BaseModel):
    query_id: UUID
    verdicts: List[CouncilVerdict]
    final_decision: Decision
    aggregate_safety_flag: bool = False
    aggregate_grounding_score: float = 0.0
    disagreement_score: float = 0.0
    supported_claim_ratio: float = 0.0


class EvaluationRecord(BaseModel):
    eval_id: UUID = Field(default_factory=uuid4)
    query_id: UUID
    experiment_id: Optional[str] = None
    evaluator_model: str = "unknown"
    model_version: str = "1.0.0"
    failure_type: Optional[FailureType] = None
    category: str = "general"
    adversarial: bool = False
    verdicts: List[CouncilVerdict] = Field(default_factory=list)
    risk_score: float = 0.0
    sequence_id: int = 0
    decision: Decision
    supported_claim_ratio: float = 0.0
    policy_violation: bool = False
    hallucination_flag: bool = False
    leakage_detected: bool = False
    injection_success: bool = False
    instability_detected: bool = False
    retrieval_precision_at_k: Optional[float] = None
    retrieval_recall_at_k: Optional[float] = None
    citation_precision: Optional[float] = None
    generation_latency_ms: float = 0.0
    retrieval_latency_ms: Optional[float] = None
    context_token_count: int = 0
    timestamp: datetime = Field(default_factory=lambda: datetime.now())


class ExperimentMetrics(BaseModel):
    experiment_id: str
    timestamp: datetime
    total_queries: int = 0
    refusal_rate: float = 0.0
    hallucination_rate: float = 0.0
    policy_violation_rate: float = 0.0
    leakage_rate: float = 0.0
    avg_risk_score: float = 0.0
    injection_success_rate: float = 0.0
    instability_rate: float = 0.0
    avg_retrieval_precision_at_k: Optional[float] = None
    avg_retrieval_recall_at_k: Optional[float] = None
    avg_citation_precision: Optional[float] = None
    avg_supported_claim_ratio: float = 0.0
    council_disagreement_avg: float = 0.0
    safety_override_frequency: float = 0.0
    self_correction_rate: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    avg_context_length_tokens: float = 0.0
