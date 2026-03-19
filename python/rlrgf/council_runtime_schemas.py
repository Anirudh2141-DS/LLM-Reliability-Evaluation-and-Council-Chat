"""
Typed schemas for live council orchestration.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


RUNTIME_CONTRACT_VERSION = "council_runtime_v1"


class CouncilMode(str, Enum):
    FULL_COUNCIL = "full_council"
    FAST_COUNCIL = "fast_council"


class ExecutionMode(str, Enum):
    INTERACTIVE = "interactive"
    BENCHMARK = "benchmark"


class AvailabilityStatus(str, Enum):
    READY = "ready"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class FailureFlag(str, Enum):
    TIMEOUT = "timeout"
    MALFORMED_JSON = "malformed_json"
    EMPTY_RESPONSE = "empty_response"
    UNAVAILABLE_MODEL = "unavailable_model"
    CONTRADICTION = "contradiction"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    WEAK_CRITIQUE = "weak_critique"
    SYNTHESIS_FAILURE = "synthesis_failure"
    RUNTIME_ERROR = "runtime_error"


class CouncilStage(str, Enum):
    INITIAL_ANSWER = "initial_answer"
    PEER_CRITIQUE = "peer_critique"
    REVISION = "revision"
    SYNTHESIS = "synthesis"
    RUNTIME = "runtime"


class CacheStatus(str, Enum):
    MISS = "miss"
    HIT = "hit"
    BYPASS = "bypass"


class CouncilRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: UUID = Field(default_factory=uuid4)
    query: str
    mode: CouncilMode = CouncilMode.FAST_COUNCIL
    execution_mode: ExecutionMode = ExecutionMode.BENCHMARK
    enable_revision_round: bool = True
    demo_mode: bool = True
    force_live_rerun: bool = False
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("query")
    @classmethod
    def _normalize_query(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("query must not be empty")
        return cleaned


class CouncilSeat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seat_id: str
    role_title: str
    model_id: str
    enabled_in_fast_mode: bool = False
    can_chair: bool = True


class InitialAnswerPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    answer: str = Field(min_length=1)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    grounding_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    key_points: list[str] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)
    cited_risks: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)

    @field_validator("answer")
    @classmethod
    def _normalize_answer(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("answer must not be empty")
        return cleaned


class InitialAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seat_id: str
    role_title: str
    model_id: str
    answer: str = Field(min_length=1)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    grounding_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    key_points: list[str] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)
    cited_risks: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    latency_ms: Optional[float] = Field(default=None, ge=0.0)

    @field_validator("answer")
    @classmethod
    def _normalize_answer(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("answer must not be empty")
        return cleaned


class PeerCritiqueItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    target_seat_id: str
    correctness: float = Field(default=0.5, ge=0.0, le=1.0)
    completeness: float = Field(default=0.5, ge=0.0, le=1.0)
    risk: float = Field(default=0.5, ge=0.0, le=1.0)
    unsupported_claims: float = Field(default=0.5, ge=0.0, le=1.0)
    clarity: float = Field(default=0.5, ge=0.0, le=1.0)
    comments: str = ""
    suggested_fix: str = ""


class PeerCritiquePayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    critiques: list[PeerCritiqueItem] = Field(default_factory=list)
    best_answer_seat_id: Optional[str] = None
    weakest_answer_seat_id: Optional[str] = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class PeerCritique(BaseModel):
    model_config = ConfigDict(extra="forbid")

    critic_seat_id: str
    critic_role_title: str
    model_id: str
    critiques: list[PeerCritiqueItem] = Field(default_factory=list)
    best_answer_seat_id: Optional[str] = None
    weakest_answer_seat_id: Optional[str] = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    latency_ms: Optional[float] = Field(default=None, ge=0.0)


class RevisedAnswerPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    revised_answer: str = Field(min_length=1)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    grounding_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    change_summary: str = ""

    @field_validator("revised_answer")
    @classmethod
    def _normalize_revised_answer(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("revised_answer must not be empty")
        return cleaned


class RevisedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seat_id: str
    role_title: str
    model_id: str
    revised_answer: str = Field(min_length=1)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    grounding_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    change_summary: str = ""
    latency_ms: Optional[float] = Field(default=None, ge=0.0)

    @field_validator("revised_answer")
    @classmethod
    def _normalize_revised_answer(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("revised_answer must not be empty")
        return cleaned


class FinalSynthesisPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    final_answer: str = Field(min_length=1)
    reasoning_summary: str = Field(min_length=1)
    winner_seat_ids: list[str] = Field(default_factory=list)
    strongest_contributors: list[str] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)
    cited_risk_notes: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("final_answer", "reasoning_summary")
    @classmethod
    def _normalize_non_empty_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("text field must not be empty")
        return cleaned


class FinalSynthesis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chair_seat_id: str
    chair_model_id: str
    final_answer: str = Field(min_length=1)
    reasoning_summary: str = Field(min_length=1)
    winner_seat_ids: list[str] = Field(default_factory=list)
    strongest_contributors: list[str] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)
    cited_risk_notes: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    fallback_used: bool = False
    latency_ms: Optional[float] = Field(default=None, ge=0.0)

    @field_validator("final_answer", "reasoning_summary")
    @classmethod
    def _normalize_non_empty_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("text field must not be empty")
        return cleaned


class ModelScoreCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seat_id: str
    role_title: str
    model_id: str
    answer_quality: float = Field(default=0.0, ge=0.0, le=1.0)
    critique_quality: float = Field(default=0.0, ge=0.0, le=1.0)
    grounding_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    consistency: float = Field(default=0.0, ge=0.0, le=1.0)
    latency_ms: Optional[float] = Field(default=None, ge=0.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    failure_flags: list[FailureFlag] = Field(default_factory=list)
    availability_status: AvailabilityStatus = AvailabilityStatus.READY


class FailureEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: CouncilStage
    flag: FailureFlag
    detail: str
    seat_id: Optional[str] = None
    model_id: Optional[str] = None
    retry_count: int = Field(default=0, ge=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("detail")
    @classmethod
    def _normalize_detail(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            return "Unknown council failure."
        return cleaned


class RoundStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: CouncilStage
    attempted: int = Field(default=0, ge=0)
    succeeded: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)


class TranscriptResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    error: Optional[str] = None
    parse_mode: str = "clean_json"
    raw_output_present: bool = False
    recovered_output_used: bool = False
    parse_error_type: Optional[str] = None
    failure_is_hard: bool = False
    usable_contribution: bool = False
    usable_for_quorum: bool = False
    latency_ms: float = Field(default=0.0, ge=0.0)
    retry_count: int = Field(default=0, ge=0)
    http_status: Optional[int] = None
    prompt_tokens: Optional[int] = Field(default=None, ge=0)
    completion_tokens: Optional[int] = Field(default=None, ge=0)
    total_tokens: Optional[int] = Field(default=None, ge=0)
    text: str = ""


class TranscriptEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order: int = Field(ge=1)
    stage: CouncilStage
    seat_id: str
    role_title: str
    model_id: str
    messages: list[dict[str, str]] = Field(default_factory=list)
    result: TranscriptResult
    parse_error: Optional[str] = None


class RuntimeObservability(BaseModel):
    model_config = ConfigDict(extra="forbid")

    execution_mode: ExecutionMode = ExecutionMode.BENCHMARK
    requested_mode: CouncilMode = CouncilMode.FAST_COUNCIL
    effective_mode: CouncilMode = CouncilMode.FAST_COUNCIL
    active_seat_ids: list[str] = Field(default_factory=list)
    active_model_ids: list[str] = Field(default_factory=list)
    number_of_models_requested: int = Field(default=0, ge=0)
    number_of_models_succeeded: int = Field(default=0, ge=0)
    number_of_models_failed: int = Field(default=0, ge=0)
    critique_enabled: bool = False
    backend_type: str = "mock"
    request_wall_time_ms: float = Field(default=0.0, ge=0.0)
    total_latency_ms: float = Field(default=0.0, ge=0.0)
    per_model_latency_ms: dict[str, float] = Field(default_factory=dict)
    stage_latency_ms: dict[str, float] = Field(default_factory=dict)
    escalation_triggered: bool = False
    escalation_reason: Optional[str] = None
    chair_selected_seat_id: Optional[str] = None
    chair_selected_model_id: Optional[str] = None
    fallback_used: bool = False
    cache_status: CacheStatus = CacheStatus.MISS
    cache_key: Optional[str] = None
    quorum_success: bool = False
    round_stats: list[RoundStats] = Field(default_factory=list)


class CouncilRunTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: str = RUNTIME_CONTRACT_VERSION
    request: CouncilRequest
    active_seats: list[CouncilSeat] = Field(default_factory=list)
    escalated_to_full: bool = False
    disagreement_score: float = Field(default=0.0, ge=0.0, le=1.0)
    council_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    quorum_success: bool = False
    initial_answers: list[InitialAnswer] = Field(default_factory=list)
    peer_critiques: list[PeerCritique] = Field(default_factory=list)
    revised_answers: list[RevisedAnswer] = Field(default_factory=list)
    final_synthesis: Optional[FinalSynthesis] = None
    scorecards: list[ModelScoreCard] = Field(default_factory=list)
    failures: list[FailureEvent] = Field(default_factory=list)
    transcript: list[TranscriptEntry] = Field(default_factory=list)
    cached: bool = False
    observability: RuntimeObservability = Field(default_factory=RuntimeObservability)
