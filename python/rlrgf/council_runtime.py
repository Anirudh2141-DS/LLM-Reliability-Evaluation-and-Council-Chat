"""
Live multi-round council runtime orchestration.
"""

from __future__ import annotations

import asyncio
from hashlib import sha256
import json
import logging
from pathlib import Path
from statistics import mean
from typing import Any, Optional

from .council_runtime_config import CouncilRuntimeConfig, load_runtime_config
from .council_runtime_inference_adapter import (
    CouncilInferenceAdapter,
    HuggingFaceRouterInferenceAdapter,
    MockCouncilInferenceAdapter,
    RemoteInferenceResult,
)
from .council_runtime_prompts import (
    build_initial_answer_messages,
    build_peer_critique_messages,
    build_revision_messages,
    build_synthesis_messages,
)
from .council_runtime_scoring import (
    build_scorecards,
    compute_council_confidence,
    compute_disagreement_score,
    quorum_success,
    rank_best_answer,
)
from .council_runtime_schemas import (
    AvailabilityStatus,
    CacheStatus,
    CouncilMode,
    CouncilRequest,
    CouncilRunTrace,
    CouncilSeat,
    CouncilStage,
    ExecutionMode,
    FailureEvent,
    FailureFlag,
    FinalSynthesis,
    FinalSynthesisPayload,
    InitialAnswer,
    InitialAnswerPayload,
    ModelScoreCard,
    PeerCritique,
    PeerCritiqueItem,
    PeerCritiquePayload,
    RevisedAnswer,
    RevisedAnswerPayload,
    RoundStats,
    RUNTIME_CONTRACT_VERSION,
    TranscriptEntry,
    TranscriptResult,
)


logger = logging.getLogger(__name__)

CACHE_SCHEMA_VERSION = 3


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class CouncilRuntime:
    """
    Executes a full council run against an OpenAI-compatible endpoint.
    """

    def __init__(
        self,
        *,
        config: Optional[CouncilRuntimeConfig] = None,
        adapter: Optional[CouncilInferenceAdapter] = None,
    ) -> None:
        self.config = config or load_runtime_config()
        self.adapter = adapter or self._build_default_adapter()
        self._log_adapter_selection()

    def run(self, request: CouncilRequest) -> CouncilRunTrace:
        cache_key = self._cache_key(request)

        if request.demo_mode and not request.force_live_rerun:
            cached = self._read_cached_trace(cache_key)
            if cached is not None:
                cached.cached = True
                cached.observability.cache_status = CacheStatus.HIT
                cached.observability.cache_key = cache_key
                return cached

        trace = CouncilRunTrace(
            contract_version=RUNTIME_CONTRACT_VERSION,
            request=request.model_copy(deep=True),
            cached=False,
        )
        trace.observability.cache_key = cache_key
        trace.observability.execution_mode = request.execution_mode
        trace.observability.requested_mode = request.mode
        trace.observability.backend_type = self._backend_type()
        if request.demo_mode:
            trace.observability.cache_status = (
                CacheStatus.BYPASS if request.force_live_rerun else CacheStatus.MISS
            )
        else:
            trace.observability.cache_status = CacheStatus.BYPASS

        # Keep mode handling explicit so interactive requests never drift into heavy benchmark logic.
        if request.execution_mode == ExecutionMode.INTERACTIVE:
            effective_mode = self._run_interactive(request=request, trace=trace)
        else:
            effective_mode = self._run_benchmark(request=request, trace=trace)
        trace.observability.effective_mode = effective_mode
        trace.observability.quorum_success = trace.quorum_success
        self._finalize_observability(trace)

        if request.demo_mode:
            self._write_cached_trace(cache_key, trace)
        return trace

    def _backend_type(self) -> str:
        return "remote" if not isinstance(self.adapter, MockCouncilInferenceAdapter) else "mock"

    def _run_interactive(
        self,
        *,
        request: CouncilRequest,
        trace: CouncilRunTrace,
    ) -> CouncilMode:
        selected_seats = self._interactive_seats(request)
        trace.active_seats = selected_seats
        self._sync_active_seat_observability(trace)

        initial_answers = self._run_initial_round(
            request=request,
            seats=trace.active_seats,
            trace=trace,
        )
        trace.initial_answers = initial_answers
        trace.escalated_to_full = False
        trace.observability.escalation_triggered = False

        # Interactive lane keeps critique cheap and bounded to at most one reviewer call.
        interactive_review_enabled = (
            self.config.interactive_enable_secondary_review
            and request.enable_revision_round
            and len(trace.active_seats) > 1
        )
        if interactive_review_enabled:
            trace.peer_critiques = self._run_summary_review(
                request=request,
                seats=trace.active_seats,
                initial_answers=trace.initial_answers,
                trace=trace,
            )
        else:
            trace.peer_critiques = []
        trace.observability.critique_enabled = bool(trace.peer_critiques)
        trace.revised_answers = []

        trace.scorecards = self._build_scorecards_safely(trace)
        trace.disagreement_score = compute_disagreement_score(trace.initial_answers)
        trace.council_confidence = compute_council_confidence(
            trace.scorecards, trace.disagreement_score
        )
        ready_model_count = self._usable_initial_contributor_count(
            trace=trace,
            seat_ids={seat.seat_id for seat in trace.active_seats},
        )
        # Interactive path must degrade gracefully and stay usable with one surviving model.
        trace.quorum_success = ready_model_count >= 1

        chair = self._select_chair(trace.active_seats, trace.scorecards)
        chair_candidates = self._chair_candidates(
            primary_chair=chair,
            seats=trace.active_seats,
            scorecards=trace.scorecards,
        )
        trace.final_synthesis = self._run_synthesis(
            request=request,
            chairs=chair_candidates,
            initial_answers=trace.initial_answers,
            critiques=trace.peer_critiques,
            revisions=trace.revised_answers,
            trace=trace,
        )
        trace.observability.fallback_used = (
            trace.final_synthesis.fallback_used if trace.final_synthesis is not None else True
        )
        return CouncilMode.FAST_COUNCIL

    def _run_benchmark(
        self,
        *,
        request: CouncilRequest,
        trace: CouncilRunTrace,
    ) -> CouncilMode:
        effective_mode = request.mode
        trace.active_seats = self.config.active_seats(effective_mode)
        self._sync_active_seat_observability(trace)

        initial_answers = self._run_initial_round_parallel(
            request=request,
            seats=trace.active_seats,
            trace=trace,
        )
        trace.initial_answers = initial_answers

        if request.mode == CouncilMode.FAST_COUNCIL:
            disagreement_score = compute_disagreement_score(initial_answers)
            bootstrap_confidence = self._bootstrap_confidence(initial_answers)
            escalation_reasons = self._fast_escalation_reasons(
                disagreement_score=disagreement_score,
                council_confidence=bootstrap_confidence,
                ready_model_count=self._usable_initial_contributor_count(
                    trace=trace,
                    seat_ids={seat.seat_id for seat in trace.active_seats},
                ),
            )
            if escalation_reasons:
                trace.escalated_to_full = True
                trace.observability.escalation_triggered = True
                trace.observability.escalation_reason = "; ".join(escalation_reasons)
                effective_mode = CouncilMode.FULL_COUNCIL
                full_seats = self.config.active_seats(CouncilMode.FULL_COUNCIL)
                seen_ids = {answer.seat_id for answer in initial_answers}
                missing = [seat for seat in full_seats if seat.seat_id not in seen_ids]
                if missing:
                    initial_answers.extend(
                        self._run_initial_round_parallel(
                            request=request,
                            seats=missing,
                            trace=trace,
                        )
                    )
                trace.initial_answers = initial_answers
                trace.active_seats = full_seats
                self._sync_active_seat_observability(trace)

        pairwise_critique_enabled = (
            request.enable_revision_round
            and self.config.enable_revision_round
            and self.config.benchmark_enable_pairwise_critique
        )
        summary_review_enabled = (
            request.enable_revision_round
            and self.config.enable_revision_round
            and not pairwise_critique_enabled
            and self.config.benchmark_enable_summary_review
        )
        trace.observability.critique_enabled = (
            pairwise_critique_enabled or summary_review_enabled
        )

        if pairwise_critique_enabled:
            trace.peer_critiques = self._run_peer_critiques(
                request=request,
                seats=trace.active_seats,
                initial_answers=trace.initial_answers,
                trace=trace,
            )
        elif summary_review_enabled:
            trace.peer_critiques = self._run_summary_review(
                request=request,
                seats=trace.active_seats,
                initial_answers=trace.initial_answers,
                trace=trace,
            )
        else:
            trace.peer_critiques = []

        if pairwise_critique_enabled:
            trace.revised_answers = self._run_revision_round(
                request=request,
                seats=trace.active_seats,
                initial_answers=trace.initial_answers,
                critiques=trace.peer_critiques,
                trace=trace,
            )
        else:
            trace.revised_answers = []

        trace.scorecards = self._build_scorecards_safely(trace)
        trace.disagreement_score = compute_disagreement_score(trace.initial_answers)
        trace.council_confidence = compute_council_confidence(
            trace.scorecards, trace.disagreement_score
        )
        ready_model_count = self._usable_initial_contributor_count(
            trace=trace,
            seat_ids={seat.seat_id for seat in trace.active_seats},
        )
        trace.quorum_success = quorum_success(
            config=self.config,
            mode=effective_mode.value,
            ready_model_count=ready_model_count,
        )

        chair = self._select_chair(trace.active_seats, trace.scorecards)
        chair_candidates = self._chair_candidates(
            primary_chair=chair,
            seats=trace.active_seats,
            scorecards=trace.scorecards,
        )
        trace.final_synthesis = self._run_synthesis(
            request=request,
            chairs=chair_candidates,
            initial_answers=trace.initial_answers,
            critiques=trace.peer_critiques,
            revisions=trace.revised_answers,
            trace=trace,
        )
        trace.observability.fallback_used = (
            trace.final_synthesis.fallback_used if trace.final_synthesis is not None else True
        )
        return effective_mode

    def _interactive_seats(self, request: CouncilRequest) -> list[CouncilSeat]:
        seats = self.config.active_seats(CouncilMode.FAST_COUNCIL)
        if not seats:
            seats = list(self.config.seats)
        if not seats:
            return []

        primary = self.config.get_seat(self.config.chair_seat_id) or seats[0]
        selected: list[CouncilSeat] = [primary]
        allow_secondary = (
            self.config.interactive_enable_secondary_review
            and request.enable_revision_round
            and self.config.interactive_max_models > 1
        )
        if not allow_secondary:
            return selected

        secondary = self.config.get_seat(self.config.backup_chair_seat_id)
        if secondary is None or secondary.seat_id == primary.seat_id:
            secondary = next((seat for seat in seats if seat.seat_id != primary.seat_id), None)
        if secondary is not None:
            selected.append(secondary)
        return selected[: self.config.interactive_max_models]

    def _run_initial_round_parallel(
        self,
        *,
        request: CouncilRequest,
        seats: list[CouncilSeat],
        trace: CouncilRunTrace,
    ) -> list[InitialAnswer]:
        if len(seats) <= 1:
            return self._run_initial_round(request=request, seats=seats, trace=trace)

        async def _invoke_for_seat(
            seat: CouncilSeat,
        ) -> tuple[
            CouncilSeat,
            list[dict[str, str]],
            Optional[InitialAnswerPayload],
            RemoteInferenceResult,
            Optional[str],
        ]:
            messages = build_initial_answer_messages(request, seat)
            try:
                payload, result, parse_error = await asyncio.to_thread(
                    self.adapter.call_json,
                    model_id=seat.model_id,
                    messages=messages,
                    schema_model=InitialAnswerPayload,
                    timeout_s=self.config.model_timeout_s,
                    temperature=0.25,
                    max_tokens=900,
                )
                return seat, messages, payload, result, parse_error
            except Exception as error:  # pragma: no cover - defensive branch
                result = RemoteInferenceResult(
                    status="error",
                    text="",
                    latency_ms=0.0,
                    model_id=seat.model_id,
                    error=str(error),
                    parse_mode="hard_failure",
                    parse_error_type="runtime_error",
                    failure_is_hard=True,
                    usable_contribution=False,
                    usable_for_quorum=False,
                )
                return seat, messages, None, result, str(error)

        async def _fan_out() -> list[
            tuple[
                CouncilSeat,
                list[dict[str, str]],
                Optional[InitialAnswerPayload],
                RemoteInferenceResult,
                Optional[str],
            ]
        ]:
            tasks = [_invoke_for_seat(seat) for seat in seats]
            gathered = await asyncio.gather(*tasks, return_exceptions=True)
            normalized: list[
                tuple[
                    CouncilSeat,
                    list[dict[str, str]],
                    Optional[InitialAnswerPayload],
                    RemoteInferenceResult,
                    Optional[str],
                ]
            ] = []
            for item in gathered:
                if isinstance(item, Exception):  # pragma: no cover - defensive branch
                    self._record_runtime_failure(
                        trace,
                        f"Parallel initial fan-out task failed: {item}",
                    )
                    continue
                normalized.append(item)
            return normalized

        outcomes = self._run_async(_fan_out())
        answers: list[InitialAnswer] = []
        attempted = len(seats)
        succeeded = 0
        failed = 0
        for seat, messages, payload, result, parse_error in outcomes:
            if payload is not None:
                result.failure_is_hard = False
                result.usable_contribution = True
                result.usable_for_quorum = True
            self._record_transcript(
                trace=trace,
                stage=CouncilStage.INITIAL_ANSWER,
                seat=seat,
                messages=messages,
                result=result,
                parse_error=parse_error,
            )
            if payload is None:
                failed += 1
                self._record_failure(
                    trace=trace,
                    stage=CouncilStage.INITIAL_ANSWER,
                    seat=seat,
                    result=result,
                    parse_error=parse_error,
                )
                continue
            succeeded += 1
            answers.append(
                InitialAnswer(
                    seat_id=seat.seat_id,
                    role_title=seat.role_title,
                    model_id=seat.model_id,
                    answer=payload.answer,
                    confidence=_clamp(payload.confidence),
                    grounding_confidence=_clamp(payload.grounding_confidence),
                    key_points=list(payload.key_points),
                    uncertainty_notes=list(payload.uncertainty_notes),
                    cited_risks=list(payload.cited_risks),
                    citations=list(payload.citations),
                    latency_ms=result.latency_ms,
                )
            )
        failed += max(0, attempted - (succeeded + failed))
        self._update_round_stats(
            trace=trace,
            stage=CouncilStage.INITIAL_ANSWER,
            attempted=attempted,
            succeeded=succeeded,
            failed=failed,
        )
        return answers

    def _run_summary_review(
        self,
        *,
        request: CouncilRequest,
        seats: list[CouncilSeat],
        initial_answers: list[InitialAnswer],
        trace: CouncilRunTrace,
    ) -> list[PeerCritique]:
        seat_by_id = {seat.seat_id: seat for seat in seats}
        answers_by_id = {answer.seat_id: answer for answer in initial_answers}
        if len(answers_by_id) < 2:
            self._update_round_stats(
                trace=trace,
                stage=CouncilStage.PEER_CRITIQUE,
                attempted=0,
                succeeded=0,
                failed=0,
            )
            return []

        primary_id = (
            self.config.chair_seat_id
            if self.config.chair_seat_id in answers_by_id
            else next(iter(answers_by_id.keys()))
        )
        reviewer_candidates = [self.config.backup_chair_seat_id] + [
            seat_id for seat_id in answers_by_id if seat_id != primary_id
        ]
        reviewer_seat: Optional[CouncilSeat] = None
        for candidate_id in reviewer_candidates:
            candidate = seat_by_id.get(candidate_id)
            if candidate is None:
                continue
            if candidate.seat_id not in answers_by_id:
                continue
            if candidate.seat_id == primary_id:
                continue
            reviewer_seat = candidate
            break
        if reviewer_seat is None:
            self._update_round_stats(
                trace=trace,
                stage=CouncilStage.PEER_CRITIQUE,
                attempted=0,
                succeeded=0,
                failed=0,
            )
            return []

        peer_answers = [
            answer.model_dump(mode="json")
            for answer in initial_answers
            if answer.seat_id != reviewer_seat.seat_id
        ]
        messages = build_peer_critique_messages(request, reviewer_seat, peer_answers)
        payload, result, parse_error = self.adapter.call_json(
            model_id=reviewer_seat.model_id,
            messages=messages,
            schema_model=PeerCritiquePayload,
            timeout_s=self.config.model_timeout_s,
            temperature=0.0,
            max_tokens=900,
        )
        self._record_transcript(
            trace=trace,
            stage=CouncilStage.PEER_CRITIQUE,
            seat=reviewer_seat,
            messages=messages,
            result=result,
            parse_error=parse_error,
        )
        if payload is None:
            self._record_failure(
                trace=trace,
                stage=CouncilStage.PEER_CRITIQUE,
                seat=reviewer_seat,
                result=result,
                parse_error=parse_error,
            )
            self._update_round_stats(
                trace=trace,
                stage=CouncilStage.PEER_CRITIQUE,
                attempted=1,
                succeeded=0,
                failed=1,
            )
            return []

        valid_target_ids = {
            answer.seat_id
            for answer in initial_answers
            if answer.seat_id != reviewer_seat.seat_id
        }
        filtered_items: list[PeerCritiqueItem] = [
            item for item in payload.critiques if item.target_seat_id in valid_target_ids
        ]
        best_answer_id = (
            payload.best_answer_seat_id
            if payload.best_answer_seat_id in valid_target_ids
            else None
        )
        weakest_answer_id = (
            payload.weakest_answer_seat_id
            if payload.weakest_answer_seat_id in valid_target_ids
            else None
        )
        critique = PeerCritique(
            critic_seat_id=reviewer_seat.seat_id,
            critic_role_title=reviewer_seat.role_title,
            model_id=reviewer_seat.model_id,
            critiques=filtered_items,
            best_answer_seat_id=best_answer_id,
            weakest_answer_seat_id=weakest_answer_id,
            confidence=_clamp(payload.confidence),
            latency_ms=result.latency_ms,
        )
        self._update_round_stats(
            trace=trace,
            stage=CouncilStage.PEER_CRITIQUE,
            attempted=1,
            succeeded=1,
            failed=0,
        )
        return [critique]

    def _finalize_observability(self, trace: CouncilRunTrace) -> None:
        seat_ids = {seat.seat_id for seat in trace.active_seats}
        requested = len(trace.active_seats)
        succeeded = self._usable_initial_contributor_count(trace=trace, seat_ids=seat_ids)
        failed = max(0, requested - succeeded)
        trace.observability.number_of_models_requested = requested
        trace.observability.number_of_models_succeeded = succeeded
        trace.observability.number_of_models_failed = failed

        per_model_latency: dict[str, float] = {}
        total_latency = 0.0
        for entry in trace.transcript:
            latency = max(0.0, float(entry.result.latency_ms or 0.0))
            total_latency += latency
            per_model_latency[entry.model_id] = (
                per_model_latency.get(entry.model_id, 0.0) + latency
            )
        trace.observability.total_latency_ms = round(total_latency, 1)
        trace.observability.per_model_latency_ms = {
            model_id: round(latency, 1)
            for model_id, latency in per_model_latency.items()
        }

    def _run_async(self, coroutine: Any) -> Any:
        try:
            return asyncio.run(coroutine)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coroutine)
            finally:
                loop.close()

    def _build_default_adapter(self) -> CouncilInferenceAdapter:
        if (
            self.config.default_execution_mode == ExecutionMode.INTERACTIVE
            and self.config.interactive_prefer_remote
            and not self.config.use_real_models
            and self._has_runtime_token()
        ):
            self.config.use_real_models = True
            if not self.config.base_url:
                self.config.base_url = self.config.hf_router_base_url

        if self.config.use_real_models:
            if not self._has_runtime_token():
                logger.warning(
                    "Real-model mode requested but token missing; falling back to mock adapter."
                )
                return MockCouncilInferenceAdapter()
            return HuggingFaceRouterInferenceAdapter(
                api_key=self.config.api_key,
                base_url=self.config.base_url or self.config.hf_router_base_url,
                request_timeout_s=self.config.request_timeout_s,
                max_retries=self.config.max_retries,
                retry_backoff_s=self.config.retry_backoff_s,
                retry_backoff_cap_s=self.config.retry_backoff_cap_s,
            )
        return MockCouncilInferenceAdapter()

    def _has_runtime_token(self) -> bool:
        return bool((self.config.api_key or "").strip())

    def _token_source(self, token_found: bool) -> str:
        source = str(getattr(self.config, "hf_token_source", "none")).strip().lower()
        if source in {"file", "environment"}:
            return source
        return "environment" if token_found else "none"

    def _log_adapter_selection(self) -> None:
        adapter_name = type(self.adapter).__name__
        remote_enabled = not isinstance(self.adapter, MockCouncilInferenceAdapter)
        base_url = getattr(self.adapter, "base_url", "")
        token_found = self._has_runtime_token()
        token_source = self._token_source(token_found)
        logger.info(
            "Council runtime adapter selected: %s | remote_requests=%s | token_found=%s | token_source=%s | use_real_models=%s | default_execution_mode=%s | base_url=%s",
            adapter_name,
            remote_enabled,
            token_found,
            token_source,
            self.config.use_real_models,
            self.config.default_execution_mode.value,
            base_url or "<none>",
        )
        if self.config.default_execution_mode == ExecutionMode.INTERACTIVE:
            logger.info(
                "Interactive lane flags: prefer_remote=%s | disable_cpu_fallback=%s | skip_heavy_probe=%s",
                self.config.interactive_prefer_remote,
                self.config.interactive_disable_cpu_fallback,
                self.config.interactive_skip_heavy_probe,
            )

    def _run_initial_round(
        self,
        *,
        request: CouncilRequest,
        seats: list[CouncilSeat],
        trace: CouncilRunTrace,
    ) -> list[InitialAnswer]:
        answers: list[InitialAnswer] = []
        attempted = 0
        succeeded = 0
        failed = 0
        for seat in seats:
            attempted += 1
            messages = build_initial_answer_messages(request, seat)
            payload, result, parse_error = self.adapter.call_json(
                model_id=seat.model_id,
                messages=messages,
                schema_model=InitialAnswerPayload,
                timeout_s=self.config.model_timeout_s,
                temperature=0.25,
                max_tokens=900,
            )
            if payload is not None:
                result.failure_is_hard = False
                result.usable_contribution = True
                result.usable_for_quorum = True
            self._record_transcript(
                trace=trace,
                stage=CouncilStage.INITIAL_ANSWER,
                seat=seat,
                messages=messages,
                result=result,
                parse_error=parse_error,
            )
            if payload is None:
                failed += 1
                self._record_failure(
                    trace=trace,
                    stage=CouncilStage.INITIAL_ANSWER,
                    seat=seat,
                    result=result,
                    parse_error=parse_error,
                )
                continue
            succeeded += 1
            answers.append(
                InitialAnswer(
                    seat_id=seat.seat_id,
                    role_title=seat.role_title,
                    model_id=seat.model_id,
                    answer=payload.answer,
                    confidence=_clamp(payload.confidence),
                    grounding_confidence=_clamp(payload.grounding_confidence),
                    key_points=list(payload.key_points),
                    uncertainty_notes=list(payload.uncertainty_notes),
                    cited_risks=list(payload.cited_risks),
                    citations=list(payload.citations),
                    latency_ms=result.latency_ms,
                )
            )
        self._update_round_stats(
            trace=trace,
            stage=CouncilStage.INITIAL_ANSWER,
            attempted=attempted,
            succeeded=succeeded,
            failed=failed,
        )
        return answers

    def _run_peer_critiques(
        self,
        *,
        request: CouncilRequest,
        seats: list[CouncilSeat],
        initial_answers: list[InitialAnswer],
        trace: CouncilRunTrace,
    ) -> list[PeerCritique]:
        answers_by_id = {answer.seat_id: answer for answer in initial_answers}
        critiques: list[PeerCritique] = []
        attempted = 0
        succeeded = 0
        failed = 0

        for seat in seats:
            if answers_by_id.get(seat.seat_id) is None:
                continue
            peer_answers = [
                answer.model_dump(mode="json")
                for answer in initial_answers
                if answer.seat_id != seat.seat_id
            ]
            if not peer_answers:
                continue
            attempted += 1

            messages = build_peer_critique_messages(request, seat, peer_answers)
            payload, result, parse_error = self.adapter.call_json(
                model_id=seat.model_id,
                messages=messages,
                schema_model=PeerCritiquePayload,
                timeout_s=self.config.model_timeout_s,
                temperature=0.0,
                max_tokens=1100,
            )
            self._record_transcript(
                trace=trace,
                stage=CouncilStage.PEER_CRITIQUE,
                seat=seat,
                messages=messages,
                result=result,
                parse_error=parse_error,
            )
            if payload is None:
                failed += 1
                self._record_failure(
                    trace=trace,
                    stage=CouncilStage.PEER_CRITIQUE,
                    seat=seat,
                    result=result,
                    parse_error=parse_error,
                )
                continue

            succeeded += 1
            valid_target_ids = {
                answer.seat_id for answer in initial_answers if answer.seat_id != seat.seat_id
            }
            filtered_items: list[PeerCritiqueItem] = [
                item
                for item in payload.critiques
                if item.target_seat_id in valid_target_ids
            ]
            best_answer_id = (
                payload.best_answer_seat_id
                if payload.best_answer_seat_id in valid_target_ids
                else None
            )
            weakest_answer_id = (
                payload.weakest_answer_seat_id
                if payload.weakest_answer_seat_id in valid_target_ids
                else None
            )

            critiques.append(
                PeerCritique(
                    critic_seat_id=seat.seat_id,
                    critic_role_title=seat.role_title,
                    model_id=seat.model_id,
                    critiques=filtered_items,
                    best_answer_seat_id=best_answer_id,
                    weakest_answer_seat_id=weakest_answer_id,
                    confidence=_clamp(payload.confidence),
                    latency_ms=result.latency_ms,
                )
            )
        self._update_round_stats(
            trace=trace,
            stage=CouncilStage.PEER_CRITIQUE,
            attempted=attempted,
            succeeded=succeeded,
            failed=failed,
        )
        return critiques

    def _run_revision_round(
        self,
        *,
        request: CouncilRequest,
        seats: list[CouncilSeat],
        initial_answers: list[InitialAnswer],
        critiques: list[PeerCritique],
        trace: CouncilRunTrace,
    ) -> list[RevisedAnswer]:
        answers_by_id = {answer.seat_id: answer for answer in initial_answers}
        revisions: list[RevisedAnswer] = []
        attempted = 0
        succeeded = 0
        failed = 0

        received_by_seat: dict[str, list[dict[str, Any]]] = {}
        for critique in critiques:
            for item in critique.critiques:
                received_by_seat.setdefault(item.target_seat_id, []).append(
                    {
                        "critic_seat_id": critique.critic_seat_id,
                        "critic_role_title": critique.critic_role_title,
                        **item.model_dump(mode="json"),
                    }
                )

        for seat in seats:
            original = answers_by_id.get(seat.seat_id)
            if original is None:
                continue
            attempted += 1

            messages = build_revision_messages(
                request=request,
                seat=seat,
                original_answer=original.model_dump(mode="json"),
                received_critiques=received_by_seat.get(seat.seat_id, []),
            )
            payload, result, parse_error = self.adapter.call_json(
                model_id=seat.model_id,
                messages=messages,
                schema_model=RevisedAnswerPayload,
                timeout_s=self.config.model_timeout_s,
                temperature=0.15,
                max_tokens=900,
            )
            self._record_transcript(
                trace=trace,
                stage=CouncilStage.REVISION,
                seat=seat,
                messages=messages,
                result=result,
                parse_error=parse_error,
            )
            if payload is None:
                failed += 1
                self._record_failure(
                    trace=trace,
                    stage=CouncilStage.REVISION,
                    seat=seat,
                    result=result,
                    parse_error=parse_error,
                )
                continue

            succeeded += 1
            revisions.append(
                RevisedAnswer(
                    seat_id=seat.seat_id,
                    role_title=seat.role_title,
                    model_id=seat.model_id,
                    revised_answer=payload.revised_answer,
                    confidence=_clamp(payload.confidence),
                    grounding_confidence=_clamp(payload.grounding_confidence),
                    change_summary=payload.change_summary,
                    latency_ms=result.latency_ms,
                )
            )
        self._update_round_stats(
            trace=trace,
            stage=CouncilStage.REVISION,
            attempted=attempted,
            succeeded=succeeded,
            failed=failed,
        )
        return revisions

    def _run_synthesis(
        self,
        *,
        request: CouncilRequest,
        chairs: list[CouncilSeat],
        initial_answers: list[InitialAnswer],
        critiques: list[PeerCritique],
        revisions: list[RevisedAnswer],
        trace: CouncilRunTrace,
    ) -> FinalSynthesis:
        if not initial_answers:
            return self._fallback_synthesis(
                trace=trace,
                chair=chairs[0] if chairs else None,
                reason="No initial answers available for synthesis.",
            )
        if not chairs:
            return self._fallback_synthesis(
                trace=trace,
                chair=None,
                reason="No chair candidate available for synthesis.",
            )

        attempted = 0
        succeeded = 0
        failed = 0
        seat_ids = {seat.seat_id for seat in trace.active_seats}
        for chair in chairs:
            attempted += 1
            messages = build_synthesis_messages(
                request=request,
                chair=chair,
                initial_answers=[item.model_dump(mode="json") for item in initial_answers],
                critiques=[item.model_dump(mode="json") for item in critiques],
                revisions=[item.model_dump(mode="json") for item in revisions],
                scorecards=[item.model_dump(mode="json") for item in trace.scorecards],
            )
            payload, result, parse_error = self.adapter.call_json(
                model_id=chair.model_id,
                messages=messages,
                schema_model=FinalSynthesisPayload,
                timeout_s=self.config.model_timeout_s,
                temperature=0.1,
                max_tokens=1300,
            )
            self._record_transcript(
                trace=trace,
                stage=CouncilStage.SYNTHESIS,
                seat=chair,
                messages=messages,
                result=result,
                parse_error=parse_error,
            )
            if payload is None:
                failed += 1
                self._record_failure(
                    trace=trace,
                    stage=CouncilStage.SYNTHESIS,
                    seat=chair,
                    result=result,
                    parse_error=parse_error,
                    override_flag=FailureFlag.SYNTHESIS_FAILURE,
                )
                continue

            succeeded += 1
            winner_ids = [
                seat_id
                for seat_id in dict.fromkeys(payload.winner_seat_ids)
                if seat_id in seat_ids
            ]
            contributor_ids = [
                seat_id
                for seat_id in dict.fromkeys(payload.strongest_contributors)
                if seat_id in seat_ids
            ]
            trace.observability.chair_selected_seat_id = chair.seat_id
            trace.observability.chair_selected_model_id = chair.model_id
            self._update_round_stats(
                trace=trace,
                stage=CouncilStage.SYNTHESIS,
                attempted=attempted,
                succeeded=succeeded,
                failed=failed,
            )
            return FinalSynthesis(
                chair_seat_id=chair.seat_id,
                chair_model_id=chair.model_id,
                final_answer=payload.final_answer,
                reasoning_summary=payload.reasoning_summary,
                winner_seat_ids=winner_ids,
                strongest_contributors=contributor_ids,
                uncertainty_notes=list(payload.uncertainty_notes),
                cited_risk_notes=list(payload.cited_risk_notes),
                confidence=_clamp(payload.confidence),
                fallback_used=False,
                latency_ms=result.latency_ms,
            )

        self._update_round_stats(
            trace=trace,
            stage=CouncilStage.SYNTHESIS,
            attempted=attempted,
            succeeded=succeeded,
            failed=failed,
        )
        return self._fallback_synthesis(
            trace=trace,
            chair=chairs[0],
            reason="All chair synthesis attempts failed.",
        )

    def _fallback_synthesis(
        self,
        *,
        trace: CouncilRunTrace,
        chair: Optional[CouncilSeat],
        reason: str,
    ) -> FinalSynthesis:
        initial_by_seat = {item.seat_id: item for item in trace.initial_answers}
        revised_by_seat = {item.seat_id: item for item in trace.revised_answers}
        best = rank_best_answer(trace.scorecards)
        if best is not None:
            revised = revised_by_seat.get(best.seat_id)
            initial = initial_by_seat.get(best.seat_id)
            final_answer = (
                revised.revised_answer
                if revised is not None
                else (initial.answer if initial is not None else "")
            )
            reasoning = (
                "Fallback synthesis selected the highest-ranked available seat answer."
            )
            winner_ids = [best.seat_id]
            confidence = _clamp(best.confidence)
        else:
            final_answer = (
                "Council could not produce a reliable synthesis. Please retry with full council."
            )
            reasoning = "Fallback synthesis had no available model outputs to select from."
            winner_ids = []
            confidence = 0.0

        if chair is not None:
            trace.observability.chair_selected_seat_id = chair.seat_id
            trace.observability.chair_selected_model_id = chair.model_id

        return FinalSynthesis(
            chair_seat_id=chair.seat_id if chair is not None else self.config.chair_seat_id,
            chair_model_id=chair.model_id if chair is not None else "",
            final_answer=final_answer,
            reasoning_summary=reasoning,
            winner_seat_ids=winner_ids,
            strongest_contributors=winner_ids,
            uncertainty_notes=[reason],
            cited_risk_notes=[],
            confidence=confidence,
            fallback_used=True,
            latency_ms=None,
        )

    def _record_transcript(
        self,
        *,
        trace: CouncilRunTrace,
        stage: CouncilStage,
        seat: CouncilSeat,
        messages: list[dict[str, str]],
        result: RemoteInferenceResult,
        parse_error: Optional[str],
    ) -> None:
        trace.transcript.append(
            TranscriptEntry(
                order=len(trace.transcript) + 1,
                stage=stage,
                seat_id=seat.seat_id,
                role_title=seat.role_title,
                model_id=seat.model_id,
                messages=messages,
                result=TranscriptResult(
                    status=result.status,
                    error=result.error,
                    parse_mode=result.parse_mode,
                    raw_output_present=result.raw_output_present,
                    recovered_output_used=result.recovered_output_used,
                    parse_error_type=result.parse_error_type,
                    failure_is_hard=result.failure_is_hard,
                    usable_contribution=result.usable_contribution,
                    usable_for_quorum=result.usable_for_quorum,
                    latency_ms=max(0.0, float(result.latency_ms)),
                    retry_count=max(0, int(result.retry_count)),
                    http_status=result.http_status,
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    total_tokens=result.total_tokens,
                    text=result.text or "",
                ),
                parse_error=parse_error.strip() if parse_error else None,
            )
        )

    def _record_failure(
        self,
        *,
        trace: CouncilRunTrace,
        stage: CouncilStage,
        seat: CouncilSeat,
        result: RemoteInferenceResult,
        parse_error: Optional[str],
        override_flag: Optional[FailureFlag] = None,
    ) -> None:
        flag = self._classify_failure_flag(
            result=result,
            parse_error=parse_error,
            override_flag=override_flag,
        )
        detail = (parse_error or result.error or "Unknown council failure.").strip()
        key = (stage, flag, seat.seat_id, seat.model_id, detail)
        for existing in trace.failures:
            if (
                existing.stage == key[0]
                and existing.flag == key[1]
                and existing.seat_id == key[2]
                and existing.model_id == key[3]
                and existing.detail == key[4]
            ):
                return

        trace.failures.append(
            FailureEvent(
                stage=stage,
                flag=flag,
                detail=detail,
                seat_id=seat.seat_id,
                model_id=seat.model_id,
                retry_count=max(0, result.retry_count),
            )
        )

    def _classify_failure_flag(
        self,
        *,
        result: RemoteInferenceResult,
        parse_error: Optional[str],
        override_flag: Optional[FailureFlag],
    ) -> FailureFlag:
        if override_flag is not None:
            return override_flag
        if parse_error:
            parse_error_type = (result.parse_error_type or "").strip().lower()
            if parse_error_type == "empty_response" or "empty response" in parse_error.lower():
                return FailureFlag.EMPTY_RESPONSE
            if parse_error_type == "timeout":
                return FailureFlag.TIMEOUT
            return FailureFlag.MALFORMED_JSON
        if result.status == "timeout":
            return FailureFlag.TIMEOUT
        if result.status == "unavailable":
            return FailureFlag.UNAVAILABLE_MODEL
        if result.status == "ok" and not result.text.strip():
            return FailureFlag.EMPTY_RESPONSE
        if result.status == "error":
            return FailureFlag.UNAVAILABLE_MODEL
        return FailureFlag.RUNTIME_ERROR

    def _record_runtime_failure(self, trace: CouncilRunTrace, detail: str) -> None:
        if any(
            item.stage == CouncilStage.RUNTIME
            and item.flag == FailureFlag.RUNTIME_ERROR
            and item.detail == detail
            for item in trace.failures
        ):
            return
        trace.failures.append(
            FailureEvent(
                stage=CouncilStage.RUNTIME,
                flag=FailureFlag.RUNTIME_ERROR,
                detail=detail,
            )
        )

    def _update_round_stats(
        self,
        *,
        trace: CouncilRunTrace,
        stage: CouncilStage,
        attempted: int,
        succeeded: int,
        failed: int,
    ) -> None:
        stats: Optional[RoundStats] = next(
            (item for item in trace.observability.round_stats if item.stage == stage),
            None,
        )
        if stats is None:
            stats = RoundStats(stage=stage)
            trace.observability.round_stats.append(stats)
        stats.attempted += attempted
        stats.succeeded += succeeded
        stats.failed += failed

    def _sync_active_seat_observability(self, trace: CouncilRunTrace) -> None:
        trace.observability.active_seat_ids = [seat.seat_id for seat in trace.active_seats]
        trace.observability.active_model_ids = [seat.model_id for seat in trace.active_seats]

    def _select_chair(
        self,
        seats: list[CouncilSeat],
        scorecards: list[ModelScoreCard],
    ) -> Optional[CouncilSeat]:
        seat_by_id = {seat.seat_id: seat for seat in seats if seat.can_chair}
        unavailable_ids = {
            card.seat_id
            for card in scorecards
            if card.availability_status == AvailabilityStatus.UNAVAILABLE
        }
        for seat_id in (self.config.chair_seat_id, self.config.backup_chair_seat_id):
            chair = seat_by_id.get(seat_id)
            if chair is not None and seat_id not in unavailable_ids:
                return chair

        best = rank_best_answer(scorecards)
        if best is not None:
            candidate = seat_by_id.get(best.seat_id)
            if candidate is not None:
                return candidate
        return next(iter(seat_by_id.values()), None)

    def _chair_candidates(
        self,
        *,
        primary_chair: Optional[CouncilSeat],
        seats: list[CouncilSeat],
        scorecards: list[ModelScoreCard],
    ) -> list[CouncilSeat]:
        seat_by_id = {seat.seat_id: seat for seat in seats if seat.can_chair}
        unavailable_ids = {
            card.seat_id
            for card in scorecards
            if card.availability_status == AvailabilityStatus.UNAVAILABLE
        }

        candidates: list[CouncilSeat] = []
        seen: set[str] = set()

        def add_candidate(seat: Optional[CouncilSeat]) -> None:
            if seat is None:
                return
            if seat.seat_id in seen:
                return
            if seat.seat_id in unavailable_ids:
                return
            seen.add(seat.seat_id)
            candidates.append(seat)

        add_candidate(primary_chair)
        add_candidate(seat_by_id.get(self.config.backup_chair_seat_id))
        best = rank_best_answer(scorecards)
        if best is not None:
            add_candidate(seat_by_id.get(best.seat_id))
        return candidates

    def _bootstrap_confidence(self, initial_answers: list[InitialAnswer]) -> float:
        if not initial_answers:
            return 0.0
        return _clamp(
            mean(
                _clamp((answer.confidence * 0.6) + (answer.grounding_confidence * 0.4))
                for answer in initial_answers
            )
        )

    def _usable_initial_contributor_count(
        self,
        *,
        trace: CouncilRunTrace,
        seat_ids: set[str],
    ) -> int:
        usable_seat_ids: set[str] = set()
        for entry in trace.transcript:
            if entry.stage != CouncilStage.INITIAL_ANSWER:
                continue
            if entry.seat_id not in seat_ids:
                continue
            if entry.result.failure_is_hard:
                continue
            if not entry.result.usable_for_quorum:
                continue
            usable_seat_ids.add(entry.seat_id)
        return len(usable_seat_ids)

    def _fast_escalation_reasons(
        self,
        *,
        disagreement_score: float,
        council_confidence: float,
        ready_model_count: int,
    ) -> list[str]:
        reasons: list[str] = []
        if ready_model_count < self.config.fast_quorum:
            reasons.append(
                f"ready_model_count({ready_model_count}) < fast_quorum({self.config.fast_quorum})"
            )
        if disagreement_score >= self.config.escalation_disagreement_threshold:
            reasons.append(
                "disagreement_score("
                f"{disagreement_score:.3f}) >= escalation_disagreement_threshold("
                f"{self.config.escalation_disagreement_threshold:.3f})"
            )
        if council_confidence <= self.config.escalation_confidence_threshold:
            reasons.append(
                "council_confidence("
                f"{council_confidence:.3f}) <= escalation_confidence_threshold("
                f"{self.config.escalation_confidence_threshold:.3f})"
            )
        return reasons

    def _build_scorecards_safely(self, trace: CouncilRunTrace) -> list[ModelScoreCard]:
        try:
            raw_cards = build_scorecards(
                seats=trace.active_seats,
                initial_answers=trace.initial_answers,
                peer_critiques=trace.peer_critiques,
                revised_answers=trace.revised_answers,
                failures=trace.failures,
            )
        except Exception as error:
            self._record_runtime_failure(
                trace,
                f"Scorecard build failed: {error}",
            )
            return self._fallback_scorecards(trace)

        cards_by_seat: dict[str, ModelScoreCard] = {}
        for card in raw_cards:
            try:
                validated = ModelScoreCard.model_validate(card)
            except Exception as error:
                self._record_runtime_failure(
                    trace,
                    f"Invalid scorecard for seat '{getattr(card, 'seat_id', 'unknown')}': {error}",
                )
                continue
            if validated.seat_id not in cards_by_seat:
                cards_by_seat[validated.seat_id] = validated

        normalized: list[ModelScoreCard] = []
        for seat in trace.active_seats:
            existing = cards_by_seat.get(seat.seat_id)
            if existing is not None:
                normalized.append(existing)
                continue
            seat_flags = list(
                dict.fromkeys(
                    failure.flag
                    for failure in trace.failures
                    if failure.seat_id == seat.seat_id
                )
            )
            if not seat_flags:
                seat_flags = [FailureFlag.RUNTIME_ERROR]
            availability = (
                AvailabilityStatus.UNAVAILABLE
                if FailureFlag.UNAVAILABLE_MODEL in seat_flags
                else AvailabilityStatus.DEGRADED
            )
            normalized.append(
                ModelScoreCard(
                    seat_id=seat.seat_id,
                    role_title=seat.role_title,
                    model_id=seat.model_id,
                    failure_flags=seat_flags,
                    availability_status=availability,
                )
            )
        return normalized

    def _fallback_scorecards(self, trace: CouncilRunTrace) -> list[ModelScoreCard]:
        cards: list[ModelScoreCard] = []
        for seat in trace.active_seats:
            seat_flags = list(
                dict.fromkeys(
                    failure.flag
                    for failure in trace.failures
                    if failure.seat_id == seat.seat_id
                )
            )
            if not seat_flags:
                seat_flags = [FailureFlag.RUNTIME_ERROR]
            availability = (
                AvailabilityStatus.UNAVAILABLE
                if FailureFlag.UNAVAILABLE_MODEL in seat_flags
                else AvailabilityStatus.DEGRADED
            )
            cards.append(
                ModelScoreCard(
                    seat_id=seat.seat_id,
                    role_title=seat.role_title,
                    model_id=seat.model_id,
                    failure_flags=seat_flags,
                    availability_status=availability,
                )
            )
        return cards

    def _cache_key(self, request: CouncilRequest) -> str:
        seat_state = [
            {
                "seat_id": seat.seat_id,
                "model_id": seat.model_id,
                "fast": seat.enabled_in_fast_mode,
            }
            for seat in self.config.seats
        ]
        payload = {
            "contract_version": RUNTIME_CONTRACT_VERSION,
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "query": request.query.strip(),
            "mode": request.mode.value,
            "execution_mode": request.execution_mode.value,
            "enable_revision_round": request.enable_revision_round,
            "seat_state": seat_state,
            "chair_seat_id": self.config.chair_seat_id,
            "backup_chair_seat_id": self.config.backup_chair_seat_id,
            "fast_quorum": self.config.fast_quorum,
            "full_quorum": self.config.full_quorum,
            "default_execution_mode": self.config.default_execution_mode.value,
            "benchmark_enable_pairwise_critique": self.config.benchmark_enable_pairwise_critique,
            "benchmark_enable_summary_review": self.config.benchmark_enable_summary_review,
            "interactive_max_models": self.config.interactive_max_models,
            "interactive_enable_secondary_review": self.config.interactive_enable_secondary_review,
            "escalation_disagreement_threshold": self.config.escalation_disagreement_threshold,
            "escalation_confidence_threshold": self.config.escalation_confidence_threshold,
            "base_url": self.config.base_url,
        }
        blob = json.dumps(payload, sort_keys=True)
        return sha256(blob.encode("utf-8")).hexdigest()

    def _read_cached_trace(self, cache_key: str) -> Optional[CouncilRunTrace]:
        path = Path(self.config.cache_path)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Failed to parse council cache from %s", path)
            return None
        if not isinstance(raw, dict):
            return None
        cache_entry = raw.get(cache_key)
        if not isinstance(cache_entry, dict):
            return None
        schema_version = cache_entry.get("schema_version")
        trace_payload = cache_entry.get("trace")
        if schema_version != CACHE_SCHEMA_VERSION or not isinstance(trace_payload, dict):
            return None
        try:
            cached = CouncilRunTrace.model_validate(trace_payload)
        except Exception:
            logger.warning("Failed to validate cached council trace for key %s", cache_key)
            return None
        if cached.contract_version != RUNTIME_CONTRACT_VERSION:
            return None
        return cached

    def _write_cached_trace(self, cache_key: str, trace: CouncilRunTrace) -> None:
        path = Path(self.config.cache_path)
        cache: dict[str, Any] = {}
        if path.exists():
            try:
                current = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(current, dict):
                    cache = current
            except Exception:
                logger.warning("Unable to read existing council cache at %s", path)
        cache[cache_key] = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "trace": trace.model_dump(mode="json"),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
