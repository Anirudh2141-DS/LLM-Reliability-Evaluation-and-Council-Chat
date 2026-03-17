"""
Scoring utilities for council runs.
"""

from __future__ import annotations

import math
import re
from statistics import mean
from typing import Optional

from .council_runtime_config import CouncilRuntimeConfig
from .council_runtime_schemas import (
    AvailabilityStatus,
    CouncilSeat,
    FailureEvent,
    FailureFlag,
    InitialAnswer,
    ModelScoreCard,
    PeerCritique,
    RevisedAnswer,
)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2}


def _jaccard_similarity(a: str, b: str) -> float:
    set_a = _tokenize(a)
    set_b = _tokenize(b)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def compute_disagreement_score(initial_answers: list[InitialAnswer]) -> float:
    if len(initial_answers) < 2:
        return 0.0
    similarities: list[float] = []
    for index, left in enumerate(initial_answers):
        for right in initial_answers[index + 1 :]:
            similarities.append(_jaccard_similarity(left.answer, right.answer))
    if not similarities:
        return 0.0
    return _clamp(1.0 - mean(similarities))


def _score_received_critiques(target_id: str, critiques: list[PeerCritique]) -> tuple[float, float, float]:
    entries = []
    for critique in critiques:
        entries.extend([item for item in critique.critiques if item.target_seat_id == target_id])
    if not entries:
        return 0.0, 0.0, 0.0

    answer_quality = mean(
        (
            _clamp(item.correctness)
            + _clamp(item.completeness)
            + _clamp(item.clarity)
            + (1.0 - _clamp(item.risk))
            + (1.0 - _clamp(item.unsupported_claims))
        )
        / 5.0
        for item in entries
    )
    unsupported = mean(_clamp(item.unsupported_claims) for item in entries)
    correctness = mean(_clamp(item.correctness) for item in entries)
    return answer_quality, unsupported, correctness


def _score_authored_critiques(critic_id: str, critiques: list[PeerCritique]) -> float:
    critique = next((item for item in critiques if item.critic_seat_id == critic_id), None)
    if critique is None or not critique.critiques:
        return 0.0

    quality_scores = []
    for item in critique.critiques:
        metrics = [
            _clamp(item.correctness),
            _clamp(item.completeness),
            _clamp(item.risk),
            _clamp(item.unsupported_claims),
            _clamp(item.clarity),
        ]
        spread = max(metrics) - min(metrics)
        comment_bonus = 0.15 if len(item.comments.strip()) >= 16 else 0.0
        all_mid = sum(abs(metric - 0.5) < 0.08 for metric in metrics) == len(metrics)
        weak_penalty = 0.2 if all_mid else 0.0
        quality_scores.append(_clamp(0.35 + spread + comment_bonus - weak_penalty))
    return mean(quality_scores)


def _consistency_score(initial_answer: InitialAnswer, revised: Optional[RevisedAnswer]) -> float:
    if revised is None:
        return 1.0
    similarity = _jaccard_similarity(initial_answer.answer, revised.revised_answer)
    return _clamp(0.4 + (0.6 * similarity))


def build_scorecards(
    *,
    seats: list[CouncilSeat],
    initial_answers: list[InitialAnswer],
    peer_critiques: list[PeerCritique],
    revised_answers: list[RevisedAnswer],
    failures: list[FailureEvent],
) -> list[ModelScoreCard]:
    answers_by_seat = {item.seat_id: item for item in initial_answers}
    revised_by_seat = {item.seat_id: item for item in revised_answers}
    failures_by_seat: dict[str, list[FailureEvent]] = {}
    for failure in failures:
        if failure.seat_id:
            failures_by_seat.setdefault(failure.seat_id, []).append(failure)

    scorecards: list[ModelScoreCard] = []
    for seat in seats:
        answer = answers_by_seat.get(seat.seat_id)
        revised = revised_by_seat.get(seat.seat_id)
        seat_failures = failures_by_seat.get(seat.seat_id, [])
        failure_flags = [failure.flag for failure in seat_failures]

        answer_quality, unsupported_level, correctness_level = _score_received_critiques(
            seat.seat_id, peer_critiques
        )
        critique_quality = _score_authored_critiques(seat.seat_id, peer_critiques)

        if unsupported_level > 0.65 and FailureFlag.UNSUPPORTED_CLAIM not in failure_flags:
            failure_flags.append(FailureFlag.UNSUPPORTED_CLAIM)
        if (
            unsupported_level > 0.65
            and correctness_level < 0.35
            and FailureFlag.CONTRADICTION not in failure_flags
        ):
            failure_flags.append(FailureFlag.CONTRADICTION)
        if critique_quality < 0.35 and FailureFlag.WEAK_CRITIQUE not in failure_flags:
            failure_flags.append(FailureFlag.WEAK_CRITIQUE)

        availability = AvailabilityStatus.READY
        if FailureFlag.UNAVAILABLE_MODEL in failure_flags:
            availability = AvailabilityStatus.UNAVAILABLE
        elif answer is None and revised is None and any(
            flag
            in failure_flags
            for flag in (
                FailureFlag.TIMEOUT,
                FailureFlag.EMPTY_RESPONSE,
                FailureFlag.MALFORMED_JSON,
            )
        ):
            availability = AvailabilityStatus.UNAVAILABLE
        elif failure_flags:
            availability = AvailabilityStatus.DEGRADED

        latency_values = [
            value
            for value in (
                answer.latency_ms if answer else None,
                next((c.latency_ms for c in peer_critiques if c.critic_seat_id == seat.seat_id), None),
                revised.latency_ms if revised else None,
            )
            if value is not None
        ]
        latency_ms = float(sum(latency_values)) if latency_values else None

        grounding = (
            revised.grounding_confidence
            if revised is not None
            else (answer.grounding_confidence if answer is not None else 0.0)
        )
        confidence = (
            revised.confidence
            if revised is not None
            else (answer.confidence if answer is not None else 0.0)
        )
        consistency = _consistency_score(answer, revised) if answer is not None else 0.0
        revision_similarity: Optional[float] = None
        if answer is not None and revised is not None:
            revision_similarity = _jaccard_similarity(answer.answer, revised.revised_answer)
            if (
                revision_similarity < 0.08
                and FailureFlag.CONTRADICTION not in failure_flags
            ):
                failure_flags.append(FailureFlag.CONTRADICTION)

        scorecards.append(
            ModelScoreCard(
                seat_id=seat.seat_id,
                role_title=seat.role_title,
                model_id=seat.model_id,
                answer_quality=_clamp(answer_quality),
                critique_quality=_clamp(critique_quality),
                grounding_confidence=_clamp(grounding),
                consistency=_clamp(consistency),
                latency_ms=latency_ms,
                confidence=_clamp(confidence),
                failure_flags=list(dict.fromkeys(failure_flags)),
                availability_status=availability,
            )
        )
    return scorecards


def compute_council_confidence(
    scorecards: list[ModelScoreCard],
    disagreement_score: float,
) -> float:
    ready_cards = [
        card for card in scorecards if card.availability_status != AvailabilityStatus.UNAVAILABLE
    ]
    if not ready_cards:
        return 0.0

    per_model = [
        _clamp(
            card.answer_quality * 0.35
            + card.critique_quality * 0.15
            + card.grounding_confidence * 0.2
            + card.consistency * 0.1
            + card.confidence * 0.2
        )
        for card in ready_cards
    ]
    base_confidence = mean(per_model)
    disagreement_penalty = _clamp(disagreement_score) * 0.5
    return _clamp(base_confidence - disagreement_penalty)


def should_escalate_fast_mode(
    *,
    config: CouncilRuntimeConfig,
    disagreement_score: float,
    council_confidence: float,
    ready_model_count: int,
) -> bool:
    if ready_model_count < config.fast_quorum:
        return True
    if disagreement_score >= config.escalation_disagreement_threshold:
        return True
    if council_confidence <= config.escalation_confidence_threshold:
        return True
    return False


def quorum_success(
    *,
    config: CouncilRuntimeConfig,
    mode: str,
    ready_model_count: int,
) -> bool:
    required = config.full_quorum if mode == "full_council" else config.fast_quorum
    return ready_model_count >= required


def rank_best_answer(scorecards: list[ModelScoreCard]) -> Optional[ModelScoreCard]:
    available = [
        card for card in scorecards if card.availability_status != AvailabilityStatus.UNAVAILABLE
    ]
    if not available:
        return None
    return sorted(
        available,
        key=lambda card: (
            card.answer_quality,
            card.grounding_confidence,
            card.confidence,
            -math.inf if card.latency_ms is None else -card.latency_ms,
        ),
        reverse=True,
    )[0]
