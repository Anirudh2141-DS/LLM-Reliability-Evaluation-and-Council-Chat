"""
Prompt templates for live multi-round council execution.
"""

from __future__ import annotations

import json
from typing import Any

from .council_runtime_schemas import (
    CouncilRequest,
    CouncilSeat,
    FinalSynthesisPayload,
    InitialAnswerPayload,
    PeerCritiquePayload,
    RevisedAnswerPayload,
)


def _schema_block(schema_model: type) -> str:
    return json.dumps(schema_model.model_json_schema(), indent=2)


def build_initial_answer_messages(
    request: CouncilRequest,
    seat: CouncilSeat,
) -> list[dict[str, str]]:
    system_prompt = (
        "You are part of an LLM Council. "
        f"Your seat is '{seat.role_title}'. "
        "Give a concise, grounded technical answer. "
        "Respond with valid JSON only and no markdown."
    )
    user_prompt = (
        f"User question:\n{request.query}\n\n"
        "Return exactly one JSON object matching this schema:\n"
        f"{_schema_block(InitialAnswerPayload)}\n\n"
        "Guidance:\n"
        "- Keep answer concise and practical.\n"
        "- confidence and grounding_confidence must be 0.0 to 1.0.\n"
        "- key_points should contain 2-5 bullets as short strings.\n"
        "- citations should be short source hints if available; otherwise empty list."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_peer_critique_messages(
    request: CouncilRequest,
    seat: CouncilSeat,
    answers: list[dict[str, Any]],
) -> list[dict[str, str]]:
    system_prompt = (
        "You are a strict peer reviewer in an LLM Council. "
        f"Your seat is '{seat.role_title}'. "
        "Critique each peer answer fairly and precisely. "
        "Respond with valid JSON only."
    )
    user_prompt = (
        f"User question:\n{request.query}\n\n"
        "Peer initial answers:\n"
        f"{json.dumps(answers, indent=2)}\n\n"
        "Return exactly one JSON object matching this schema:\n"
        f"{_schema_block(PeerCritiquePayload)}\n\n"
        "Scoring rubric (0.0 to 1.0):\n"
        "- correctness: factual and technically sound\n"
        "- completeness: covers major requirements\n"
        "- risk: deployment/policy/security risk (higher = riskier)\n"
        "- unsupported_claims: unsupported or speculative claims (higher = worse)\n"
        "- clarity: readable and actionable\n"
        "Best/weakest seat ids must refer to peer seat ids."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_revision_messages(
    request: CouncilRequest,
    seat: CouncilSeat,
    original_answer: dict[str, Any],
    received_critiques: list[dict[str, Any]],
) -> list[dict[str, str]]:
    system_prompt = (
        "You are revising your prior council answer after peer review. "
        f"Your seat is '{seat.role_title}'. "
        "Improve correctness and risk posture while staying concise. "
        "Respond with valid JSON only."
    )
    user_prompt = (
        f"User question:\n{request.query}\n\n"
        f"Your original answer:\n{json.dumps(original_answer, indent=2)}\n\n"
        f"Critiques received:\n{json.dumps(received_critiques, indent=2)}\n\n"
        "Return exactly one JSON object matching this schema:\n"
        f"{_schema_block(RevisedAnswerPayload)}\n\n"
        "If no improvement is needed, keep revised_answer close to original and explain why."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_synthesis_messages(
    request: CouncilRequest,
    chair: CouncilSeat,
    initial_answers: list[dict[str, Any]],
    critiques: list[dict[str, Any]],
    revisions: list[dict[str, Any]],
    scorecards: list[dict[str, Any]],
) -> list[dict[str, str]]:
    system_prompt = (
        "You are council chairman and final synthesizer. "
        f"Chair seat role: '{chair.role_title}'. "
        "Produce one final decision-quality answer with explicit uncertainty handling. "
        "Respond with valid JSON only."
    )
    user_prompt = (
        f"User question:\n{request.query}\n\n"
        f"Initial answers:\n{json.dumps(initial_answers, indent=2)}\n\n"
        f"Peer critiques:\n{json.dumps(critiques, indent=2)}\n\n"
        f"Revised answers:\n{json.dumps(revisions, indent=2)}\n\n"
        f"Model scorecards:\n{json.dumps(scorecards, indent=2)}\n\n"
        "Return exactly one JSON object matching this schema:\n"
        f"{_schema_block(FinalSynthesisPayload)}\n\n"
        "Chairman rules:\n"
        "- Mention uncertainty when evidence quality is low.\n"
        "- Cite risk/failure notes when relevant.\n"
        "- Keep final answer concise and actionable."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_json_repair_messages(raw_output: str, schema_model: type) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Repair malformed model output into a single valid JSON object only. "
                "Do not add markdown."
            ),
        },
        {
            "role": "user",
            "content": (
                "Malformed output:\n"
                f"{raw_output}\n\n"
                "Target schema:\n"
                f"{_schema_block(schema_model)}\n\n"
                "Return repaired JSON only."
            ),
        },
    ]
