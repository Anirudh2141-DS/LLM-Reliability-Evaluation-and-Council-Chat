"""
Remote inference adapter for OpenAI-compatible chat endpoints.
"""

from __future__ import annotations

import asyncio
import ast
from dataclasses import dataclass
import json
import logging
import re
import time
from typing import Any, Awaitable, Callable, Optional, Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from .council_runtime_prompts import build_json_repair_messages


logger = logging.getLogger(__name__)

SchemaT = TypeVar("SchemaT", bound=BaseModel)
HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"


@dataclass
class RemoteInferenceResult:
    status: str
    text: str
    latency_ms: float
    model_id: str
    error: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    retry_count: int = 0
    http_status: Optional[int] = None
    parse_mode: str = "clean_json"
    raw_output_present: bool = False
    recovered_output_used: bool = False
    parse_error_type: Optional[str] = None
    failure_is_hard: bool = False
    usable_contribution: bool = False
    usable_for_quorum: bool = False


class CouncilInferenceAdapter(Protocol):
    def call_json(
        self,
        *,
        model_id: str,
        messages: list[dict[str, str]],
        schema_model: type[SchemaT],
        timeout_s: float,
        temperature: float = 0.2,
        max_tokens: int = 900,
    ) -> tuple[Optional[SchemaT], RemoteInferenceResult, Optional[str]]:
        ...

    async def call_json_async(
        self,
        *,
        model_id: str,
        messages: list[dict[str, str]],
        schema_model: type[SchemaT],
        timeout_s: float,
        temperature: float = 0.2,
        max_tokens: int = 900,
    ) -> tuple[Optional[SchemaT], RemoteInferenceResult, Optional[str]]:
        ...


class MockCouncilInferenceAdapter:
    """
    Deterministic offline adapter for tests and local demo mode.
    """

    def __init__(self, *, latency_ms: float = 5.0) -> None:
        self.latency_ms = latency_ms

    def call_json(
        self,
        *,
        model_id: str,
        messages: list[dict[str, str]],
        schema_model: type[SchemaT],
        timeout_s: float,
        temperature: float = 0.2,
        max_tokens: int = 900,
    ) -> tuple[Optional[SchemaT], RemoteInferenceResult, Optional[str]]:
        _ = (messages, timeout_s, temperature, max_tokens)
        payload_dict = _mock_payload_for_schema(schema_model=schema_model, model_id=model_id)
        result = RemoteInferenceResult(
            status="ok",
            text=json.dumps(payload_dict),
            latency_ms=self.latency_ms,
            model_id=model_id,
            parse_mode="clean_json",
            raw_output_present=True,
            recovered_output_used=False,
            parse_error_type=None,
            failure_is_hard=False,
            usable_contribution=True,
            usable_for_quorum=True,
        )
        try:
            model = schema_model.model_validate(payload_dict)
            return model, result, None
        except ValidationError as error:
            result.parse_mode = "hard_failure"
            result.parse_error_type = "schema_validation_error"
            result.failure_is_hard = True
            result.usable_contribution = False
            result.usable_for_quorum = False
            return None, result, str(error)

    async def call_json_async(
        self,
        *,
        model_id: str,
        messages: list[dict[str, str]],
        schema_model: type[SchemaT],
        timeout_s: float,
        temperature: float = 0.2,
        max_tokens: int = 900,
    ) -> tuple[Optional[SchemaT], RemoteInferenceResult, Optional[str]]:
        return self.call_json(
            model_id=model_id,
            messages=messages,
            schema_model=schema_model,
            timeout_s=timeout_s,
            temperature=temperature,
            max_tokens=max_tokens,
        )


class OpenAICompatibleInferenceAdapter:
    """
    Adapter for OpenAI-compatible JSON chat completion APIs.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        request_timeout_s: float = 30.0,
        max_retries: int = 2,
        retry_backoff_s: float = 0.8,
        retry_backoff_cap_s: float = 4.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.request_timeout_s = request_timeout_s
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        self.retry_backoff_cap_s = retry_backoff_cap_s
        self._client = httpx.Client()

    def _chat_url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _sleep_backoff(self, attempt: int) -> None:
        delay = min(self.retry_backoff_cap_s, self.retry_backoff_s * (2**attempt))
        time.sleep(delay)

    async def _sleep_backoff_async(self, attempt: int) -> None:
        delay = min(self.retry_backoff_cap_s, self.retry_backoff_s * (2**attempt))
        await asyncio.sleep(delay)

    def chat(
        self,
        *,
        model_id: str,
        messages: list[dict[str, str]],
        timeout_s: float,
        temperature: float = 0.2,
        max_tokens: int = 900,
        response_format_json: bool = True,
    ) -> RemoteInferenceResult:
        if not self.base_url:
            return RemoteInferenceResult(
                status="unavailable",
                text="",
                latency_ms=0.0,
                model_id=model_id,
                error="COUNCIL_API_BASE_URL is not configured.",
            )

        url = self._chat_url()
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format_json:
            payload["response_format"] = {"type": "json_object"}

        retries = self.max_retries
        for attempt in range(retries + 1):
            start = time.perf_counter()
            try:
                response = self._client.post(
                    url,
                    headers=self._headers(),
                    json=payload,
                    timeout=min(timeout_s, self.request_timeout_s),
                )
                latency_ms = (time.perf_counter() - start) * 1000.0

                if response.status_code in (408, 429) or response.status_code >= 500:
                    if attempt < retries:
                        self._sleep_backoff(attempt)
                        continue
                    return RemoteInferenceResult(
                        status="error",
                        text="",
                        latency_ms=latency_ms,
                        model_id=model_id,
                        error=f"HTTP {response.status_code}: retry budget exhausted.",
                        retry_count=attempt,
                        http_status=response.status_code,
                    )

                if response.status_code >= 400:
                    status = (
                        "unavailable"
                        if response.status_code in (401, 403, 404)
                        else "error"
                    )
                    return RemoteInferenceResult(
                        status=status,
                        text="",
                        latency_ms=latency_ms,
                        model_id=model_id,
                        error=f"HTTP {response.status_code}: {response.text[:300]}",
                        retry_count=attempt,
                        http_status=response.status_code,
                    )

                body = response.json()
                content = _extract_content(body)
                usage = body.get("usage", {}) if isinstance(body, dict) else {}
                return RemoteInferenceResult(
                    status="ok",
                    text=content.strip(),
                    latency_ms=latency_ms,
                    model_id=model_id,
                    prompt_tokens=_coerce_int(usage.get("prompt_tokens")),
                    completion_tokens=_coerce_int(usage.get("completion_tokens")),
                    total_tokens=_coerce_int(usage.get("total_tokens")),
                    retry_count=attempt,
                    http_status=response.status_code,
                )
            except httpx.TimeoutException:
                latency_ms = (time.perf_counter() - start) * 1000.0
                if attempt < retries:
                    self._sleep_backoff(attempt)
                    continue
                return RemoteInferenceResult(
                    status="timeout",
                    text="",
                    latency_ms=latency_ms,
                    model_id=model_id,
                    error="Request timed out.",
                    retry_count=attempt,
                )
            except Exception as error:  # pragma: no cover - defensive branch
                latency_ms = (time.perf_counter() - start) * 1000.0
                logger.warning("Remote call failed for %s: %s", model_id, error)
                if attempt < retries:
                    self._sleep_backoff(attempt)
                    continue
                return RemoteInferenceResult(
                    status="error",
                    text="",
                    latency_ms=latency_ms,
                    model_id=model_id,
                    error=str(error),
                    retry_count=attempt,
                )

        # Not expected to execute.
        return RemoteInferenceResult(
            status="error",
            text="",
            latency_ms=0.0,
            model_id=model_id,
            error="Unexpected retry loop exit.",
            retry_count=retries,
        )

    async def chat_async(
        self,
        *,
        model_id: str,
        messages: list[dict[str, str]],
        timeout_s: float,
        temperature: float = 0.2,
        max_tokens: int = 900,
        response_format_json: bool = True,
    ) -> RemoteInferenceResult:
        if not self.base_url:
            return RemoteInferenceResult(
                status="unavailable",
                text="",
                latency_ms=0.0,
                model_id=model_id,
                error="COUNCIL_API_BASE_URL is not configured.",
            )

        url = self._chat_url()
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format_json:
            payload["response_format"] = {"type": "json_object"}

        retries = self.max_retries
        async with httpx.AsyncClient() as client:
            for attempt in range(retries + 1):
                start = time.perf_counter()
                try:
                    response = await client.post(
                        url,
                        headers=self._headers(),
                        json=payload,
                        timeout=min(timeout_s, self.request_timeout_s),
                    )
                    latency_ms = (time.perf_counter() - start) * 1000.0

                    if response.status_code in (408, 429) or response.status_code >= 500:
                        if attempt < retries:
                            await self._sleep_backoff_async(attempt)
                            continue
                        return RemoteInferenceResult(
                            status="error",
                            text="",
                            latency_ms=latency_ms,
                            model_id=model_id,
                            error=f"HTTP {response.status_code}: retry budget exhausted.",
                            retry_count=attempt,
                            http_status=response.status_code,
                        )

                    if response.status_code >= 400:
                        status = (
                            "unavailable"
                            if response.status_code in (401, 403, 404)
                            else "error"
                        )
                        return RemoteInferenceResult(
                            status=status,
                            text="",
                            latency_ms=latency_ms,
                            model_id=model_id,
                            error=f"HTTP {response.status_code}: {response.text[:300]}",
                            retry_count=attempt,
                            http_status=response.status_code,
                        )

                    body = response.json()
                    content = _extract_content(body)
                    usage = body.get("usage", {}) if isinstance(body, dict) else {}
                    return RemoteInferenceResult(
                        status="ok",
                        text=content.strip(),
                        latency_ms=latency_ms,
                        model_id=model_id,
                        prompt_tokens=_coerce_int(usage.get("prompt_tokens")),
                        completion_tokens=_coerce_int(usage.get("completion_tokens")),
                        total_tokens=_coerce_int(usage.get("total_tokens")),
                        retry_count=attempt,
                        http_status=response.status_code,
                    )
                except httpx.TimeoutException:
                    latency_ms = (time.perf_counter() - start) * 1000.0
                    if attempt < retries:
                        await self._sleep_backoff_async(attempt)
                        continue
                    return RemoteInferenceResult(
                        status="timeout",
                        text="",
                        latency_ms=latency_ms,
                        model_id=model_id,
                        error="Request timed out.",
                        retry_count=attempt,
                    )
                except Exception as error:  # pragma: no cover - defensive branch
                    latency_ms = (time.perf_counter() - start) * 1000.0
                    logger.warning("Remote call failed for %s: %s", model_id, error)
                    if attempt < retries:
                        await self._sleep_backoff_async(attempt)
                        continue
                    return RemoteInferenceResult(
                        status="error",
                        text="",
                        latency_ms=latency_ms,
                        model_id=model_id,
                        error=str(error),
                        retry_count=attempt,
                    )

        return RemoteInferenceResult(
            status="error",
            text="",
            latency_ms=0.0,
            model_id=model_id,
            error="Unexpected retry loop exit.",
            retry_count=retries,
        )

    def call_json(
        self,
        *,
        model_id: str,
        messages: list[dict[str, str]],
        schema_model: type[SchemaT],
        timeout_s: float,
        temperature: float = 0.2,
        max_tokens: int = 900,
    ) -> tuple[Optional[SchemaT], RemoteInferenceResult, Optional[str]]:
        async def _chat_via_sync(**chat_kwargs: Any) -> RemoteInferenceResult:
            return self.chat(**chat_kwargs)

        return _run_coroutine(
            self._call_json_via_chat(
                chat_fn=_chat_via_sync,
                model_id=model_id,
                messages=messages,
                schema_model=schema_model,
                timeout_s=timeout_s,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )

    async def call_json_async(
        self,
        *,
        model_id: str,
        messages: list[dict[str, str]],
        schema_model: type[SchemaT],
        timeout_s: float,
        temperature: float = 0.2,
        max_tokens: int = 900,
    ) -> tuple[Optional[SchemaT], RemoteInferenceResult, Optional[str]]:
        return await self._call_json_via_chat(
            chat_fn=self.chat_async,
            model_id=model_id,
            messages=messages,
            schema_model=schema_model,
            timeout_s=timeout_s,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def _call_json_via_chat(
        self,
        *,
        chat_fn: Callable[..., Awaitable[RemoteInferenceResult]],
        model_id: str,
        messages: list[dict[str, str]],
        schema_model: type[SchemaT],
        timeout_s: float,
        temperature: float = 0.2,
        max_tokens: int = 900,
    ) -> tuple[Optional[SchemaT], RemoteInferenceResult, Optional[str]]:
        result = await chat_fn(
            model_id=model_id,
            messages=messages,
            timeout_s=timeout_s,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format_json=True,
        )
        result.raw_output_present = bool((result.text or "").strip())
        if result.status != "ok":
            result.parse_mode = "hard_failure"
            result.parse_error_type = _infer_parse_error_type(result.error or result.status)
            result.failure_is_hard = True
            result.usable_contribution = False
            result.usable_for_quorum = False
            return None, result, result.error

        parsed, parse_mode, parse_error_type, parse_error = _parse_json_text(result.text)
        initial_error = parse_error or parse_error_type
        if parsed is not None:
            try:
                model = schema_model.model_validate(parsed)
                _mark_success(
                    result,
                    parse_mode=parse_mode,
                    parse_error_type=parse_error_type,
                    recovered=parse_mode != "clean_json",
                )
                parse_note = None
                if parse_mode != "clean_json":
                    parse_note = f"{parse_mode}: {parse_error_type or 'recovered malformed json'}"
                return model, result, parse_note
            except ValidationError as error:
                parse_error = str(error)
                parse_error_type = "schema_validation_error"
                initial_error = parse_error

        # One repair pass on malformed JSON/validation failure.
        repair_messages = build_json_repair_messages(result.text, schema_model)
        repair_result = await chat_fn(
            model_id=model_id,
            messages=repair_messages,
            timeout_s=timeout_s,
            temperature=0.0,
            max_tokens=max_tokens,
            response_format_json=True,
        )
        result.latency_ms += repair_result.latency_ms
        result.retry_count += repair_result.retry_count + 1
        result.http_status = repair_result.http_status or result.http_status
        result.prompt_tokens = _sum_optional_int(result.prompt_tokens, repair_result.prompt_tokens)
        result.completion_tokens = _sum_optional_int(
            result.completion_tokens, repair_result.completion_tokens
        )
        result.total_tokens = _sum_optional_int(result.total_tokens, repair_result.total_tokens)
        result.raw_output_present = result.raw_output_present or bool(
            (repair_result.text or "").strip()
        )

        if repair_result.status != "ok":
            initial_error = _merge_errors(initial_error, repair_result.error)
        else:
            repaired, _, repair_error_type, repair_error = _parse_json_text(repair_result.text)
            if repaired is not None:
                try:
                    model = schema_model.model_validate(repaired)
                    result.text = repair_result.text
                    _mark_success(
                        result,
                        parse_mode="repaired_json",
                        parse_error_type=parse_error_type or repair_error_type,
                        recovered=True,
                    )
                    return (
                        model,
                        result,
                        None,
                    )
                except ValidationError as error:
                    repair_error = str(error)
                    repair_error_type = "schema_validation_error"
            initial_error = _merge_errors(initial_error, repair_error or repair_error_type)

        fallback_source = result.text if (result.text or "").strip() else repair_result.text
        fallback_payload, fallback_text = _plain_text_fallback_payload(
            schema_model=schema_model,
            raw_text=fallback_source,
        )
        if fallback_payload is not None:
            try:
                model = schema_model.model_validate(fallback_payload)
                if fallback_text:
                    result.text = fallback_text
                _mark_success(
                    result,
                    parse_mode="plain_text_fallback",
                    parse_error_type=parse_error_type or "malformed_json",
                    recovered=True,
                )
                return (
                    model,
                    result,
                    _merge_errors(initial_error, "plain_text_fallback"),
                )
            except ValidationError as error:
                initial_error = _merge_errors(initial_error, str(error))

        result.parse_mode = "hard_failure"
        result.parse_error_type = _infer_parse_error_type(initial_error)
        result.recovered_output_used = False
        result.failure_is_hard = True
        result.usable_contribution = False
        result.usable_for_quorum = False
        return None, result, _merge_errors(initial_error, "hard_failure")


def _mark_success(
    result: RemoteInferenceResult,
    *,
    parse_mode: str,
    parse_error_type: Optional[str],
    recovered: bool,
) -> None:
    result.parse_mode = parse_mode
    result.parse_error_type = parse_error_type
    result.recovered_output_used = recovered
    result.failure_is_hard = False
    result.usable_contribution = True
    result.usable_for_quorum = True


def _plain_text_fallback_payload(
    *,
    schema_model: type[BaseModel],
    raw_text: str,
) -> tuple[Optional[dict[str, Any]], str]:
    fallback_text = _normalize_plain_text_fallback(raw_text)
    if _looks_like_broken_json_fragment(fallback_text):
        return None, ""
    if not fallback_text:
        return None, ""
    name = schema_model.__name__
    if name == "InitialAnswerPayload":
        return (
            {
                "answer": fallback_text,
                "confidence": 0.2,
                "grounding_confidence": 0.2,
                "key_points": [],
                "uncertainty_notes": ["Unstructured output fallback used."],
                "cited_risks": [],
                "citations": [],
            },
            fallback_text,
        )
    if name == "PeerCritiquePayload":
        return (
            {
                "critiques": [],
                "best_answer_seat_id": None,
                "weakest_answer_seat_id": None,
                "confidence": 0.2,
            },
            fallback_text,
        )
    if name == "RevisedAnswerPayload":
        return (
            {
                "revised_answer": fallback_text,
                "confidence": 0.2,
                "grounding_confidence": 0.2,
                "change_summary": "Unstructured output fallback used.",
            },
            fallback_text,
        )
    if name == "FinalSynthesisPayload":
        return (
            {
                "final_answer": fallback_text,
                "reasoning_summary": "Plain-text fallback synthesis was used due malformed structured output.",
                "winner_seat_ids": [],
                "strongest_contributors": [],
                "uncertainty_notes": ["Unstructured output fallback used."],
                "cited_risk_notes": [],
                "confidence": 0.2,
            },
            fallback_text,
        )
    return None, fallback_text


def _normalize_plain_text_fallback(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    cleaned = _strip_code_fences(cleaned)
    cleaned = re.sub(
        r"^\s*(?:here(?:'s| is)?\s+)?(?:the\s+)?json(?:\s+response)?\s*:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:2000]


def _looks_like_broken_json_fragment(text: str) -> bool:
    cleaned = (text or "").strip()
    if not cleaned:
        return False
    if cleaned[0] not in "{[":
        return False
    return _load_json_candidate(cleaned) is None


class HuggingFaceRouterInferenceAdapter(OpenAICompatibleInferenceAdapter):
    """
    Adapter for Hugging Face router OpenAI-compatible chat endpoint.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = HF_ROUTER_BASE_URL,
        request_timeout_s: float = 30.0,
        max_retries: int = 2,
        retry_backoff_s: float = 0.8,
        retry_backoff_cap_s: float = 4.0,
    ) -> None:
        super().__init__(
            base_url=base_url or HF_ROUTER_BASE_URL,
            api_key=api_key,
            request_timeout_s=request_timeout_s,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
            retry_backoff_cap_s=retry_backoff_cap_s,
        )


def _mock_payload_for_schema(*, schema_model: type[BaseModel], model_id: str) -> dict[str, Any]:
    name = schema_model.__name__
    if name == "InitialAnswerPayload":
        return {
            "answer": f"[MOCK] Initial answer from {model_id}.",
            "confidence": 0.68,
            "grounding_confidence": 0.62,
            "key_points": ["mock baseline", "offline adapter"],
            "uncertainty_notes": ["offline mode: no remote evidence"],
            "cited_risks": [],
            "citations": [],
        }
    if name == "PeerCritiquePayload":
        return {
            "critiques": [],
            "best_answer_seat_id": None,
            "weakest_answer_seat_id": None,
            "confidence": 0.55,
        }
    if name == "RevisedAnswerPayload":
        return {
            "revised_answer": f"[MOCK] Revised answer from {model_id}.",
            "confidence": 0.64,
            "grounding_confidence": 0.58,
            "change_summary": "Offline mock revision pass.",
        }
    if name == "FinalSynthesisPayload":
        return {
            "final_answer": f"[MOCK] Final synthesis from {model_id}.",
            "reasoning_summary": "Offline mock adapter synthesized deterministic output.",
            "winner_seat_ids": [],
            "strongest_contributors": [],
            "uncertainty_notes": ["offline mode enabled"],
            "cited_risk_notes": [],
            "confidence": 0.6,
        }
    return {}


def _coerce_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_content(body: Any) -> str:
    if not isinstance(body, dict):
        return ""
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return ""
    message = first_choice.get("message")
    if isinstance(message, dict):
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(parts)
    return str(first_choice.get("text", ""))


def _parse_json_text(
    text: str,
) -> tuple[Optional[dict[str, Any]], str, Optional[str], Optional[str]]:
    cleaned = (text or "").strip()
    if not cleaned:
        return None, "hard_failure", "empty_response", "Empty response."

    strict = _load_json_candidate(cleaned)
    if strict is not None:
        return strict, "clean_json", None, None

    parse_error = "Malformed JSON output."
    normalized = _strip_leading_json_prefix(_strip_code_fences(cleaned))
    candidates = [normalized]
    preamble_stripped = _strip_leading_preamble(normalized)
    if preamble_stripped and preamble_stripped != normalized:
        candidates.append(preamble_stripped)
    unwrapped = _unwrap_json_string_literal(normalized)
    if unwrapped:
        candidates.append(unwrapped)
    if preamble_stripped:
        unwrapped_preamble = _unwrap_json_string_literal(preamble_stripped)
        if unwrapped_preamble:
            candidates.append(unwrapped_preamble)

    extracted = _extract_json_block(normalized)
    if extracted:
        candidates.append(extracted)
    extracted_original = _extract_json_block(cleaned)
    if extracted_original:
        candidates.append(extracted_original)

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate:
            continue
        for variant in _repair_json_variants(candidate):
            if variant in seen:
                continue
            seen.add(variant)
            parsed = _load_json_candidate(variant)
            if parsed is not None:
                return parsed, "repaired_json", "malformed_json", parse_error
    return None, "hard_failure", "malformed_json", parse_error


def _sum_optional_int(left: Optional[int], right: Optional[int]) -> Optional[int]:
    if left is None and right is None:
        return None
    return int(left or 0) + int(right or 0)


def _run_coroutine(coroutine: Any) -> Any:
    try:
        return asyncio.run(coroutine)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coroutine)
        finally:
            loop.close()


def _merge_errors(primary: Optional[str], secondary: Optional[str]) -> str:
    primary_clean = (primary or "").strip()
    secondary_clean = (secondary or "").strip()
    if primary_clean and secondary_clean:
        return f"{primary_clean} | repair: {secondary_clean}"
    return primary_clean or secondary_clean or "Unknown parse failure."


def _infer_parse_error_type(error: Optional[str]) -> str:
    cleaned = (error or "").strip().lower()
    if not cleaned:
        return "unknown"
    if "empty response" in cleaned:
        return "empty_response"
    if "timeout" in cleaned:
        return "timeout"
    if "validation" in cleaned or "field required" in cleaned:
        return "schema_validation_error"
    if "malformed" in cleaned or "json" in cleaned:
        return "malformed_json"
    return "parse_error"


def _load_json_candidate(candidate: str) -> Optional[dict[str, Any]]:
    current = (candidate or "").strip()
    for _ in range(3):
        try:
            parsed = json.loads(current)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    return item
            return None
        if isinstance(parsed, str):
            nested = parsed.strip()
            if not nested or nested == current:
                return None
            current = nested
            continue
        return None
    return None


def _strip_code_fences(text: str) -> str:
    fenced = re.match(r"^\s*```(?:json)?\s*([\s\S]*?)\s*```\s*$", text, flags=re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return re.sub(r"```(?:json)?|```", "", text, flags=re.IGNORECASE).strip()


def _strip_leading_json_prefix(text: str) -> str:
    return re.sub(
        r"^\s*(?:here(?:'s| is)?\s+)?(?:the\s+)?(?:json|response|output|payload)(?:\s+(?:response|object))?\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()


def _strip_leading_preamble(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    match = re.search(r"[{\[]|['\"]\s*[{\[]", cleaned)
    if match is None:
        return cleaned
    return cleaned[match.start() :].strip()


def _repair_json_variants(text: str) -> list[str]:
    variants: list[str] = [text.strip()]
    smart_quotes = _normalize_quotes(variants[0])
    variants.append(smart_quotes)
    variants.append(_coerce_single_quoted_json(variants[0]))

    for base in list(variants):
        no_trailing_commas = _remove_trailing_commas(base)
        escaped_newlines = _escape_newlines_in_json_strings(base)
        single_quote_fixed = _coerce_single_quoted_json(base)
        variants.append(no_trailing_commas)
        variants.append(escaped_newlines)
        variants.append(_escape_newlines_in_json_strings(no_trailing_commas))
        variants.append(single_quote_fixed)
        variants.append(_remove_trailing_commas(single_quote_fixed))

    compacted: list[str] = []
    seen: set[str] = set()
    for candidate in variants:
        cleaned = candidate.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        compacted.append(cleaned)
    return compacted


def _normalize_quotes(text: str) -> str:
    return (
        text.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )


def _coerce_single_quoted_json(text: str) -> str:
    candidate = text.strip()
    if not candidate:
        return ""
    candidate = re.sub(
        r"(^|[{,]\s*)'([^'\\]+)'\s*:",
        lambda match: f'{match.group(1)}"{match.group(2)}":',
        candidate,
        flags=re.MULTILINE,
    )
    candidate = re.sub(
        r":\s*'([^'\\]*(?:\\.[^'\\]*)*)'(\s*[,}\]])",
        _replace_single_quoted_value,
        candidate,
    )
    return candidate


def _replace_single_quoted_value(match: re.Match[str]) -> str:
    value = match.group(1).replace("\\'", "'").replace('"', '\\"')
    return f': "{value}"{match.group(2)}'


def _remove_trailing_commas(text: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", text)


def _escape_newlines_in_json_strings(text: str) -> str:
    pieces: list[str] = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                pieces.append(char)
                escaped = False
                continue
            if char == "\\":
                pieces.append(char)
                escaped = True
                continue
            if char == '"':
                pieces.append(char)
                in_string = False
                continue
            if char == "\n":
                pieces.append("\\n")
                continue
            if char == "\r":
                pieces.append("\\r")
                continue
            pieces.append(char)
            continue
        if char == '"':
            in_string = True
        pieces.append(char)
    return "".join(pieces)


def _extract_json_block(text: str) -> str:
    start = -1
    for idx, char in enumerate(text):
        if char in "{[":
            start = idx
            break
    if start < 0:
        return ""

    stack: list[str] = []
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char in "{[":
            stack.append(char)
            continue
        if char in "}]":
            if not stack:
                continue
            opening = stack.pop()
            if (opening == "{" and char != "}") or (opening == "[" and char != "]"):
                return ""
            if not stack:
                return text[start : index + 1]
    return ""


def _unwrap_json_string_literal(text: str) -> str:
    candidate = (text or "").strip()
    if len(candidate) < 2:
        return ""
    if candidate[0] not in {'"', "'"} or candidate[-1] != candidate[0]:
        return ""
    try:
        value = ast.literal_eval(candidate)
    except (SyntaxError, ValueError):
        return ""
    if isinstance(value, str):
        return value.strip()
    return ""
