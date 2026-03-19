from __future__ import annotations

import asyncio
from typing import Any

from rlrgf.council_runtime_inference_adapter import (
    OpenAICompatibleInferenceAdapter,
    RemoteInferenceResult,
)
from rlrgf.council_runtime_schemas import InitialAnswerPayload


def test_repair_pass_succeeds_after_malformed_initial_json(monkeypatch) -> None:
    adapter = OpenAICompatibleInferenceAdapter(
        base_url="http://unit-test.local",
        api_key="token",
    )
    responses = [
        RemoteInferenceResult(
            status="ok",
            text="{not-json",
            latency_ms=10.0,
            model_id="model-a",
        ),
        RemoteInferenceResult(
            status="ok",
            text=(
                '{"answer":"fixed","confidence":0.8,"grounding_confidence":0.8,'
                '"key_points":[],"uncertainty_notes":[],"cited_risks":[],"citations":[]}'
            ),
            latency_ms=11.0,
            model_id="model-a",
        ),
    ]

    def _fake_chat(**kwargs: Any) -> RemoteInferenceResult:
        _ = kwargs
        return responses.pop(0)

    monkeypatch.setattr(adapter, "chat", _fake_chat)

    payload, result, parse_error = adapter.call_json(
        model_id="model-a",
        messages=[{"role": "user", "content": "hello"}],
        schema_model=InitialAnswerPayload,
        timeout_s=10.0,
    )

    assert payload is not None
    assert payload.answer == "fixed"
    assert parse_error is None
    assert result.retry_count == 1
    assert result.latency_ms == 21.0


def test_repair_pass_failure_returns_merged_parse_error(monkeypatch) -> None:
    adapter = OpenAICompatibleInferenceAdapter(
        base_url="http://unit-test.local",
        api_key="token",
    )
    responses = [
        RemoteInferenceResult(
            status="ok",
            text="{broken",
            latency_ms=9.0,
            model_id="model-a",
        ),
        RemoteInferenceResult(
            status="error",
            text="",
            latency_ms=15.0,
            model_id="model-a",
            error="HTTP 500",
        ),
    ]

    def _fake_chat(**kwargs: Any) -> RemoteInferenceResult:
        _ = kwargs
        return responses.pop(0)

    monkeypatch.setattr(adapter, "chat", _fake_chat)

    payload, result, parse_error = adapter.call_json(
        model_id="model-a",
        messages=[{"role": "user", "content": "hello"}],
        schema_model=InitialAnswerPayload,
        timeout_s=10.0,
    )

    assert payload is None
    assert parse_error is not None
    assert "Malformed JSON output." in parse_error
    assert "repair:" in parse_error
    assert result.retry_count == 1


def test_async_call_json_repair_pass_succeeds_after_malformed_initial_json(
    monkeypatch,
) -> None:
    adapter = OpenAICompatibleInferenceAdapter(
        base_url="http://unit-test.local",
        api_key="token",
    )
    responses = [
        RemoteInferenceResult(
            status="ok",
            text="{not-json",
            latency_ms=10.0,
            model_id="model-a",
        ),
        RemoteInferenceResult(
            status="ok",
            text=(
                '{"answer":"fixed","confidence":0.8,"grounding_confidence":0.8,'
                '"key_points":[],"uncertainty_notes":[],"cited_risks":[],"citations":[]}'
            ),
            latency_ms=11.0,
            model_id="model-a",
        ),
    ]

    async def _fake_chat_async(**kwargs: Any) -> RemoteInferenceResult:
        _ = kwargs
        return responses.pop(0)

    monkeypatch.setattr(adapter, "chat_async", _fake_chat_async)

    payload, result, parse_error = asyncio.run(
        adapter.call_json_async(
            model_id="model-a",
            messages=[{"role": "user", "content": "hello"}],
            schema_model=InitialAnswerPayload,
            timeout_s=10.0,
        )
    )

    assert payload is not None
    assert payload.answer == "fixed"
    assert parse_error is None
    assert result.retry_count == 1
    assert result.latency_ms == 21.0
