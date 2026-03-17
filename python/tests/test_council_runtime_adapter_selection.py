from __future__ import annotations

from pathlib import Path

import pytest

from rlrgf.council_runtime import CouncilRuntime
from rlrgf.council_runtime_config import CouncilRuntimeConfig
from rlrgf.council_runtime_inference_adapter import (
    HuggingFaceRouterInferenceAdapter,
    MockCouncilInferenceAdapter,
)


def test_runtime_defaults_to_mock_adapter_when_real_models_disabled(tmp_path: Path) -> None:
    config = CouncilRuntimeConfig(
        use_real_models=False,
        cache_path=str(tmp_path / "cache.json"),
    )

    runtime = CouncilRuntime(config=config)

    assert isinstance(runtime.adapter, MockCouncilInferenceAdapter)


def test_runtime_uses_hf_adapter_when_real_models_enabled(tmp_path: Path) -> None:
    config = CouncilRuntimeConfig(
        use_real_models=True,
        api_key="hf_test_token",
        hf_token_found=True,
        hf_token_source="environment",
        base_url="https://router.huggingface.co/v1",
        cache_path=str(tmp_path / "cache.json"),
    )

    runtime = CouncilRuntime(config=config)

    assert isinstance(runtime.adapter, HuggingFaceRouterInferenceAdapter)


def test_runtime_errors_when_real_models_enabled_without_hf_token(
    tmp_path: Path,
) -> None:
    expected_path = (
        r"E:\MLOps\LLM Failure Evaluation Engine\python\rlrgf\hf_token.txt"
    )
    config = CouncilRuntimeConfig(
        use_real_models=True,
        api_key="",
        hf_token_found=False,
        hf_token_source="none",
        hf_token_path=expected_path,
        base_url="https://router.huggingface.co/v1",
        cache_path=str(tmp_path / "cache.json"),
    )

    with pytest.raises(ValueError) as error:
        CouncilRuntime(config=config)

    message = str(error.value)
    assert "Token file missing or empty" in message
    assert expected_path in message
