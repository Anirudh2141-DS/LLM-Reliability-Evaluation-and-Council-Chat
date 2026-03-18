from __future__ import annotations

from pathlib import Path

from rlrgf.council_runtime import CouncilRuntime
from rlrgf.council_runtime_config import CouncilRuntimeConfig
from rlrgf.council_runtime_inference_adapter import (
    HuggingFaceRouterInferenceAdapter,
    MockCouncilInferenceAdapter,
)
from rlrgf.council_runtime_schemas import ExecutionMode


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


def test_runtime_falls_back_to_mock_when_real_models_enabled_without_hf_token(
    tmp_path: Path,
) -> None:
    config = CouncilRuntimeConfig(
        use_real_models=True,
        api_key="",
        hf_token_found=False,
        hf_token_source="none",
        hf_token_path=r"E:\MLOps\LLM Failure Evaluation Engine\python\rlrgf\hf_token.txt",
        base_url="https://router.huggingface.co/v1",
        cache_path=str(tmp_path / "cache.json"),
    )

    runtime = CouncilRuntime(config=config)

    assert isinstance(runtime.adapter, MockCouncilInferenceAdapter)


def test_interactive_mode_prefers_remote_adapter_when_token_exists(tmp_path: Path) -> None:
    config = CouncilRuntimeConfig(
        default_execution_mode=ExecutionMode.INTERACTIVE,
        interactive_prefer_remote=True,
        use_real_models=False,
        api_key="hf_test_token",
        hf_token_found=True,
        hf_token_source="environment",
        base_url="https://router.huggingface.co/v1",
        cache_path=str(tmp_path / "cache.json"),
    )

    runtime = CouncilRuntime(config=config)

    assert isinstance(runtime.adapter, HuggingFaceRouterInferenceAdapter)
