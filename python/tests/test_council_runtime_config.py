from __future__ import annotations

import json
import os
from pathlib import Path

import rlrgf.council_runtime_config as runtime_config_module
from rlrgf.council_runtime_config import CouncilRuntimeConfig, load_runtime_config
from rlrgf.council_runtime_schemas import ExecutionMode


def test_with_seat_model_overrides_updates_only_targeted_seat() -> None:
    config = CouncilRuntimeConfig()

    updated = config.with_seat_model_overrides({"phi-3-mini": "custom/phi-3-mini"})

    phi = next(seat for seat in updated.seats if seat.seat_id == "phi-3-mini")
    llama = next(seat for seat in updated.seats if seat.seat_id == "llama-3-8b")
    assert phi.model_id == "custom/phi-3-mini"
    assert llama.model_id == "meta-llama/Meta-Llama-3-8B-Instruct"


def test_load_runtime_config_respects_file_and_env_overrides(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        runtime_config_module, "HF_TOKEN_FILE_PATH", tmp_path / "missing_hf_token.txt"
    )
    monkeypatch.delenv("HF_TOKEN", raising=False)
    config_path = tmp_path / "council_models.json"
    config_path.write_text(
        json.dumps(
            {
                "chair_seat_id": "phi-3-mini",
                "backup_chair_seat_id": "qwen-2-7b",
                "fast_seat_ids": ["llama-3-8b", "phi-3-mini"],
                "model_ids": {"llama-3-8b": "from/file/llama"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("COUNCIL_MODEL_LLAMA_3_8B", "from/env/llama")

    loaded = load_runtime_config(str(config_path))

    llama = next(seat for seat in loaded.seats if seat.seat_id == "llama-3-8b")
    qwen = next(seat for seat in loaded.seats if seat.seat_id == "qwen-2-7b")
    assert loaded.chair_seat_id == "phi-3-mini"
    assert loaded.backup_chair_seat_id == "qwen-2-7b"
    assert llama.model_id == "from/env/llama"
    assert llama.enabled_in_fast_mode is True
    assert qwen.enabled_in_fast_mode is False


def test_load_runtime_config_coerces_numeric_and_boolean_fields(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        runtime_config_module, "HF_TOKEN_FILE_PATH", tmp_path / "missing_hf_token.txt"
    )
    monkeypatch.delenv("HF_TOKEN", raising=False)
    config_path = tmp_path / "council_models.json"
    config_path.write_text(
        json.dumps(
            {
                "enable_revision_round": "false",
                "request_timeout_s": "12.5",
                "model_timeout_s": "9",
                "max_retries": "3",
                "retry_backoff_s": "1.2",
                "retry_backoff_cap_s": "2.4",
                "escalation_disagreement_threshold": "0.61",
                "escalation_confidence_threshold": "0.44",
                "fast_quorum": "2",
                "full_quorum": "3",
            }
        ),
        encoding="utf-8",
    )

    loaded = load_runtime_config(str(config_path))

    assert loaded.enable_revision_round is False
    assert loaded.request_timeout_s == 12.5
    assert loaded.model_timeout_s == 9.0
    assert loaded.max_retries == 3
    assert loaded.retry_backoff_s == 1.2
    assert loaded.retry_backoff_cap_s == 2.4
    assert loaded.escalation_disagreement_threshold == 0.61
    assert loaded.escalation_confidence_threshold == 0.44
    assert loaded.fast_quorum == 2
    assert loaded.full_quorum == 3


def test_hf_token_loaded_from_file_and_sets_runtime_env(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "council_models.json"
    config_path.write_text("{}", encoding="utf-8")
    token_path = tmp_path / "hf_token.txt"
    token_path.write_text("  hf_file_token\n", encoding="utf-8")
    monkeypatch.setattr(runtime_config_module, "HF_TOKEN_FILE_PATH", token_path)
    monkeypatch.setenv("HF_TOKEN", "hf_env_token")

    loaded = load_runtime_config(str(config_path))

    assert loaded.use_real_models is True
    assert loaded.api_key == "hf_file_token"
    assert loaded.hf_token_found is True
    assert loaded.hf_token_source == "file"
    assert loaded.hf_token_path == str(token_path)
    assert os.getenv("HF_TOKEN") == "hf_file_token"


def test_hf_token_falls_back_to_environment_when_file_missing(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "council_models.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        runtime_config_module, "HF_TOKEN_FILE_PATH", tmp_path / "missing_hf_token.txt"
    )
    monkeypatch.setenv("HF_TOKEN", "  hf_test_token  ")

    loaded = load_runtime_config(str(config_path))

    assert loaded.use_real_models is True
    assert loaded.api_key == "hf_test_token"
    assert loaded.hf_token_found is True
    assert loaded.hf_token_source == "environment"
    assert loaded.base_url == "https://router.huggingface.co/v1"


def test_explicit_env_switch_can_disable_real_models(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "council_models.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        runtime_config_module, "HF_TOKEN_FILE_PATH", tmp_path / "missing_hf_token.txt"
    )
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    monkeypatch.setenv("COUNCIL_USE_REAL_MODELS", "false")

    loaded = load_runtime_config(str(config_path))

    assert loaded.use_real_models is False


def test_load_runtime_config_parses_execution_mode_and_critique_flags(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        runtime_config_module, "HF_TOKEN_FILE_PATH", tmp_path / "missing_hf_token.txt"
    )
    monkeypatch.delenv("HF_TOKEN", raising=False)
    config_path = tmp_path / "council_models.json"
    config_path.write_text(
        json.dumps(
            {
                "default_execution_mode": "interactive",
                "benchmark_enable_pairwise_critique": True,
                "benchmark_enable_summary_review": False,
                "interactive_max_models": 1,
                "interactive_enable_secondary_review": True,
                "interactive_prefer_remote": True,
                "interactive_disable_cpu_fallback": True,
                "interactive_skip_heavy_probe": True,
            }
        ),
        encoding="utf-8",
    )

    loaded = load_runtime_config(str(config_path))

    assert loaded.default_execution_mode == ExecutionMode.INTERACTIVE
    assert loaded.benchmark_enable_pairwise_critique is True
    assert loaded.benchmark_enable_summary_review is False
    assert loaded.interactive_max_models == 1
    assert loaded.interactive_enable_secondary_review is True
    assert loaded.interactive_prefer_remote is True
    assert loaded.interactive_disable_cpu_fallback is True
    assert loaded.interactive_skip_heavy_probe is True
