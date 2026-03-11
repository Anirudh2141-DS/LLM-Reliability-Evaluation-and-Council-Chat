from __future__ import annotations

import pytest

from rlrgf import runtime_dependencies


def test_runtime_dependency_validation_reports_missing_required(monkeypatch) -> None:
    def fake_find_spec(module_name: str):
        if module_name == "torch":
            return None
        return object()

    monkeypatch.setattr(runtime_dependencies.importlib.util, "find_spec", fake_find_spec)

    report = runtime_dependencies.validate_local_backend_dependencies()

    assert "torch" in report.missing_required
    assert "Missing dependency: torch." in report.format_error()


def test_runtime_dependency_validation_raises_in_strict_mode(monkeypatch) -> None:
    def fake_find_spec(module_name: str):
        if module_name in {"torch", "transformers"}:
            return None
        return object()

    monkeypatch.setattr(runtime_dependencies.importlib.util, "find_spec", fake_find_spec)

    with pytest.raises(RuntimeError) as error:
        runtime_dependencies.validate_local_backend_dependencies(strict=True)

    message = str(error.value)
    assert "Missing dependency: torch." in message
    assert "Missing dependency: transformers." in message
