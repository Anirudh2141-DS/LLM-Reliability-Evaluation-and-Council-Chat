from __future__ import annotations

from pathlib import Path

from rlrgf import council_chatbot


def test_streamlit_is_declared_dependency() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    raw = pyproject.read_text(encoding="utf-8")
    assert "streamlit" in raw


def test_legacy_council_chatbot_delegates_to_dashboard(monkeypatch) -> None:
    seen = {}

    def fake_run_module(name: str, run_name: str | None = None):
        seen["name"] = name
        seen["run_name"] = run_name

    monkeypatch.setattr(council_chatbot.runpy, "run_module", fake_run_module)

    council_chatbot.main()

    assert seen["name"] == "rlrgf.dashboard"
    assert seen["run_name"] == "__main__"
