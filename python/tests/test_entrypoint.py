from __future__ import annotations

import asyncio
import inspect
import runpy
import sys
import types

import pytest


def test_module_entrypoint_uses_asyncio_run(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel_exit_code = 17

    async def fake_main() -> int:
        return sentinel_exit_code

    fake_run_experiment = types.ModuleType("rlrgf.run_experiment")
    fake_run_experiment.main = fake_main
    monkeypatch.setitem(sys.modules, "rlrgf.run_experiment", fake_run_experiment)

    seen: dict[str, bool] = {}

    def fake_asyncio_run(coro):
        seen["used_coroutine"] = inspect.iscoroutine(coro)
        coro.close()
        return sentinel_exit_code

    monkeypatch.setattr(asyncio, "run", fake_asyncio_run)

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("rlrgf.__main__", run_name="__main__")

    assert seen.get("used_coroutine") is True
    assert exit_info.value.code == sentinel_exit_code
