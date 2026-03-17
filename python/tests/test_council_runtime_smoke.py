from __future__ import annotations

import json
from pathlib import Path

from rlrgf.run_council_runtime_smoke import _load_prompt_pack, _parse_modes
from rlrgf.council_runtime_schemas import CouncilMode


def test_parse_modes_accepts_fast_and_full() -> None:
    modes = _parse_modes("fast,full")
    assert modes == [CouncilMode.FAST_COUNCIL, CouncilMode.FULL_COUNCIL]


def test_load_prompt_pack_requires_non_empty_queries(tmp_path: Path) -> None:
    prompt_pack = tmp_path / "prompts.json"
    prompt_pack.write_text(
        json.dumps({"prompts": [{"id": "p1", "category": "test", "query": "hello"}]}),
        encoding="utf-8",
    )

    loaded = _load_prompt_pack(prompt_pack)

    assert loaded[0]["id"] == "p1"
    assert loaded[0]["query"] == "hello"
