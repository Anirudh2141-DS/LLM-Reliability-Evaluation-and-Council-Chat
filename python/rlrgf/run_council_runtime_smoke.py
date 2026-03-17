"""
Lightweight smoke harness for the live council runtime.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from .council_runtime import CouncilRuntime
from .council_runtime_config import load_runtime_config
from .council_runtime_schemas import CouncilMode, CouncilRequest


def _parse_modes(raw: str) -> list[CouncilMode]:
    items = [part.strip().lower() for part in raw.split(",") if part.strip()]
    modes: list[CouncilMode] = []
    for item in items:
        if item in {"fast", "fast_council"}:
            modes.append(CouncilMode.FAST_COUNCIL)
        elif item in {"full", "full_council"}:
            modes.append(CouncilMode.FULL_COUNCIL)
        else:
            raise ValueError(f"unsupported mode: {item}")
    if not modes:
        raise ValueError("at least one mode is required")
    return modes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run council runtime smoke prompts.")
    parser.add_argument(
        "--prompt-pack",
        type=str,
        default="config/council_smoke_prompts.json",
        help="Path to prompt pack JSON.",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="fast,full",
        help="Comma-separated modes: fast,full",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="",
        help="Optional runtime config JSON path.",
    )
    parser.add_argument(
        "--demo-mode",
        action="store_true",
        help="Enable cache usage during smoke runs.",
    )
    parser.add_argument(
        "--force-live-rerun",
        action="store_true",
        help="Bypass cache reads and force fresh calls.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="",
        help="Output JSON file path. Default writes to output/smoke/ with timestamp.",
    )
    parser.add_argument(
        "--include-full-trace",
        action="store_true",
        help="Include full trace payload for each run in output JSON.",
    )
    parser.add_argument(
        "--use-real-models",
        action="store_true",
        help="Force real remote model calls.",
    )
    parser.add_argument(
        "--use-mock-models",
        action="store_true",
        help="Force offline mock adapter usage.",
    )
    return parser


def _load_prompt_pack(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    prompts = raw.get("prompts", [])
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("prompt pack must define a non-empty 'prompts' list")
    normalized: list[dict] = []
    for index, item in enumerate(prompts, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"prompt entry #{index} must be an object")
        query = str(item.get("query", "")).strip()
        if not query:
            raise ValueError(f"prompt entry #{index} is missing a non-empty query")
        normalized.append(
            {
                "id": str(item.get("id", f"prompt_{index:02d}")).strip(),
                "category": str(item.get("category", "general")).strip(),
                "query": query,
            }
        )
    return normalized


def _resolve_prompt_pack_path(path: Path) -> Path:
    if path.exists():
        return path
    repo_default = Path(__file__).resolve().parents[2] / "config" / path.name
    if repo_default.exists():
        return repo_default
    return path


def _default_output_path() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("output") / "smoke" / f"council_runtime_smoke_{stamp}.json"


def main() -> int:
    args = build_parser().parse_args()
    if args.use_real_models and args.use_mock_models:
        raise SystemExit("Cannot set both --use-real-models and --use-mock-models.")

    modes = _parse_modes(args.modes)
    prompt_pack_path = _resolve_prompt_pack_path(Path(args.prompt_pack))
    prompts = _load_prompt_pack(prompt_pack_path)

    config = load_runtime_config(args.config_path or None)
    if args.use_real_models:
        config.use_real_models = True
        if not config.base_url:
            config.base_url = config.hf_router_base_url
    elif args.use_mock_models:
        config.use_real_models = False

    runtime = CouncilRuntime(config=config)
    adapter_name = type(runtime.adapter).__name__
    remote_requests = adapter_name != "MockCouncilInferenceAdapter"
    adapter_base_url = getattr(runtime.adapter, "base_url", "")
    token_found = bool((runtime.config.api_key or "").strip())
    token_source = str(getattr(runtime.config, "hf_token_source", "none")).strip().lower()
    if token_source not in {"file", "environment"}:
        token_source = "environment" if token_found else "none"
    print(
        "[council-runtime-smoke] "
        f"adapter={adapter_name} "
        f"remote_requests={str(remote_requests).lower()} "
        f"token_found={str(token_found).lower()} "
        f"token_source={token_source} "
        f"base_url={adapter_base_url or '<none>'}"
    )

    runs: list[dict] = []
    for prompt in prompts:
        for mode in modes:
            request = CouncilRequest(
                query=prompt["query"],
                mode=mode,
                demo_mode=args.demo_mode,
                force_live_rerun=args.force_live_rerun,
            )
            trace = runtime.run(request)
            run_payload = {
                "prompt_id": prompt["id"],
                "category": prompt["category"],
                "query": prompt["query"],
                "mode": mode.value,
                "contract_version": trace.contract_version,
                "final_answer": (
                    trace.final_synthesis.final_answer if trace.final_synthesis else ""
                ),
                "escalated_to_full": trace.escalated_to_full,
                "fallback_used": (
                    trace.final_synthesis.fallback_used if trace.final_synthesis else True
                ),
                "quorum_success": trace.quorum_success,
                "cache_status": trace.observability.cache_status.value,
                "failure_events": [
                    failure.model_dump(mode="json") for failure in trace.failures
                ],
                "scorecards": [
                    scorecard.model_dump(mode="json") for scorecard in trace.scorecards
                ],
                "round_stats": [
                    stat.model_dump(mode="json") for stat in trace.observability.round_stats
                ],
                "transcript_event_count": len(trace.transcript),
            }
            if args.include_full_trace:
                run_payload["trace"] = trace.model_dump(mode="json")
            runs.append(run_payload)

    summary = {
        "total_runs": len(runs),
        "total_prompts": len(prompts),
        "modes": [mode.value for mode in modes],
        "escalation_count": sum(1 for run in runs if run["escalated_to_full"]),
        "fallback_count": sum(1 for run in runs if run["fallback_used"]),
        "quorum_failure_count": sum(1 for run in runs if not run["quorum_success"]),
        "runs_with_failures": sum(1 for run in runs if run["failure_events"]),
    }
    output_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "prompt_pack_path": str(prompt_pack_path),
        "summary": summary,
        "runs": runs,
    }

    output_path = Path(args.output_file) if args.output_file else _default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    print(f"smoke output written to: {output_path}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
