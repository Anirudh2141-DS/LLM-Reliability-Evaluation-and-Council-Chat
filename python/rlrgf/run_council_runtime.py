"""
CLI for the live council runtime orchestration path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    package_root = Path(__file__).resolve().parents[1]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from rlrgf.council_runtime import CouncilRuntime
    from rlrgf.council_runtime_config import load_runtime_config
    from rlrgf.council_runtime_schemas import CouncilMode, CouncilRequest, ExecutionMode
else:
    from .council_runtime import CouncilRuntime
    from .council_runtime_config import load_runtime_config
    from .council_runtime_schemas import CouncilMode, CouncilRequest, ExecutionMode


def _parse_mode(value: str) -> CouncilMode:
    normalized = value.strip().lower()
    if normalized in {"fast", "fast_council"}:
        return CouncilMode.FAST_COUNCIL
    if normalized in {"full", "full_council"}:
        return CouncilMode.FULL_COUNCIL
    raise argparse.ArgumentTypeError("mode must be one of: fast, full")


def _parse_execution_mode(value: str) -> ExecutionMode:
    normalized = value.strip().lower()
    if normalized == ExecutionMode.INTERACTIVE.value:
        return ExecutionMode.INTERACTIVE
    if normalized == ExecutionMode.BENCHMARK.value:
        return ExecutionMode.BENCHMARK
    raise argparse.ArgumentTypeError("execution mode must be one of: interactive, benchmark")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run live multi-round council runtime.")
    parser.add_argument("query", nargs="?", default="", help="User question to evaluate.")
    parser.add_argument(
        "--query",
        dest="query_option",
        default="",
        help="User question to evaluate (same as positional query).",
    )
    parser.add_argument(
        "--mode",
        type=_parse_mode,
        default=CouncilMode.FAST_COUNCIL,
        help="Council mode: fast or full (default: fast).",
    )
    parser.add_argument(
        "--execution-mode",
        type=_parse_execution_mode,
        default=ExecutionMode.BENCHMARK,
        help="Execution lane: interactive or benchmark (default: benchmark).",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="",
        help="Optional runtime config JSON path (default: config/council_models.json).",
    )
    parser.add_argument(
        "--no-revision-round",
        action="store_true",
        help="Disable the revision round.",
    )
    parser.add_argument(
        "--demo-mode",
        action="store_true",
        help="Enable transcript cache reads/writes.",
    )
    parser.add_argument(
        "--force-live-rerun",
        action="store_true",
        help="Ignore cache and force a fresh live run.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full trace JSON output.",
    )
    parser.add_argument(
        "--use-real-models",
        action="store_true",
        help="Force real remote model calls (expects HF_TOKEN or configured API key).",
    )
    parser.add_argument(
        "--use-mock-models",
        action="store_true",
        help="Force offline mock adapter usage.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="",
        help="Optional file path for writing full trace JSON.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.use_real_models and args.use_mock_models:
        raise SystemExit("Cannot set both --use-real-models and --use-mock-models.")
    query = (args.query_option or args.query or "").strip()
    if not query:
        raise SystemExit("Query is required. Provide positional query text or --query.")

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
        "[council-runtime] "
        f"adapter={adapter_name} "
        f"remote_requests={str(remote_requests).lower()} "
        f"token_found={str(token_found).lower()} "
        f"token_source={token_source} "
        f"base_url={adapter_base_url or '<none>'}"
    )

    request = CouncilRequest(
        query=query,
        mode=args.mode,
        execution_mode=args.execution_mode,
        enable_revision_round=not args.no_revision_round,
        demo_mode=args.demo_mode,
        force_live_rerun=args.force_live_rerun,
    )
    trace = runtime.run(request)
    trace_json = json.dumps(trace.model_dump(mode="json"), indent=2)

    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(trace_json, encoding="utf-8")

    if args.json:
        print(trace_json)
        return 0

    synthesis = trace.final_synthesis
    final_answer = synthesis.final_answer if synthesis is not None else ""
    print(final_answer)
    print("")
    print(
        json.dumps(
            {
                "mode": request.mode.value,
                "execution_mode": request.execution_mode.value,
                "escalated_to_full": trace.escalated_to_full,
                "cached": trace.cached,
                "disagreement_score": trace.disagreement_score,
                "council_confidence": trace.council_confidence,
                "quorum_success": trace.quorum_success,
                "failure_count": len(trace.failures),
                "contract_version": trace.contract_version,
                "cache_status": trace.observability.cache_status.value,
                "effective_mode": trace.observability.effective_mode.value,
                "chair_selected_seat_id": trace.observability.chair_selected_seat_id,
                "fallback_used": (
                    trace.final_synthesis.fallback_used
                    if trace.final_synthesis is not None
                    else True
                ),
            },
            indent=2,
        )
    )
    if args.output_file:
        print(f"\ntrace written to: {args.output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
