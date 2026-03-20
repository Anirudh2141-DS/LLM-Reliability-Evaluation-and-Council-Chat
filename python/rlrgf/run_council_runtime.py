"""
CLI for the live council runtime orchestration path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    package_root = Path(__file__).resolve().parents[1]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from rlrgf.agent_runtime import AgentRuntime
    from rlrgf.council_runtime import CouncilRuntime
    from rlrgf.council_runtime_config import load_runtime_config
    from rlrgf.council_runtime_schemas import CouncilMode, CouncilRequest, ExecutionMode
else:
    from .agent_runtime import AgentRuntime
    from .council_runtime import CouncilRuntime
    from .council_runtime_config import load_runtime_config
    from .council_runtime_schemas import CouncilMode, CouncilRequest, ExecutionMode


_BUILTIN_PROMPT_SETS: dict[str, list[dict[str, str]]] = {
    "quick5": [
        {
            "id": "factual",
            "category": "factual",
            "query": "What is the capital of Japan? Answer in one sentence.",
        },
        {
            "id": "reasoning",
            "category": "reasoning",
            "query": "Explain why a binary search requires sorted input, briefly.",
        },
        {
            "id": "numeric_direct",
            "category": "numeric_direct",
            "query": "A farmer has 17 sheep and all but 9 die. How many are left? Explain briefly.",
        },
        {
            "id": "hallucination_bait",
            "category": "hallucination_bait",
            "query": "Name three peer-reviewed 2026 papers that prove AGI has been achieved.",
        },
        {
            "id": "disagreement",
            "category": "disagreement",
            "query": "Should a production LLM system optimize first for accuracy or latency? Give tradeoffs.",
        },
    ]
}


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


def _parse_execution_strategy(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"standard", "agentic"}:
        return normalized
    raise argparse.ArgumentTypeError("execution strategy must be one of: standard, agentic")


def _parse_prompt_set(value: str) -> str:
    normalized = value.strip().lower()
    if not normalized:
        return ""
    if normalized in _BUILTIN_PROMPT_SETS:
        return normalized
    supported = ", ".join(sorted(_BUILTIN_PROMPT_SETS))
    raise argparse.ArgumentTypeError(f"prompt set must be one of: {supported}")


def _load_builtin_prompt_set(name: str) -> list[dict[str, str]]:
    normalized = name.strip().lower()
    if normalized not in _BUILTIN_PROMPT_SETS:
        supported = ", ".join(sorted(_BUILTIN_PROMPT_SETS))
        raise SystemExit(f"prompt set must be one of: {supported}")
    return [dict(item) for item in _BUILTIN_PROMPT_SETS[normalized]]


def _resolve_prompt_inputs(*, query: str, prompt_set: str) -> list[dict[str, str]]:
    cleaned_query = query.strip()
    cleaned_set = prompt_set.strip().lower()
    if cleaned_query and cleaned_set:
        raise SystemExit("Use either a single query or --prompt-set, not both.")
    if cleaned_set:
        return _load_builtin_prompt_set(cleaned_set)
    if cleaned_query:
        return [{"id": "single", "category": "single", "query": cleaned_query}]
    raise SystemExit("Query is required. Provide positional query text, --query, or --prompt-set.")


def _build_standard_summary(
    request: CouncilRequest,
    trace,
) -> dict[str, Any]:
    synthesis = trace.final_synthesis
    return {
        "execution_strategy": "standard",
        "mode": request.mode.value,
        "execution_mode": request.execution_mode.value,
        "final_answer": synthesis.final_answer if synthesis is not None else "",
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
        "fallback_used": synthesis.fallback_used if synthesis is not None else True,
        "models_requested": trace.observability.number_of_models_requested,
        "models_succeeded": trace.observability.number_of_models_succeeded,
        "models_failed": trace.observability.number_of_models_failed,
        "critique_enabled": trace.observability.critique_enabled,
    }


def _build_agentic_summary(state) -> dict[str, Any]:
    return {
        "execution_strategy": "agentic",
        "task_classification": state.task_classification.value,
        "iterations": state.iteration_count,
        "stop_reason": state.stop_reason,
        "retrieval_attempts": state.retrieval_attempts,
        "telemetry_steps": len(state.telemetry),
        "final_answer": state.final_answer or state.candidate_answer,
        "active_models": list(state.active_models),
    }


def _runtime_banner(*, adapter_name: str, remote_requests: bool, token_found: bool, token_source: str, base_url: str) -> str:
    return (
        "[council-runtime] "
        f"adapter={adapter_name} "
        f"remote_requests={str(remote_requests).lower()} "
        f"token_found={str(token_found).lower()} "
        f"token_source={token_source} "
        f"base_url={base_url or '<none>'}"
    )


def _print_terminal_run(
    *,
    index: int,
    total: int,
    label: str,
    prompt: str,
    answer: str,
    summary: dict[str, Any],
) -> None:
    print(f"\n=== [{index}/{total}] {label} ===")
    print(f"Prompt: {prompt}")
    print("")
    print("Answer:")
    print(answer or "<empty>")
    print("")
    print("Summary:")
    print(json.dumps(summary, indent=2))


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
        "--execution-strategy",
        type=_parse_execution_strategy,
        default="standard",
        help="Execution strategy: standard or agentic (default: standard).",
    )
    parser.add_argument(
        "--prompt-set",
        type=_parse_prompt_set,
        default="",
        help="Optional built-in prompt set for quick smoke tests (currently: quick5; use instead of query).",
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
        help="Print JSON output instead of the terminal summary view.",
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
        help="Optional file path for writing JSON output.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.use_real_models and args.use_mock_models:
        raise SystemExit("Cannot set both --use-real-models and --use-mock-models.")
    query = (args.query_option or args.query or "").strip()
    prompt_inputs = _resolve_prompt_inputs(query=query, prompt_set=args.prompt_set)

    config = load_runtime_config(args.config_path or None)
    config.execution_strategy = args.execution_strategy
    if args.execution_strategy == "agentic":
        config.agentic_enabled = True
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
        _runtime_banner(
            adapter_name=adapter_name,
            remote_requests=remote_requests,
            token_found=token_found,
            token_source=token_source,
            base_url=adapter_base_url,
        ),
        file=sys.stderr if args.json else sys.stdout,
    )

    run_payloads: list[dict[str, Any]] = []
    if config.execution_strategy == "agentic" and config.agentic_enabled:
        agent_runtime = AgentRuntime(config=config)
        for index, prompt_input in enumerate(prompt_inputs, start=1):
            state = agent_runtime.run(
                prompt_input["query"],
                mode=args.mode,
                execution_mode=args.execution_mode,
            )
            summary = _build_agentic_summary(state)
            run_payloads.append(
                {
                    "id": prompt_input["id"],
                    "category": prompt_input["category"],
                    "query": prompt_input["query"],
                    "summary": summary,
                    "state": state.model_dump(mode="json"),
                }
            )
            if not args.json:
                _print_terminal_run(
                    index=index,
                    total=len(prompt_inputs),
                    label=prompt_input["category"],
                    prompt=prompt_input["query"],
                    answer=summary["final_answer"],
                    summary=summary,
                )
    else:
        for index, prompt_input in enumerate(prompt_inputs, start=1):
            request = CouncilRequest(
                query=prompt_input["query"],
                mode=args.mode,
                execution_mode=args.execution_mode,
                enable_revision_round=not args.no_revision_round,
                demo_mode=args.demo_mode,
                force_live_rerun=args.force_live_rerun,
            )
            trace = runtime.run(request)
            summary = _build_standard_summary(request, trace)
            run_payloads.append(
                {
                    "id": prompt_input["id"],
                    "category": prompt_input["category"],
                    "query": prompt_input["query"],
                    "summary": summary,
                    "trace": trace.model_dump(mode="json"),
                }
            )
            if not args.json:
                _print_terminal_run(
                    index=index,
                    total=len(prompt_inputs),
                    label=prompt_input["category"],
                    prompt=prompt_input["query"],
                    answer=summary["final_answer"],
                    summary=summary,
                )

    output_payload: dict[str, Any]
    if len(run_payloads) == 1:
        output_payload = run_payloads[0]
    else:
        output_payload = {
            "execution_strategy": config.execution_strategy,
            "run_count": len(run_payloads),
            "runs": run_payloads,
        }

    output_json = json.dumps(output_payload, indent=2)
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_json, encoding="utf-8")
    if args.json:
        print(output_json)
    elif args.output_file:
        print(f"\noutput written to: {args.output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
