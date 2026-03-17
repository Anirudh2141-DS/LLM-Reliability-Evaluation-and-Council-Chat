"""
Configuration for live council runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import os
from pathlib import Path
from typing import Optional

from .council_runtime_schemas import CouncilMode, CouncilSeat


DEFAULT_COUNCIL_CONFIG_PATH = Path("config/council_models.json")
HF_TOKEN_FILE_PATH = Path(
    r"E:\MLOps\LLM Failure Evaluation Engine\python\rlrgf\hf_token.txt"
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_default_config_path() -> Path:
    local_default = DEFAULT_COUNCIL_CONFIG_PATH
    if local_default.exists():
        return local_default
    repo_default = _project_root() / "config" / "council_models.json"
    if repo_default.exists():
        return repo_default
    return local_default


def _seat_env_key(seat_id: str) -> str:
    normalized = "".join(ch if ch.isalnum() else "_" for ch in seat_id).upper()
    return f"COUNCIL_MODEL_{normalized}"


def _seat_copy(seat: CouncilSeat, **updates: object) -> CouncilSeat:
    return seat.model_copy(update=updates)


def _load_hf_token() -> tuple[str, str]:
    if HF_TOKEN_FILE_PATH.exists():
        try:
            file_token = HF_TOKEN_FILE_PATH.read_text(encoding="utf-8").strip()
        except OSError:
            file_token = ""
        if file_token:
            os.environ["HF_TOKEN"] = file_token
            return file_token, "file"

    env_token = os.getenv("HF_TOKEN", "").strip()
    if env_token:
        return env_token, "environment"
    return "", "none"


def _coerce_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _coerce_int(value: object, default: int, *, minimum: Optional[int] = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    if minimum is not None:
        parsed = max(minimum, parsed)
    return parsed


def _coerce_float(
    value: object,
    default: float,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def default_seats() -> list[CouncilSeat]:
    return [
        CouncilSeat(
            seat_id="llama-3-8b",
            role_title="Principal Systems Architect",
            model_id="meta-llama/Meta-Llama-3-8B-Instruct",
            enabled_in_fast_mode=True,
            can_chair=True,
        ),
        CouncilSeat(
            seat_id="gemma-7b",
            role_title="Machine Learning Engineer",
            model_id="google/gemma-1.1-7b-it",
            enabled_in_fast_mode=False,
            can_chair=True,
        ),
        CouncilSeat(
            seat_id="mistral-7b-v0.3",
            role_title="Software Performance Engineer",
            model_id="mistralai/Mistral-7B-Instruct-v0.3",
            enabled_in_fast_mode=False,
            can_chair=True,
        ),
        CouncilSeat(
            seat_id="phi-3-mini",
            role_title="Technical Documentation Specialist",
            model_id="microsoft/Phi-3-mini-4k-instruct",
            enabled_in_fast_mode=True,
            can_chair=True,
        ),
        CouncilSeat(
            seat_id="qwen-2-7b",
            role_title="Security Compliance Officer",
            model_id="Qwen/Qwen2-7B-Instruct",
            enabled_in_fast_mode=True,
            can_chair=True,
        ),
    ]


@dataclass
class CouncilRuntimeConfig:
    base_url: str = ""
    api_key: str = ""
    hf_token_found: bool = False
    hf_token_source: str = "none"
    hf_token_path: str = str(HF_TOKEN_FILE_PATH)
    use_real_models: bool = False
    hf_router_base_url: str = "https://router.huggingface.co/v1"
    request_timeout_s: float = 45.0
    model_timeout_s: float = 30.0
    max_retries: int = 2
    retry_backoff_s: float = 0.8
    retry_backoff_cap_s: float = 4.0
    escalation_disagreement_threshold: float = 0.45
    escalation_confidence_threshold: float = 0.55
    fast_quorum: int = 2
    full_quorum: int = 3
    enable_revision_round: bool = True
    chair_seat_id: str = "llama-3-8b"
    backup_chair_seat_id: str = "phi-3-mini"
    cache_path: str = "./output/council_transcript_cache.json"
    featured_prompts: list[str] = field(
        default_factory=lambda: [
            "Give an architecture recommendation for a secure RAG system in production.",
            "How should we detect and contain prompt-injection in retrieval context?",
            "Design a latency budget strategy for multi-model council chat under 8 seconds.",
            "What governance checks should block release of a new LLM assistant?",
            "How should we evaluate answer quality when retrieval evidence is conflicting?",
        ]
    )
    seats: list[CouncilSeat] = field(default_factory=default_seats)

    def get_seat(self, seat_id: str) -> Optional[CouncilSeat]:
        for seat in self.seats:
            if seat.seat_id == seat_id:
                return seat
        return None

    def active_seats(self, mode: CouncilMode) -> list[CouncilSeat]:
        if mode == CouncilMode.FULL_COUNCIL:
            return list(self.seats)
        return [seat for seat in self.seats if seat.enabled_in_fast_mode]

    def quorum_for_mode(self, mode: CouncilMode) -> int:
        if mode == CouncilMode.FULL_COUNCIL:
            return self.full_quorum
        return self.fast_quorum

    def with_seat_model_overrides(self, model_map: dict[str, str]) -> "CouncilRuntimeConfig":
        seats: list[CouncilSeat] = []
        for seat in self.seats:
            model_id = model_map.get(seat.seat_id, seat.model_id)
            seats.append(_seat_copy(seat, model_id=model_id))
        return replace(self, seats=seats)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_runtime_config(config_path: Optional[str] = None) -> CouncilRuntimeConfig:
    cfg = CouncilRuntimeConfig()

    path = Path(config_path) if config_path else _resolve_default_config_path()
    raw = _read_json(path)

    model_map = raw.get("model_ids", {})
    if isinstance(model_map, dict):
        cfg = cfg.with_seat_model_overrides(
            {str(key): str(value) for key, value in model_map.items()}
        )

    fast_seats = raw.get("fast_seat_ids")
    if isinstance(fast_seats, list):
        fast_set = {str(item) for item in fast_seats}
        cfg.seats = [
            _seat_copy(seat, enabled_in_fast_mode=seat.seat_id in fast_set)
            for seat in cfg.seats
        ]

    cfg.chair_seat_id = str(raw.get("chair_seat_id", cfg.chair_seat_id))
    cfg.backup_chair_seat_id = str(
        raw.get("backup_chair_seat_id", cfg.backup_chair_seat_id)
    )
    cfg.enable_revision_round = _coerce_bool(
        raw.get("enable_revision_round", cfg.enable_revision_round),
        cfg.enable_revision_round,
    )
    cfg.cache_path = str(raw.get("cache_path", cfg.cache_path))
    cfg.hf_router_base_url = str(raw.get("hf_router_base_url", cfg.hf_router_base_url))

    if isinstance(raw.get("featured_prompts"), list):
        cfg.featured_prompts = [str(item) for item in raw["featured_prompts"]]

    cfg.request_timeout_s = _coerce_float(
        raw.get("request_timeout_s", cfg.request_timeout_s),
        cfg.request_timeout_s,
        minimum=1.0,
    )
    cfg.model_timeout_s = _coerce_float(
        raw.get("model_timeout_s", cfg.model_timeout_s),
        cfg.model_timeout_s,
        minimum=1.0,
    )
    cfg.max_retries = _coerce_int(
        raw.get("max_retries", cfg.max_retries),
        cfg.max_retries,
        minimum=0,
    )
    cfg.retry_backoff_s = _coerce_float(
        raw.get("retry_backoff_s", cfg.retry_backoff_s),
        cfg.retry_backoff_s,
        minimum=0.0,
    )
    cfg.retry_backoff_cap_s = _coerce_float(
        raw.get("retry_backoff_cap_s", cfg.retry_backoff_cap_s),
        cfg.retry_backoff_cap_s,
        minimum=0.0,
    )
    cfg.escalation_disagreement_threshold = _coerce_float(
        raw.get(
            "escalation_disagreement_threshold",
            cfg.escalation_disagreement_threshold,
        ),
        cfg.escalation_disagreement_threshold,
        minimum=0.0,
        maximum=1.0,
    )
    cfg.escalation_confidence_threshold = _coerce_float(
        raw.get("escalation_confidence_threshold", cfg.escalation_confidence_threshold),
        cfg.escalation_confidence_threshold,
        minimum=0.0,
        maximum=1.0,
    )
    cfg.fast_quorum = _coerce_int(
        raw.get("fast_quorum", cfg.fast_quorum),
        cfg.fast_quorum,
        minimum=1,
    )
    cfg.full_quorum = _coerce_int(
        raw.get("full_quorum", cfg.full_quorum),
        cfg.full_quorum,
        minimum=1,
    )

    hf_token, hf_token_source = _load_hf_token()
    cfg.hf_token_found = bool(hf_token)
    cfg.hf_token_source = hf_token_source
    cfg.hf_token_path = str(HF_TOKEN_FILE_PATH)

    # Environment overrides for endpoint credentials.
    cfg.base_url = (
        os.getenv("COUNCIL_API_BASE_URL")
        or os.getenv("HF_INFERENCE_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or str(raw.get("base_url", ""))
    )
    cfg.api_key = (
        hf_token
        or os.getenv("COUNCIL_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or str(raw.get("api_key", ""))
    )
    raw_use_real_models = raw.get("use_real_models", cfg.use_real_models)
    cfg.use_real_models = _coerce_bool(raw_use_real_models, cfg.use_real_models)
    env_use_real_models = os.getenv("COUNCIL_USE_REAL_MODELS")
    if env_use_real_models is not None:
        cfg.use_real_models = _coerce_bool(env_use_real_models, cfg.use_real_models)
    elif hf_token:
        # Auto-enable real remote calls when HF credentials are present.
        cfg.use_real_models = True
    if cfg.use_real_models and not cfg.base_url:
        cfg.base_url = cfg.hf_router_base_url

    # Per-seat env override: COUNCIL_MODEL_<SEAT_ID>.
    env_map: dict[str, str] = {}
    for seat in cfg.seats:
        value = os.getenv(_seat_env_key(seat.seat_id))
        if value:
            env_map[seat.seat_id] = value
    if env_map:
        cfg = cfg.with_seat_model_overrides(env_map)

    cache_path = Path(cfg.cache_path)
    if not cache_path.is_absolute():
        base_dir = path.parent if path.exists() else _project_root()
        cfg.cache_path = str((base_dir / cache_path).resolve())

    return cfg
