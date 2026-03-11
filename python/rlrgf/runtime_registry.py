"""
Canonical model runtime registry for availability, telemetry, and failover.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from typing import Callable, Optional
from uuid import UUID
import time

from .inference import (
    InferenceConfig,
    InferenceEngine,
    RUNTIME_STATUS_EVICTED,
    RUNTIME_STATUS_LOADING,
    RUNTIME_STATUS_PROBE_FAILED,
    RUNTIME_STATUS_READY,
    RUNTIME_STATUS_READY_CPU,
    RUNTIME_STATUS_REGISTERED,
    RUNTIME_STATUS_UNAVAILABLE,
    model_size_priority,
    resolve_model_spec,
)
from .models import ModelOutput


logger = logging.getLogger(__name__)

MODEL_STATUS_REGISTERED = RUNTIME_STATUS_REGISTERED
MODEL_STATUS_LOADING = RUNTIME_STATUS_LOADING
MODEL_STATUS_READY = RUNTIME_STATUS_READY
MODEL_STATUS_READY_CPU = RUNTIME_STATUS_READY_CPU
MODEL_STATUS_PROBE_FAILED = RUNTIME_STATUS_PROBE_FAILED
MODEL_STATUS_UNAVAILABLE = RUNTIME_STATUS_UNAVAILABLE
MODEL_STATUS_EVICTED = RUNTIME_STATUS_EVICTED
MODEL_STATUS_NOT_MEASURED = MODEL_STATUS_REGISTERED

READY_STATUSES = {MODEL_STATUS_READY, MODEL_STATUS_READY_CPU}


@dataclass
class ModelRuntimeState:
    model: str
    model_id: str
    preferred_device: str
    quantization_eligible: bool
    status: str = MODEL_STATUS_REGISTERED
    available: bool = False
    resident: bool = False
    backend: str = "local"
    runtime_device: Optional[str] = None
    quantization_mode: Optional[str] = None
    load_attempted: bool = False
    last_probe_at: Optional[float] = None
    last_probe_result: Optional[str] = None
    load_error: Optional[str] = None
    last_error: Optional[str] = None
    last_latency_ms: Optional[float] = None
    updated_at: Optional[str] = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ModelRuntimeRegistry:
    """
    Single source of truth for model registration, lazy loading, eviction, and runtime readiness.
    """

    def __init__(
        self,
        engine_factory: Optional[Callable[[str], InferenceEngine]] = None,
        probe_retry_seconds: float = 15.0,
        max_loaded_cuda_models: int = 1,
    ) -> None:
        self._engine_factory = engine_factory or (
            lambda model_name: InferenceEngine(InferenceConfig(model_name=model_name))
        )
        self._probe_retry_seconds = probe_retry_seconds
        self._max_loaded_cuda_models = max_loaded_cuda_models
        self._engines: dict[str, InferenceEngine] = {}
        self._states: dict[str, ModelRuntimeState] = {}

    def register_models(self, model_names: list[str]) -> None:
        for model_name in model_names:
            self.register_model(model_name)

    def register_model(self, model_name: str) -> ModelRuntimeState:
        state = self._states.get(model_name)
        if state is not None:
            return state
        spec = resolve_model_spec(model_name)
        state = ModelRuntimeState(
            model=model_name,
            model_id=spec.model_id,
            preferred_device=spec.preferred_device,
            quantization_eligible=spec.quantization_eligible,
            backend=spec.backend,
        )
        self._states[model_name] = state
        return state

    def get_state(self, model_name: str) -> ModelRuntimeState:
        return self.register_model(model_name)

    def _get_engine(self, model_name: str) -> InferenceEngine:
        engine = self._engines.get(model_name)
        if engine is None:
            engine = self._engine_factory(model_name)
            self._engines[model_name] = engine
        self._sync_state_from_engine(model_name, engine)
        return engine

    def _sync_state_from_engine(
        self, model_name: str, engine: InferenceEngine
    ) -> ModelRuntimeState:
        state = self.register_model(model_name)
        state.model_id = engine.spec.model_id
        state.preferred_device = engine.spec.preferred_device
        state.quantization_eligible = engine.spec.quantization_eligible
        state.backend = engine.backend_name
        state.runtime_device = engine.runtime_device
        state.quantization_mode = engine.quantization_mode
        state.resident = engine.is_loaded
        state.last_latency_ms = engine.last_probe_latency_ms
        if engine.runtime_status:
            state.status = engine.runtime_status
        state.available = state.status in READY_STATUSES and state.resident
        if engine.last_load_error:
            state.load_error = engine.last_load_error
            state.last_error = engine.last_load_error
        if engine.last_probe_error:
            state.last_error = engine.last_probe_error
        state.updated_at = _utc_now_iso()
        return state

    def ensure_ready_state(self, model_name: str) -> ModelRuntimeState:
        state = self.register_model(model_name)
        engine = self._get_engine(model_name)
        if state.status in READY_STATUSES and engine.is_loaded:
            return self._sync_state_from_engine(model_name, engine)

        now = time.monotonic()
        if state.load_attempted and state.last_probe_at is not None:
            elapsed = now - state.last_probe_at
            if state.status in {MODEL_STATUS_UNAVAILABLE, MODEL_STATUS_PROBE_FAILED}:
                if elapsed < self._probe_retry_seconds:
                    return state

        state.load_attempted = True
        state.last_probe_at = now
        state.status = MODEL_STATUS_LOADING
        state.available = False
        state.updated_at = _utc_now_iso()

        self._evict_for_load(model_name)

        loaded = engine.load_model()
        state = self._sync_state_from_engine(model_name, engine)
        state.last_probe_at = now
        if loaded:
            state.last_probe_result = "probe_ok"
            state.available = state.status in READY_STATUSES
            return state
        state.last_probe_result = "probe_failed"
        state.available = False
        if state.status not in {MODEL_STATUS_UNAVAILABLE, MODEL_STATUS_PROBE_FAILED}:
            state.status = MODEL_STATUS_UNAVAILABLE
        return state

    def unload_model(self, model_name: str, reason: str = "manual") -> ModelRuntimeState:
        state = self.register_model(model_name)
        engine = self._engines.get(model_name)
        if engine is None:
            state.status = MODEL_STATUS_EVICTED
            state.available = False
            state.resident = False
            state.last_probe_result = reason
            state.updated_at = _utc_now_iso()
            return state

        engine.unload_model(evicted=True)
        state = self._sync_state_from_engine(model_name, engine)
        state.status = MODEL_STATUS_EVICTED
        state.available = False
        state.resident = False
        state.last_probe_result = reason
        state.updated_at = _utc_now_iso()
        logger.info("%s evicted: %s", model_name, reason)
        return state

    def _evict_for_load(self, target_model_name: str) -> None:
        target_engine = self._get_engine(target_model_name)
        loaded_models = [
            model_name
            for model_name, engine in self._engines.items()
            if model_name != target_model_name
            and engine.is_loaded
        ]
        if not loaded_models:
            return

        loaded_cuda_models = [
            model_name
            for model_name in loaded_models
            if self._engines[model_name].runtime_device == "cuda"
        ]
        target_priority = model_size_priority(target_model_name)
        should_evict_all = target_priority >= 1
        if (
            target_engine.spec.preferred_device == "cuda"
            and len(loaded_cuda_models) >= self._max_loaded_cuda_models
        ):
            should_evict_all = True
        if not should_evict_all and not loaded_cuda_models:
            return

        for model_name in sorted(
            loaded_models if should_evict_all else loaded_cuda_models,
            key=model_size_priority,
            reverse=True,
        ):
            self.unload_model(model_name, reason=f"evicted for {target_model_name}")

    def order_models_for_request(self, model_names: list[str]) -> list[str]:
        self.register_models(model_names)
        indexed = list(enumerate(model_names))
        indexed.sort(
            key=lambda item: (
                model_size_priority(item[1]),
                item[0],
            )
        )
        return [name for _, name in indexed]

    def generate(
        self,
        model_name: str,
        prompt: str,
        query_id: UUID,
        prompt_hash: str = "",
    ) -> tuple[ModelOutput, ModelRuntimeState]:
        state = self.ensure_ready_state(model_name)
        engine = self._get_engine(model_name)
        output = engine.generate(
            prompt=prompt,
            query_id=query_id,
            prompt_hash=prompt_hash,
            model_name=model_name,
        )
        state = self._update_state_from_output(model_name, output)
        return output, state

    def _update_state_from_output(
        self, model_name: str, output: ModelOutput
    ) -> ModelRuntimeState:
        engine = self._get_engine(model_name)
        state = self._sync_state_from_engine(model_name, engine)
        answer = output.generated_answer or ""
        answer_clean = answer.strip()
        answer_upper = answer_clean.upper()
        state.updated_at = _utc_now_iso()

        if answer_upper.startswith("[MODEL UNAVAILABLE]"):
            state.available = False
            if state.status == MODEL_STATUS_LOADING:
                state.status = MODEL_STATUS_UNAVAILABLE
            if state.status not in {MODEL_STATUS_REGISTERED, MODEL_STATUS_EVICTED}:
                state.status = MODEL_STATUS_UNAVAILABLE
            state.last_error = answer_clean
            state.last_latency_ms = None
        elif answer_upper.startswith("[RUNTIME ERROR]"):
            state.status = MODEL_STATUS_PROBE_FAILED
            state.available = False
            state.last_error = answer_clean
            state.last_latency_ms = None
        else:
            state.status = engine.runtime_status
            state.available = state.status in READY_STATUSES
            state.last_error = None
            state.last_latency_ms = output.generation_latency_ms

        return state


def pick_best_available_output(outputs: list[dict]) -> Optional[dict]:
    def _coerce_float(value: object, default: float) -> float:
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    available_outputs = [
        item
        for item in outputs
        if item.get("available") and item.get("status") in READY_STATUSES
    ]
    if not available_outputs:
        return None
    return sorted(
        available_outputs,
        key=lambda item: (
            1 if item.get("status") == MODEL_STATUS_READY else 0,
            _coerce_float(item.get("supported_claim_ratio"), float("-inf")),
            -_coerce_float(item.get("risk_score"), float("inf")),
        ),
        reverse=True,
    )[0]


def choose_council_answer(council_outputs: list[dict], strict_grounding: bool) -> str:
    if not council_outputs:
        return "I encountered an error while consulting the council."

    best = pick_best_available_output(council_outputs)
    if best is None:
        unavailable = [item.get("model", "unknown") for item in council_outputs]
        unavailable_list = ", ".join(unavailable) if unavailable else "none"
        return (
            "Council unavailable: no eligible models are currently ready for live inference. "
            f"Unavailable models: {unavailable_list}."
        )

    if strict_grounding and (best.get("supported_claim_ratio") or 0.0) < 0.5:
        return (
            "I must abstain from answering as the retrieved evidence is insufficient "
            "to maintain strict grounding."
        )

    return str(best.get("raw_answer", "")).strip()


def summarize_model_health(
    model_name: str, entries: list[dict], runtime_state: ModelRuntimeState
) -> dict:
    def _is_error_answer(raw_answer: object) -> bool:
        text = str(raw_answer or "").strip().upper()
        return text.startswith("[MODEL UNAVAILABLE]") or text.startswith("[RUNTIME ERROR]")

    ready_entries = [
        e
        for e in entries
        if (
            e.get("available")
            and e.get("status") in READY_STATUSES
            and not _is_error_answer(e.get("raw_answer"))
        )
    ]

    if not ready_entries:
        return {
            "model": model_name,
            "model_id": runtime_state.model_id,
            "health": None,
            "status": runtime_state.status,
            "avg_latency_ms": None,
            "avg_support": None,
            "avg_risk": None,
            "policy_flags": 0,
            "fallback_rate": None,
            "available": runtime_state.available,
            "resident": runtime_state.resident,
            "backend": runtime_state.backend,
            "runtime_device": runtime_state.runtime_device,
            "quantization_mode": runtime_state.quantization_mode,
            "last_probe_result": runtime_state.last_probe_result,
            "error": (
                runtime_state.last_error or runtime_state.load_error
                if runtime_state.status in {MODEL_STATUS_PROBE_FAILED, MODEL_STATUS_UNAVAILABLE}
                else None
            ),
        }

    latencies = [e.get("latency_ms") for e in ready_entries if e.get("latency_ms") is not None]
    supports = [
        e.get("supported_claim_ratio")
        for e in ready_entries
        if e.get("supported_claim_ratio") is not None
    ]
    risks = [e.get("risk_score") for e in ready_entries if e.get("risk_score") is not None]
    fallback_count = sum(
        1
        for e in ready_entries
        if "i don't have enough" in str(e.get("raw_answer", "")).lower()
    )
    fallback_rate = fallback_count / len(ready_entries)

    avg_latency = (sum(latencies) / len(latencies)) if latencies else None
    avg_support = (sum(supports) / len(supports)) if supports else None
    avg_risk = (sum(risks) / len(risks)) if risks else None

    health = None
    if avg_support is not None and avg_risk is not None and avg_latency is not None:
        latency_component = max(0.0, 1.0 - (avg_latency / 1200.0))
        quality_component = max(0.0, 1.0 - fallback_rate)
        health = int(
            round(
                (
                    avg_support * 0.35
                    + (1.0 - avg_risk) * 0.25
                    + latency_component * 0.10
                    + quality_component * 0.30
                )
                * 100
            )
        )

    return {
        "model": model_name,
        "model_id": runtime_state.model_id,
        "health": health,
        "status": runtime_state.status,
        "avg_latency_ms": round(avg_latency, 1) if avg_latency is not None else None,
        "avg_support": round(avg_support, 2) if avg_support is not None else None,
        "avg_risk": round(avg_risk, 2) if avg_risk is not None else None,
        "policy_flags": int(
            sum(
                1
                for e in ready_entries
                if any(
                    f in e.get("flags", [])
                    for f in (
                        "policy-risk",
                        "injection-following",
                        "confident-without-support",
                    )
                )
            )
        ),
        "fallback_rate": round(fallback_rate, 2),
        "available": runtime_state.available,
        "resident": runtime_state.resident,
        "backend": runtime_state.backend,
        "runtime_device": runtime_state.runtime_device,
        "quantization_mode": runtime_state.quantization_mode,
        "last_probe_result": runtime_state.last_probe_result,
        "error": None,
    }
