from __future__ import annotations

import json
from dataclasses import dataclass
from uuid import uuid4

from rlrgf.models import ModelOutput
from rlrgf.runtime_registry import (
    MODEL_STATUS_EVICTED,
    MODEL_STATUS_PROBE_FAILED,
    MODEL_STATUS_READY,
    MODEL_STATUS_READY_CPU,
    MODEL_STATUS_REGISTERED,
    MODEL_STATUS_UNAVAILABLE,
    ModelRuntimeRegistry,
    choose_council_answer,
    summarize_model_health,
)


@dataclass(frozen=True)
class _FakeSpec:
    alias: str
    model_id: str
    preferred_device: str = "cuda"
    quantization_eligible: bool = True


class _FakeEngine:
    def __init__(
        self,
        model_name: str,
        load_plan: list[dict] | None = None,
        response: str = "ready answer",
        generation_latency_ms: float = 25.0,
    ) -> None:
        self.spec = _FakeSpec(alias=model_name, model_id=f"hf/{model_name}")
        self.backend_name = "local"
        self.runtime_status = MODEL_STATUS_REGISTERED
        self.runtime_device = None
        self.quantization_mode = None
        self.last_load_error = None
        self.last_probe_error = None
        self.last_probe_latency_ms = None
        self._response = response
        self._generation_latency_ms = generation_latency_ms
        self._is_loaded = False
        self.model = None
        self.tokenizer = None
        self.load_calls = 0
        self.unload_calls = 0
        self._load_plan = load_plan or [
            {
                "success": True,
                "status": MODEL_STATUS_READY,
                "device": "cuda",
                "quantization_mode": "4bit-nf4",
                "probe_latency_ms": 11.0,
            }
        ]

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def load_model(self) -> bool:
        self.load_calls += 1
        plan = self._load_plan[min(self.load_calls - 1, len(self._load_plan) - 1)]
        if not plan["success"]:
            self._is_loaded = False
            self.model = None
            self.tokenizer = None
            self.runtime_status = plan["status"]
            self.runtime_device = None
            self.quantization_mode = None
            self.last_load_error = plan.get("error", "load failed")
            self.last_probe_latency_ms = None
            return False

        self._is_loaded = True
        self.model = object()
        self.tokenizer = object()
        self.runtime_status = plan["status"]
        self.runtime_device = plan["device"]
        self.quantization_mode = plan["quantization_mode"]
        self.last_load_error = None
        self.last_probe_latency_ms = plan.get("probe_latency_ms")
        return True

    def unload_model(self, evicted: bool = True) -> None:
        self.unload_calls += 1
        self._is_loaded = False
        self.model = None
        self.tokenizer = None
        self.runtime_device = None
        self.quantization_mode = None
        self.runtime_status = MODEL_STATUS_EVICTED if evicted else MODEL_STATUS_UNAVAILABLE

    def generate(
        self, prompt: str, query_id, prompt_hash: str = "", model_name: str = "unknown"
    ) -> ModelOutput:
        if not self._is_loaded:
            return ModelOutput(
                query_id=query_id,
                generated_answer=f"[MODEL UNAVAILABLE] {model_name} offline",
                generation_latency_ms=0.0,
                prompt_hash=prompt_hash,
            )
        if self._response.startswith("[RUNTIME ERROR]"):
            self.runtime_status = MODEL_STATUS_PROBE_FAILED
        return ModelOutput(
            query_id=query_id,
            generated_answer=self._response,
            generation_latency_ms=self._generation_latency_ms,
            prompt_hash=prompt_hash,
        )


def _build_output_entry(model: str, runtime_state, model_output: ModelOutput) -> dict:
    ready = runtime_state.status in {MODEL_STATUS_READY, MODEL_STATUS_READY_CPU}
    return {
        "model": model,
        "status": runtime_state.status,
        "available": runtime_state.available,
        "raw_answer": model_output.generated_answer,
        "supported_claim_ratio": 0.8 if ready else None,
        "risk_score": 0.2 if ready else None,
        "latency_ms": model_output.generation_latency_ms if ready else None,
    }


def test_startup_registers_models_without_eager_loading() -> None:
    engines = {
        name: _FakeEngine(name)
        for name in ("gemma-7b", "llama-3-8b", "phi-3-mini")
    }
    registry = ModelRuntimeRegistry(engine_factory=lambda model: engines[model])

    registry.register_models(["gemma-7b", "llama-3-8b", "phi-3-mini"])

    assert all(engine.load_calls == 0 for engine in engines.values())
    assert registry.get_state("gemma-7b").status == MODEL_STATUS_REGISTERED


def test_lazy_load_happens_on_first_inference_request() -> None:
    engine = _FakeEngine("phi-3-mini")
    registry = ModelRuntimeRegistry(engine_factory=lambda model: engine)

    output, state = registry.generate("phi-3-mini", "q", uuid4(), "h")

    assert engine.load_calls == 1
    assert state.status == MODEL_STATUS_READY
    assert output.generated_answer == "ready answer"


def test_unavailable_models_do_not_block_available_ones() -> None:
    engines = {
        "gemma-7b": _FakeEngine(
            "gemma-7b",
            load_plan=[
                {
                    "success": False,
                    "status": MODEL_STATUS_UNAVAILABLE,
                    "error": "token required",
                }
            ],
        ),
        "phi-3-mini": _FakeEngine(
            "phi-3-mini",
            load_plan=[
                {
                    "success": True,
                    "status": MODEL_STATUS_READY_CPU,
                    "device": "cpu",
                    "quantization_mode": "cpu-fp32",
                    "probe_latency_ms": 14.0,
                }
            ],
            response="phi answer",
        ),
    }
    registry = ModelRuntimeRegistry(engine_factory=lambda model: engines[model])

    outputs = []
    for model in ["gemma-7b", "phi-3-mini"]:
        output, state = registry.generate(model, "question", uuid4(), "hash")
        outputs.append(_build_output_entry(model, state, output))

    answer = choose_council_answer(outputs, strict_grounding=False)
    assert answer == "phi answer"


def test_telemetry_distinguishes_ready_cpu_from_registered() -> None:
    engines = {
        "gemma-7b": _FakeEngine("gemma-7b"),
        "phi-3-mini": _FakeEngine(
            "phi-3-mini",
            load_plan=[
                {
                    "success": True,
                    "status": MODEL_STATUS_READY_CPU,
                    "device": "cpu",
                    "quantization_mode": "cpu-fp32",
                    "probe_latency_ms": 8.0,
                }
            ],
            response="phi answer",
        ),
    }
    registry = ModelRuntimeRegistry(engine_factory=lambda model: engines[model])
    registry.register_models(["gemma-7b", "phi-3-mini"])

    output, state = registry.generate("phi-3-mini", "q", uuid4(), "h")
    phi_row = summarize_model_health(
        "phi-3-mini",
        [_build_output_entry("phi-3-mini", state, output)],
        registry.get_state("phi-3-mini"),
    )
    gemma_row = summarize_model_health("gemma-7b", [], registry.get_state("gemma-7b"))

    assert phi_row["status"] == MODEL_STATUS_READY_CPU
    assert phi_row["runtime_device"] == "cpu"
    assert gemma_row["status"] == MODEL_STATUS_REGISTERED
    assert gemma_row["avg_latency_ms"] is None


def test_evicted_models_are_not_marked_failed() -> None:
    engines = {
        "qwen-2-7b": _FakeEngine("qwen-2-7b"),
        "phi-3-mini": _FakeEngine("phi-3-mini"),
    }
    registry = ModelRuntimeRegistry(engine_factory=lambda model: engines[model])

    registry.generate("qwen-2-7b", "q", uuid4(), "h1")
    evicted_state = registry.unload_model("qwen-2-7b", reason="evicted for phi-3-mini")
    row = summarize_model_health("qwen-2-7b", [], evicted_state)

    assert evicted_state.status == MODEL_STATUS_EVICTED
    assert row["status"] == MODEL_STATUS_EVICTED
    assert row["error"] is None


def test_eviction_frees_runtime_objects_and_updates_state() -> None:
    engines = {
        "mistral-7b-v0.3": _FakeEngine("mistral-7b-v0.3"),
        "qwen-2-7b": _FakeEngine("qwen-2-7b"),
    }
    registry = ModelRuntimeRegistry(
        engine_factory=lambda model: engines[model],
        max_loaded_cuda_models=1,
    )

    registry.generate("mistral-7b-v0.3", "q", uuid4(), "h1")
    registry.generate("qwen-2-7b", "q", uuid4(), "h2")

    mistral_state = registry.get_state("mistral-7b-v0.3")
    qwen_state = registry.get_state("qwen-2-7b")

    assert engines["mistral-7b-v0.3"].unload_calls == 1
    assert mistral_state.status == MODEL_STATUS_EVICTED
    assert mistral_state.resident is False
    assert qwen_state.status == MODEL_STATUS_READY


def test_probe_failure_isolated_per_model() -> None:
    engines = {
        "gemma-7b": _FakeEngine("gemma-7b", response="[RUNTIME ERROR] boom"),
        "llama-3-8b": _FakeEngine("llama-3-8b", response="llama answer"),
    }
    registry = ModelRuntimeRegistry(engine_factory=lambda model: engines[model])

    _, gemma_state = registry.generate("gemma-7b", "q", uuid4(), "h1")
    gemma_status = gemma_state.status
    _, llama_state = registry.generate("llama-3-8b", "q", uuid4(), "h2")

    assert gemma_status == MODEL_STATUS_PROBE_FAILED
    assert llama_state.status == MODEL_STATUS_READY


def test_health_row_serialization_keeps_missing_values_as_null() -> None:
    registry = ModelRuntimeRegistry(engine_factory=lambda model: _FakeEngine(model))
    registry.register_model("gemma-7b")

    row = summarize_model_health("gemma-7b", [], registry.get_state("gemma-7b"))
    payload = json.dumps(row)

    assert '"avg_latency_ms": null' in payload
    assert '"avg_support": null' in payload
    assert '"avg_risk": null' in payload
