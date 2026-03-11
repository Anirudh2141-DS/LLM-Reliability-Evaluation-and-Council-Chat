"""
Inference engine with lazy loading, quantization, and CPU fallback support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import gc
import importlib.util
import logging
import os
import time
from typing import Any, Optional, Protocol
from uuid import UUID

from .models import ModelOutput, TokenUsage
from .runtime_dependencies import validate_local_backend_dependencies


logger = logging.getLogger(__name__)

RUNTIME_STATUS_REGISTERED = "REGISTERED"
RUNTIME_STATUS_LOADING = "LOADING"
RUNTIME_STATUS_READY = "READY"
RUNTIME_STATUS_READY_CPU = "READY_CPU"
RUNTIME_STATUS_PROBE_FAILED = "PROBE_FAILED"
RUNTIME_STATUS_UNAVAILABLE = "UNAVAILABLE"
RUNTIME_STATUS_EVICTED = "EVICTED"


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    model_id: str
    preferred_device: str = "cuda"
    quantization_eligible: bool = True
    allow_cpu_fallback: bool = True
    size_class: str = "large"
    trust_remote_code: bool = False
    backend: str = "local"
    requires_hf_token: bool = False


DEFAULT_MODEL_SPECS: dict[str, ModelSpec] = {
    "gemma-7b": ModelSpec(
        alias="gemma-7b",
        model_id="google/gemma-1.1-7b-it",
        requires_hf_token=True,
    ),
    "llama-3-8b": ModelSpec(
        alias="llama-3-8b",
        model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        requires_hf_token=True,
    ),
    "mistral-7b-v0.3": ModelSpec(
        alias="mistral-7b-v0.3",
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
    ),
    "qwen-2-7b": ModelSpec(
        alias="qwen-2-7b",
        model_id="Qwen/Qwen2-7B-Instruct",
    ),
    "phi-3-mini": ModelSpec(
        alias="phi-3-mini",
        model_id="microsoft/Phi-3-mini-4k-instruct",
        size_class="medium",
    ),
}

SIZE_CLASS_PRIORITY = {
    "small": 0,
    "medium": 1,
    "large": 2,
}

FP16_VRAM_ESTIMATE_GB = {
    "small": 4.0,
    "medium": 8.0,
    "large": 16.0,
}


@dataclass
class InferenceConfig:
    """Configuration for LLM inference."""

    model_name: str = "phi-3-mini"
    max_new_tokens: int = 128
    do_sample: bool = False
    use_4bit: bool = True
    device: str = "auto"
    backend: str = "local"
    allow_cpu_fallback: bool = True
    probe_prompt: str = "Reply with the single word: ok."
    probe_max_new_tokens: int = 4
    prefer_small_models_first: bool = True


@dataclass
class LoadedModelHandle:
    spec: ModelSpec
    backend_name: str
    model: Any
    tokenizer: Any
    device: str
    quantization_mode: str
    ready_state: str
    last_probe_latency_ms: Optional[float] = None
    loaded_at: float = field(default_factory=time.monotonic)


class InferenceBackend(Protocol):
    name: str

    def load(self, spec: ModelSpec, config: InferenceConfig) -> LoadedModelHandle:
        ...

    def unload(self, handle: LoadedModelHandle) -> None:
        ...

    def probe(self, handle: LoadedModelHandle, prompt: str, max_new_tokens: int) -> float:
        ...

    def generate(
        self,
        handle: LoadedModelHandle,
        prompt: str,
        max_new_tokens: int,
        do_sample: bool,
    ) -> tuple[str, TokenUsage, float]:
        ...


def resolve_model_spec(model_name: str) -> ModelSpec:
    spec = DEFAULT_MODEL_SPECS.get(model_name)
    if spec is not None:
        return spec
    if "/" in model_name:
        return ModelSpec(alias=model_name, model_id=model_name, size_class="medium")
    return ModelSpec(alias=model_name, model_id=model_name, size_class="medium")


def model_size_priority(model_name: str) -> int:
    spec = resolve_model_spec(model_name)
    return SIZE_CLASS_PRIORITY.get(spec.size_class, 99)


class LocalTransformersBackend:
    name = "local"

    def load(self, spec: ModelSpec, config: InferenceConfig) -> LoadedModelHandle:
        dependency_report = validate_local_backend_dependencies(strict=True)
        if dependency_report.missing_optional:
            logger.debug(
                "Optional local inference dependencies missing for %s: %s",
                spec.alias,
                ", ".join(sorted(dependency_report.missing_optional)),
            )

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        hf_token = os.getenv("HF_TOKEN")
        if spec.requires_hf_token and not hf_token:
            raise RuntimeError(
                f"{spec.alias} requires a Hugging Face token. Set HF_TOKEN before loading."
            )

        attempts = self._build_attempts(spec, config, torch)
        last_error: Optional[Exception] = None

        for attempt in attempts:
            attempt_device = attempt["device"]
            quantization_mode = attempt["quantization_mode"]
            logger.info(
                "Loading %s (%s) via backend=%s device=%s quantization=%s",
                spec.alias,
                spec.model_id,
                self.name,
                attempt_device,
                quantization_mode,
            )
            tokenizer = None
            model = None
            try:
                tokenizer = self._load_tokenizer(
                    AutoTokenizer=AutoTokenizer,
                    spec=spec,
                    hf_token=hf_token,
                )
                model = self._load_model(
                    AutoModelForCausalLM=AutoModelForCausalLM,
                    spec=spec,
                    config=config,
                    torch_module=torch,
                    hf_token=hf_token,
                    attempt=attempt,
                )
                ready_state = (
                    RUNTIME_STATUS_READY_CPU
                    if attempt_device == "cpu"
                    else RUNTIME_STATUS_READY
                )
                return LoadedModelHandle(
                    spec=spec,
                    backend_name=self.name,
                    model=model,
                    tokenizer=tokenizer,
                    device=attempt_device,
                    quantization_mode=quantization_mode,
                    ready_state=ready_state,
                )
            except Exception as error:
                last_error = error
                logger.warning(
                    "%s load failed on %s (%s): %s",
                    spec.alias,
                    attempt_device,
                    quantization_mode,
                    error,
                )
                self._dispose_partial(model=model, tokenizer=tokenizer)
                if attempt_device == "cuda":
                    self._clear_cuda(torch)

        if last_error is None:
            raise RuntimeError(f"{spec.alias} has no viable load attempts configured.")
        raise RuntimeError(str(last_error))

    def unload(self, handle: LoadedModelHandle) -> None:
        logger.info(
            "Unloading %s from device=%s quantization=%s",
            handle.spec.alias,
            handle.device,
            handle.quantization_mode,
        )
        torch_module = self._try_import_torch()
        self._dispose_partial(model=handle.model, tokenizer=handle.tokenizer)
        if torch_module is not None and handle.device == "cuda":
            self._clear_cuda(torch_module)

    def probe(self, handle: LoadedModelHandle, prompt: str, max_new_tokens: int) -> float:
        logger.info(
            "Probing %s on device=%s backend=%s",
            handle.spec.alias,
            handle.device,
            handle.backend_name,
        )
        _, _, latency_ms = self.generate(
            handle=handle,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        return latency_ms

    def generate(
        self,
        handle: LoadedModelHandle,
        prompt: str,
        max_new_tokens: int,
        do_sample: bool,
    ) -> tuple[str, TokenUsage, float]:
        torch = self._require_torch()
        tokenizer = handle.tokenizer
        model = handle.model
        start_time = time.time()
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = self._move_inputs(inputs, handle.device)
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=pad_token_id,
            )

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if answer.startswith(prompt):
            answer = answer[len(prompt) :].strip()
        latency_ms = (time.time() - start_time) * 1000.0
        token_usage = TokenUsage(
            prompt_tokens=len(inputs["input_ids"][0]),
            completion_tokens=len(outputs[0]) - len(inputs["input_ids"][0]),
            total_tokens=len(outputs[0]),
        )
        return answer or "", token_usage, latency_ms

    def _build_attempts(
        self,
        spec: ModelSpec,
        config: InferenceConfig,
        torch_module: Any,
    ) -> list[dict[str, str]]:
        attempts: list[dict[str, str]] = []
        requested_device = config.device.lower()
        cuda_allowed = requested_device in {"auto", "cuda"}
        if cuda_allowed and spec.preferred_device == "cuda" and torch_module.cuda.is_available():
            if config.use_4bit and spec.quantization_eligible and self._bitsandbytes_available():
                attempts.append({"device": "cuda", "quantization_mode": "4bit-nf4"})
            elif self._gpu_full_precision_viable(spec, torch_module):
                attempts.append({"device": "cuda", "quantization_mode": "fp16"})
            else:
                logger.info(
                    "%s skipping non-quantized CUDA load on constrained hardware",
                    spec.alias,
                )
        if config.allow_cpu_fallback and spec.allow_cpu_fallback:
            attempts.append({"device": "cpu", "quantization_mode": "cpu-fp32"})
        return attempts

    def _gpu_full_precision_viable(self, spec: ModelSpec, torch_module: Any) -> bool:
        try:
            total_vram_gb = (
                torch_module.cuda.get_device_properties(0).total_memory / (1024**3)
            )
        except Exception:
            return False
        required = FP16_VRAM_ESTIMATE_GB.get(spec.size_class, 16.0)
        return total_vram_gb >= required

    def _load_tokenizer(
        self,
        AutoTokenizer: Any,
        spec: ModelSpec,
        hf_token: Optional[str],
    ) -> Any:
        common_kwargs = {
            "trust_remote_code": spec.trust_remote_code,
            "token": hf_token,
        }
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                spec.model_id,
                **common_kwargs,
            )
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(
                spec.model_id,
                use_fast=False,
                **common_kwargs,
            )
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load_model(
        self,
        AutoModelForCausalLM: Any,
        spec: ModelSpec,
        config: InferenceConfig,
        torch_module: Any,
        hf_token: Optional[str],
        attempt: dict[str, str],
    ) -> Any:
        common_kwargs = {
            "low_cpu_mem_usage": True,
            "trust_remote_code": spec.trust_remote_code,
            "token": hf_token,
        }
        attempt_device = attempt["device"]
        quantization_mode = attempt["quantization_mode"]
        if attempt_device == "cuda" and quantization_mode == "4bit-nf4":
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_module.float16,
                bnb_4bit_use_double_quant=True,
            )
            return AutoModelForCausalLM.from_pretrained(
                spec.model_id,
                device_map={"": 0},
                quantization_config=quantization_config,
                **common_kwargs,
            )
        if attempt_device == "cuda":
            return AutoModelForCausalLM.from_pretrained(
                spec.model_id,
                device_map={"": 0},
                torch_dtype=torch_module.float16,
                **common_kwargs,
            )
        return AutoModelForCausalLM.from_pretrained(
            spec.model_id,
            device_map="cpu",
            torch_dtype=torch_module.float32,
            **common_kwargs,
        )

    def _move_inputs(self, inputs: Any, device: str) -> Any:
        if device == "cuda":
            return {key: value.to("cuda") for key, value in inputs.items()}
        return inputs

    def _bitsandbytes_available(self) -> bool:
        return importlib.util.find_spec("bitsandbytes") is not None

    def _dispose_partial(self, model: Any, tokenizer: Any) -> None:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()

    def _clear_cuda(self, torch_module: Any) -> None:
        try:
            if torch_module.cuda.is_available():
                torch_module.cuda.empty_cache()
                torch_module.cuda.ipc_collect()
        except Exception:
            logger.debug("CUDA cache clear skipped", exc_info=True)

    def _try_import_torch(self) -> Any | None:
        try:
            import torch

            return torch
        except Exception:
            return None

    def _require_torch(self) -> Any:
        import torch

        return torch


class RemoteInferenceBackend:
    name = "remote"

    def load(self, spec: ModelSpec, config: InferenceConfig) -> LoadedModelHandle:
        raise RuntimeError(
            f"Remote backend for {spec.alias} is not configured. Use backend=local for local inference."
        )

    def unload(self, handle: LoadedModelHandle) -> None:
        return None

    def probe(self, handle: LoadedModelHandle, prompt: str, max_new_tokens: int) -> float:
        raise RuntimeError("Remote backend probe is not implemented.")

    def generate(
        self,
        handle: LoadedModelHandle,
        prompt: str,
        max_new_tokens: int,
        do_sample: bool,
    ) -> tuple[str, TokenUsage, float]:
        raise RuntimeError("Remote backend generate is not implemented.")


def build_backend(backend_name: str) -> InferenceBackend:
    if backend_name == "local":
        return LocalTransformersBackend()
    if backend_name == "remote":
        return RemoteInferenceBackend()
    raise ValueError(f"Unsupported inference backend: {backend_name}")


class InferenceEngine:
    """
    Runtime loader/generator for one model spec with lazy load and fallback support.
    """

    def __init__(
        self,
        config: Optional[InferenceConfig] = None,
        backend: Optional[InferenceBackend] = None,
    ):
        self.config = config or InferenceConfig()
        self.spec = resolve_model_spec(self.config.model_name)
        self.backend = backend or build_backend(self.config.backend or self.spec.backend)
        self.model = None
        self.tokenizer = None
        self._is_loaded = False
        self.loaded_handle: Optional[LoadedModelHandle] = None
        self.runtime_status = RUNTIME_STATUS_REGISTERED
        self.runtime_device: Optional[str] = None
        self.quantization_mode: Optional[str] = None
        self.last_load_error: Optional[str] = None
        self.last_probe_error: Optional[str] = None
        self.last_probe_latency_ms: Optional[float] = None
        self.last_failure_stage: Optional[str] = None
        self.backend_name = self.backend.name

    def load_model(self) -> bool:
        if self._is_loaded and self.loaded_handle is not None:
            return True

        self.runtime_status = RUNTIME_STATUS_LOADING
        self.last_load_error = None
        self.last_probe_error = None
        self.last_failure_stage = None

        try:
            handle = self.backend.load(self.spec, self.config)
            probe_latency_ms = self.backend.probe(
                handle=handle,
                prompt=self.config.probe_prompt,
                max_new_tokens=self.config.probe_max_new_tokens,
            )
        except Exception as error:
            self.last_failure_stage = "load_or_probe"
            self.last_load_error = str(error)
            self.runtime_status = (
                RUNTIME_STATUS_PROBE_FAILED
                if "probe" in str(error).lower()
                else RUNTIME_STATUS_UNAVAILABLE
            )
            logger.error("Model %s failed to become ready: %s", self.spec.alias, error)
            self.unload_model(evicted=False)
            return False

        handle.last_probe_latency_ms = probe_latency_ms
        self.loaded_handle = handle
        self.model = handle.model
        self.tokenizer = handle.tokenizer
        self._is_loaded = True
        self.runtime_device = handle.device
        self.quantization_mode = handle.quantization_mode
        self.last_probe_latency_ms = probe_latency_ms
        self.runtime_status = handle.ready_state
        logger.info(
            "%s loaded on %s with quantization=%s and probe_latency_ms=%.2f",
            self.spec.alias,
            self.runtime_device,
            self.quantization_mode,
            probe_latency_ms,
        )
        return True

    def unload_model(self, evicted: bool = True) -> None:
        if self.loaded_handle is not None:
            try:
                self.backend.unload(self.loaded_handle)
            finally:
                self.loaded_handle = None
                self.model = None
                self.tokenizer = None
                self._is_loaded = False
                self.runtime_device = None
                self.quantization_mode = None
        if evicted:
            self.runtime_status = RUNTIME_STATUS_EVICTED
        elif self.runtime_status == RUNTIME_STATUS_LOADING:
            self.runtime_status = RUNTIME_STATUS_UNAVAILABLE

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded and (self.loaded_handle is not None or self.model is not None)

    def generate(
        self,
        prompt: str,
        query_id: UUID,
        prompt_hash: str = "",
        model_name: str = "unknown",
    ) -> ModelOutput:
        start_time = time.time()

        if self.is_loaded:
            if model_name != self.spec.alias and model_name != self.spec.model_id:
                latency = (time.time() - start_time) * 1000.0
                return ModelOutput(
                    query_id=query_id,
                    generated_answer=(
                        f"[MODEL UNAVAILABLE] The requested model ({model_name}) "
                        f"is not loaded in this runtime. Loaded model: {self.spec.alias}."
                    ),
                    generation_latency_ms=latency,
                    prompt_hash=prompt_hash,
                )
            if self.loaded_handle is None:
                latency = (time.time() - start_time) * 1000.0
                return ModelOutput(
                    query_id=query_id,
                    generated_answer="[RUNTIME ERROR] Model runtime is missing a loaded handle.",
                    generation_latency_ms=latency,
                    prompt_hash=prompt_hash,
                )
            try:
                answer, token_usage, latency_ms = self.backend.generate(
                    handle=self.loaded_handle,
                    prompt=prompt,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=self.config.do_sample,
                )
                return ModelOutput(
                    query_id=query_id,
                    generated_answer=answer,
                    generation_latency_ms=latency_ms,
                    prompt_hash=prompt_hash,
                    token_usage=token_usage,
                )
            except Exception as error:
                self.last_probe_error = str(error)
                self.last_failure_stage = "generate"
                self.runtime_status = RUNTIME_STATUS_PROBE_FAILED
                latency = (time.time() - start_time) * 1000.0
                logger.error("%s inference failed: %s", self.spec.alias, error)
                return ModelOutput(
                    query_id=query_id,
                    generated_answer=f"[RUNTIME ERROR] Model inference failed: {error}",
                    generation_latency_ms=latency,
                    prompt_hash=prompt_hash,
                )

        latency = (time.time() - start_time) * 1000.0
        return ModelOutput(
            query_id=query_id,
            generated_answer=(
                f"[MODEL UNAVAILABLE] The requested model ({model_name}) "
                "is currently offline or not loaded into memory."
            ),
            generation_latency_ms=latency,
            prompt_hash=prompt_hash,
        )
