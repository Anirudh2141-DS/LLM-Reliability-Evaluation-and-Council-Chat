from __future__ import annotations

import sys
import types

from rlrgf.inference import (
    InferenceConfig,
    LocalTransformersBackend,
    ModelSpec,
    RUNTIME_STATUS_READY,
    RUNTIME_STATUS_READY_CPU,
)
from rlrgf.runtime_dependencies import DependencyReport


class _FakeCuda:
    def __init__(self, available: bool = True, total_memory_gb: float = 4.0) -> None:
        self._available = available
        self._total_memory = int(total_memory_gb * (1024**3))

    def is_available(self) -> bool:
        return self._available

    def get_device_properties(self, index: int):
        return types.SimpleNamespace(total_memory=self._total_memory)

    def empty_cache(self) -> None:
        return None

    def ipc_collect(self) -> None:
        return None


def _install_fake_modules(monkeypatch, model_loader):
    fake_torch = types.SimpleNamespace(
        cuda=_FakeCuda(),
        float16="float16",
        float32="float32",
    )

    class _Tokenizer:
        pad_token = None
        eos_token = "</s>"
        pad_token_id = 0
        eos_token_id = 0

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class _BitsAndBytesConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=_Tokenizer,
        AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=model_loader),
        BitsAndBytesConfig=_BitsAndBytesConfig,
    )

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)


def test_quantized_cuda_path_selected_when_supported(monkeypatch) -> None:
    calls: list[dict] = []

    def _model_loader(model_id: str, **kwargs):
        calls.append(kwargs)
        return object()

    _install_fake_modules(monkeypatch, _model_loader)
    monkeypatch.setattr(
        "rlrgf.inference.validate_local_backend_dependencies",
        lambda strict=True: DependencyReport({}, {}),
    )

    backend = LocalTransformersBackend()
    monkeypatch.setattr(backend, "_bitsandbytes_available", lambda: True)

    handle = backend.load(
        ModelSpec(alias="phi-3-mini", model_id="microsoft/Phi-3-mini-4k-instruct", size_class="medium"),
        InferenceConfig(model_name="phi-3-mini"),
    )

    assert handle.device == "cuda"
    assert handle.quantization_mode == "4bit-nf4"
    assert handle.ready_state == RUNTIME_STATUS_READY
    assert calls[0]["device_map"] == {"": 0}
    assert calls[0]["quantization_config"] is not None


def test_gpu_failure_falls_back_to_cpu(monkeypatch) -> None:
    calls: list[dict] = []

    def _model_loader(model_id: str, **kwargs):
        calls.append(kwargs)
        if kwargs["device_map"] == {"": 0}:
            raise RuntimeError("CUDA out of memory")
        return object()

    _install_fake_modules(monkeypatch, _model_loader)
    monkeypatch.setattr(
        "rlrgf.inference.validate_local_backend_dependencies",
        lambda strict=True: DependencyReport({}, {}),
    )

    backend = LocalTransformersBackend()
    monkeypatch.setattr(backend, "_bitsandbytes_available", lambda: True)

    handle = backend.load(
        ModelSpec(alias="qwen-2-7b", model_id="Qwen/Qwen2-7B-Instruct"),
        InferenceConfig(model_name="qwen-2-7b"),
    )

    assert len(calls) == 2
    assert calls[0]["device_map"] == {"": 0}
    assert calls[1]["device_map"] == "cpu"
    assert handle.device == "cpu"
    assert handle.quantization_mode == "cpu-fp32"
    assert handle.ready_state == RUNTIME_STATUS_READY_CPU
