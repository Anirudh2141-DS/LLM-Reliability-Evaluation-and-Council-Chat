from __future__ import annotations

from uuid import uuid4

from rlrgf.inference import InferenceConfig, InferenceEngine


def test_loaded_engine_does_not_masquerade_other_models() -> None:
    engine = InferenceEngine(InferenceConfig(model_name="llama-3-8b"))
    engine._is_loaded = True
    engine.model = object()
    engine.tokenizer = object()

    output = engine.generate(
        prompt="hello",
        query_id=uuid4(),
        prompt_hash="hash",
        model_name="gemma-7b",
    )

    assert output.generated_answer.startswith("[MODEL UNAVAILABLE]")
    assert "Loaded model: llama-3-8b" in output.generated_answer
