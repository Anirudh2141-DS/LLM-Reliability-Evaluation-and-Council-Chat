import asyncio
import argparse
import sys
from uuid import UUID

class Console:
    def print(self, *args, **kwargs):
        import re
        processed = [re.sub(r'\[/?[^\]]+\]', '', str(arg)) for arg in args]
        print(*processed)
    def log(self, *args, **kwargs): print(*args)

console = Console()

from .pipeline import EvaluationPipeline, PipelineConfig, TestCase
from .guardrails import GuardrailConfig
from .inference import InferenceConfig
from .council import CouncilConfig
from .predictor import PredictorConfig
from .synthetic import SyntheticDataGenerator


async def main():
    parser = argparse.ArgumentParser(description="RLRGF Evaluation Runner")
    parser.add_argument("--experiment-id", default="exp_001")
    parser.add_argument("--output-dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--n-normal", type=int, default=15)
    parser.add_argument("--n-adversarial", type=int, default=8)
    parser.add_argument("--n-leakage", type=int, default=5, help="Number of leakage test cases")
    parser.add_argument("--n-jailbreak", type=int, default=5, help="Number of jailbreak test cases")
    parser.add_argument("--n-unicode", type=int, default=5, help="Number of unicode attack test cases")
    parser.add_argument("--n-ambiguous", type=int, default=5)
    parser.add_argument("--load-model", action="store_true")
    parser.add_argument("--model-name", default="microsoft/phi-2")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    console.print("\n+--------------------------------------------------------------+")
    console.print("|      RAG-LLM Reliability Evaluation & Governance Framework  |")
    console.print("|                         RLRGF v0.1.0                        |")
    console.print("+--------------------------------------------------------------+\n")

    config = PipelineConfig(
        experiment_id=args.experiment_id,
        output_dir=args.output_dir,
        guardrail_config=GuardrailConfig(),
        inference_config=InferenceConfig(model_name=args.model_name),
        load_model=args.load_model,
    )

    generator = SyntheticDataGenerator(seed=args.seed)
    test_cases, retrieval_results = generator.generate(
        n_normal=args.n_normal,
        n_adversarial=args.n_adversarial,
        n_leakage=args.n_leakage,
        n_jailbreak=args.n_jailbreak,
        n_unicode=args.n_unicode,
        n_ambiguous=args.n_ambiguous,
    )

    console.print(f"Generating synthetic test data... ({len(test_cases)} cases)")
    
    pipeline = EvaluationPipeline(config)
    await pipeline.run_experiment(test_cases, retrieval_results)

    console.print("[OK] Experiment complete!\n")
    return 0


if __name__ == "__main__":
    asyncio.run(main())
