"""
Pipeline Orchestrator - End-to-end experiment runner for the RLRGF framework.
"""

from __future__ import annotations

import time
import os
from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID, uuid4

from .models import (
    EvaluationRecord,
    ExperimentMetrics,
    ModelOutput,
    RetrievedChunkRef,
)
from .guardrails import GuardrailConfig, GuardrailProcessor
from .inference import InferenceConfig, InferenceEngine
from .council import (
    CouncilAggregator,
    CouncilConfig,
    Critic,
    GroundingInspector,
    SafetyAuditor,
)
from .classifier import FailureClassifier
from .metrics import MetricsEngine
from .reporting import ReportGenerator
from .predictor import FailureRiskPredictor, PredictorConfig
from .audit import AuditLogger

# Mock rich console for Windows compatibility
class SafeConsole:
    def print(self, *args, **kwargs):
        import re
        processed = [re.sub(r'\[/?[^\]]+\]', '', str(arg)) for arg in args]
        print(*processed)
    def log(self, *args, **kwargs):
        import re
        processed = [re.sub(r'\[/?[^\]]+\]', '', str(arg)) for arg in args]
        print(*processed)

console = SafeConsole()

def track(iterable, description=""):
    print(description)
    return iterable

class Table:
    def __init__(self, *args, **kwargs): pass
    def add_column(self, *args, **kwargs): pass
    def add_row(self, *args, **kwargs):
        print(f"  {args[0]}: {args[1]}")


@dataclass
class TestCase:
    """A single test case for evaluation."""
    query_id: UUID = field(default_factory=uuid4)
    query: str = ""
    expected_answer: Optional[str] = None
    adversarial: bool = False
    category: str = "general"
    expected_doc_ids: set[UUID] = field(default_factory=set)


@dataclass
class PipelineConfig:
    """Configuration for the full evaluation pipeline."""
    experiment_id: str = "exp_001"
    output_dir: str = "./output"
    guardrail_config: GuardrailConfig = field(default_factory=GuardrailConfig)
    inference_config: InferenceConfig = field(default_factory=InferenceConfig)
    council_config: CouncilConfig = field(default_factory=CouncilConfig)
    predictor_config: PredictorConfig = field(default_factory=PredictorConfig)
    load_model: bool = False


class EvaluationPipeline:
    """
    End-to-end evaluation pipeline orchestrator.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

        self.guardrails = GuardrailProcessor(self.config.guardrail_config)
        self.inference = InferenceEngine(self.config.inference_config)
        self.grounding = GroundingInspector()
        self.safety = SafetyAuditor()
        self.critic = Critic()
        self.aggregator = CouncilAggregator(self.config.council_config)
        self.classifier = FailureClassifier()
        self.metrics_engine = MetricsEngine()
        self.reporter = ReportGenerator(self.config.output_dir)
        self.predictor = FailureRiskPredictor(self.config.predictor_config)
        self.audit = AuditLogger(os.path.join(self.config.output_dir, "audit"))

        if self.config.load_model:
            self.inference.load_model()

    async def run_experiment(self, test_cases: list[TestCase], retrieval_map: dict[UUID, list[RetrievedChunkRef]]) -> list[EvaluationRecord]:
        """Run multi-model benchmarking experiment."""
        console.print("\n+----------------------------------------------+")
        console.print("|   RLRGF Evaluation Pipeline - Starting      |")
        console.print(f"|   Experiment: {self.config.experiment_id:30s} |")
        console.print(f"|   Test Cases: {len(test_cases):30d} |")
        console.print("+----------------------------------------------+\n")

        records: list[EvaluationRecord] = []
        models_to_benchmark = ["llama-3-8b", "phi-3-mini", "mistral-7b-v0.3", "gemma-7b", "qwen-2-7b"]
        
        sequence_id = 0
        for test_case in track(test_cases, description="[green]Evaluating Multimodal Council..."):
            sequence_id += 1
            retrieved_chunks, retrieval_latency_ms = self._lookup_retrieval(
                retrieval_map, test_case.query_id
            )
            for evaluator_model in models_to_benchmark:
                record = await self.evaluate_test_case(
                    test_case,
                    retrieved_chunks,
                    evaluator_model,
                    sequence_id,
                    retrieval_latency_ms=retrieval_latency_ms,
                )
                records.append(record)
                
        # Compute metrics and generate reports
        metrics = self.metrics_engine.compute(records, self.config.experiment_id)
        
        console.print("\n[yellow]Generating reports...[/]")
        self.reporter.export_dataset(records)
        self.reporter.export_metrics(metrics)
        self.reporter.generate_text_report(metrics, records)
        vis_files = self.reporter.generate_visualizations(metrics, records)

        if len(records) >= 10:
            console.print("[yellow]Training failure risk predictor...[/]")
            self.predictor.train(records)
            self.predictor.save_report(self.config.output_dir)

        self._print_summary(metrics, records, vis_files)
        return records

    async def evaluate_test_case(
        self,
        test_case: TestCase,
        chunks: list[RetrievedChunkRef],
        evaluator_model: str,
        sequence_id: int,
        retrieval_latency_ms: Optional[float] = None,
    ) -> EvaluationRecord:
        """Process a single query with a specific model."""

        # Step 1: Guardrails
        guardrail_result = self.guardrails.process(test_case.query, chunks)

        # Step 2: Inference
        model_output = self.inference.generate(
            guardrail_result.prompt,
            test_case.query_id,
            guardrail_result.prompt_hash,
            model_name=evaluator_model
        )

        # Step 3: Council evaluators
        g_verdict = await self.grounding.evaluate(
            test_case.query_id, model_output.generated_answer, 
            guardrail_result.processed_chunks, model_name=evaluator_model,
            sequence_id=sequence_id
        )
        s_verdict = await self.safety.evaluate(
            test_case.query_id, model_output.generated_answer, 
            test_case.query, guardrail_result.injection_flags, 
            model_name=evaluator_model
        )
        c_verdict = await self.critic.evaluate(
            test_case.query_id, model_output.generated_answer, 
            guardrail_result.processed_chunks, model_name=evaluator_model,
            sequence_id=sequence_id
        )

        # Step 4: Aggregation
        council_result = self.aggregator.aggregate(test_case.query_id, [g_verdict, s_verdict, c_verdict])

        retrieval_precision, retrieval_recall, citation_precision = (
            self._compute_retrieval_quality(
                test_case=test_case,
                chunks=guardrail_result.processed_chunks,
                model_output=model_output,
            )
        )

        # Step 5: Classification
        record = self.classifier.classify(
            council_result=council_result,
            model_output=model_output,
            retrieval_chunks=guardrail_result.processed_chunks,
            injection_flags=guardrail_result.injection_flags,
            retrieval_latency_ms=retrieval_latency_ms,
            retrieval_precision_at_k=retrieval_precision,
            retrieval_recall_at_k=retrieval_recall,
            citation_precision=citation_precision,
            experiment_id=self.config.experiment_id,
            evaluator_model=evaluator_model,
            sequence_id=sequence_id,
            category=test_case.category,
            adversarial=test_case.adversarial
        )

        # Audit logging
        self.audit.log_retrieval(
            str(test_case.query_id),
            len(chunks),
            retrieval_latency_ms,
        )
        self.audit.log_council_evaluation(str(test_case.query_id), council_result.final_decision.value, record.risk_score)
        
        return record

    def _lookup_retrieval(
        self,
        retrieval_map: dict[UUID, list[RetrievedChunkRef]],
        query_id: UUID,
    ) -> tuple[list[RetrievedChunkRef], float]:
        start = time.perf_counter()
        chunks = list(retrieval_map.get(query_id, []))
        latency_ms = (time.perf_counter() - start) * 1000.0
        return chunks, latency_ms

    def _compute_retrieval_quality(
        self,
        test_case: TestCase,
        chunks: list[RetrievedChunkRef],
        model_output: ModelOutput,
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:
        relevant_doc_ids = set(test_case.expected_doc_ids)
        retrieved_doc_ids = {chunk.doc_id for chunk in chunks}
        retrieved_chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}

        precision_at_k: Optional[float] = None
        recall_at_k: Optional[float] = None
        if relevant_doc_ids:
            relevant_hits = len(retrieved_doc_ids & relevant_doc_ids)
            precision_at_k = (
                relevant_hits / len(retrieved_doc_ids) if retrieved_doc_ids else 0.0
            )
            recall_at_k = relevant_hits / len(relevant_doc_ids)

        citation_precision: Optional[float] = None
        if model_output.citations:
            supported = 0
            for citation in model_output.citations:
                retrieved_chunk = retrieved_chunk_by_id.get(citation.chunk_id)
                if not retrieved_chunk:
                    continue
                if (not relevant_doc_ids) or (retrieved_chunk.doc_id in relevant_doc_ids):
                    supported += 1
            citation_precision = supported / len(model_output.citations)

        return precision_at_k, recall_at_k, citation_precision

    def _print_summary(self, metrics: ExperimentMetrics, records: list[EvaluationRecord], vis_files: list) -> None:
        console.print("\n*** Experiment Complete ***\n")
        table = Table()
        table.add_row("Total Evaluations", str(len(records)))
        table.add_row("Benchmarked Models", "Llama, Phi, Mistral, Gemma, Qwen")
        table.add_row("Avg Risk Score", f"{metrics.avg_risk_score:.2f}")
        console.print(f"Reports saved to: {self.config.output_dir}")
