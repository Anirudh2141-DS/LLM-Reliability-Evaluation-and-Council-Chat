"""
Report Generator - Produces experiment reports, datasets, and visualizations.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import EvaluationRecord, ExperimentMetrics


class ReportGenerator:
    """
    Generates experimental artifacts:
    - JSONL dataset export
    - Metrics summary JSON
    - Visualizations (matplotlib)
    - Text report
    """

    def __init__(self, output_dir: str = "./output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_dataset(
        self,
        records: list[EvaluationRecord],
        filename: str = "evaluation_dataset.jsonl",
    ) -> Path:
        """Export evaluation records as a JSONL file."""
        filepath = self.output_dir / filename
        with open(filepath, "w") as f:
            for record in records:
                # Use model_dump_json() if pydantic v2, else json()
                if hasattr(record, "model_dump_json"):
                    f.write(record.model_dump_json() + "\n")
                else:
                    f.write(record.json() + "\n")
        return filepath

    def export_metrics(
        self,
        metrics: ExperimentMetrics,
        filename: str = "experiment_metrics.json",
    ) -> Path:
        """Export experiment metrics as a JSON file."""
        filepath = self.output_dir / filename
        with open(filepath, "w") as f:
            if hasattr(metrics, "model_dump_json"):
                f.write(metrics.model_dump_json(indent=2))
            else:
                f.write(metrics.json(indent=2))
        return filepath

    def generate_text_report(
        self,
        metrics: ExperimentMetrics,
        records: list[EvaluationRecord],
        filename: str = "experiment_report.txt",
    ) -> Path:
        """Generate a human-readable text report."""
        filepath = self.output_dir / filename
        def fmt_optional(value: Optional[float], format_spec: str = ".3f") -> str:
            if value is None:
                return "N/A"
            return format(value, format_spec)

        lines = [
            "=" * 70,
            "  RLRGF - Experiment Report",
            f"  Experiment ID: {metrics.experiment_id}",
            f"  Generated: {datetime.utcnow().isoformat()}",
            "=" * 70,
            "",
            "--- CORE METRICS ---------------------------------------",
            f"  Total Queries Evaluated:  {metrics.total_queries}",
            f"  Refusal Rate:             {metrics.refusal_rate:.1%}",
            f"  Hallucination Rate:       {metrics.hallucination_rate:.1%}",
            f"  Policy Violation Rate:    {metrics.policy_violation_rate:.1%}",
            f"  Leakage Rate:             {metrics.leakage_rate:.1%}",
            f"  Injection Success Rate:   {metrics.injection_success_rate:.1%}",
            f"  Instability Rate:         {metrics.instability_rate:.1%}",
            "",
            "--- RAG METRICS ----------------------------------------",
            f"  Retrieval Precision@K:    {fmt_optional(metrics.avg_retrieval_precision_at_k)}",
            f"  Retrieval Recall@K:       {fmt_optional(metrics.avg_retrieval_recall_at_k)}",
            f"  Citation Precision:       {fmt_optional(metrics.avg_citation_precision)}",
            f"  Supported Claim Ratio:    {metrics.avg_supported_claim_ratio:.3f}",
            "",
            "--- COUNCIL METRICS ------------------------------------",
            f"  Disagreement Score (avg): {metrics.council_disagreement_avg:.3f}",
            f"  Safety Override Freq:     {metrics.safety_override_frequency:.1%}",
            f"  Self-Correction Rate:     {metrics.self_correction_rate:.1%}",
            "",
            "--- SYSTEM METRICS -------------------------------------",
            f"  P50 Latency:              {metrics.p50_latency_ms:.1f} ms",
            f"  P95 Latency:              {metrics.p95_latency_ms:.1f} ms",
            f"  Avg Context Length:       {metrics.avg_context_length_tokens:.0f} tokens",
            "",
            "--- FAILURE BREAKDOWN ----------------------------------",
        ]

        # Count failure types
        from collections import Counter
        failure_counts = Counter(
            r.failure_type.value if r.failure_type else "none"
            for r in records
        )
        for ftype, count in failure_counts.most_common():
            pct = count / max(len(records), 1) * 100
            lines.append(f"  {ftype:25s}  {count:4d}  ({pct:.1f}%)")

        lines.extend([
            "",
            "--- DECISION DISTRIBUTION ------------------------------",
        ])
        decision_counts = Counter(r.decision.value for r in records)
        for decision, count in decision_counts.most_common():
            pct = count / max(len(records), 1) * 100
            lines.append(f"  {decision:25s}  {count:4d}  ({pct:.1f}%)")

        lines.extend(["", "=" * 70, "  END OF REPORT", "=" * 70])

        report_text = "\n".join(lines)
        with open(filepath, "w") as f:
            f.write(report_text)

        return filepath

    def generate_visualizations(
        self,
        metrics: ExperimentMetrics,
        records: list[EvaluationRecord],
    ) -> list[Path]:
        """Generate visualizations."""
        generated_files: list[Path] = []

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("Warning: matplotlib not available, skipping visualizations")
            return generated_files

        vis_dir = self.output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)

        # Basic failure rate chart
        fig, ax = plt.subplots(figsize=(10, 6))
        labels = ["Hallucination", "Policy", "Leakage", "Injection", "Instability"]
        vals = [metrics.hallucination_rate, metrics.policy_violation_rate, 
                metrics.leakage_rate, metrics.injection_success_rate, metrics.instability_rate]
        ax.bar(labels, vals, color="#4D96FF")
        ax.set_title("Failure Rates")
        fig.savefig(vis_dir / "failure_rates.png")
        plt.close(fig)
        generated_files.append(vis_dir / "failure_rates.png")

        return generated_files
