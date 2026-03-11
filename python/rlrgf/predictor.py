"""
ML Failure Risk Predictor - Trains models to predict failure probability.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .models import EvaluationRecord


@dataclass
class PredictorConfig:
    """Configuration for the ML failure risk predictor."""
    algorithm: str = "gradient_boosting"
    test_size: float = 0.2
    random_state: int = 42


class FailureRiskPredictor:
    """
    Trains and evaluates ML models to predict failure probabilities.
    """

    FEATURE_NAMES = [
        "retrieval_precision_at_k",
        "retrieval_precision_missing",
        "retrieval_recall_at_k",
        "retrieval_recall_missing",
        "citation_precision",
        "citation_precision_missing",
        "supported_claim_ratio",
        "context_token_count",
        "generation_latency_ms",
        "retrieval_latency_ms",
        "retrieval_latency_missing",
        "risk_score",
    ]

    TARGET_NAMES = ["hallucination", "policy_violation", "leakage"]

    def __init__(self, config: Optional[PredictorConfig] = None):
        self.config = config or PredictorConfig()
        self.models: dict[str, Any] = {}
        self.metrics: dict[str, dict[str, float]] = {}
        self._fitted = False

    def extract_features(self, records: list[EvaluationRecord]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Extract feature matrix and target vectors."""
        features = []
        targets = {name: [] for name in self.TARGET_NAMES}

        for record in records:
            retrieval_precision, retrieval_precision_missing = self._encode_optional_metric(
                record.retrieval_precision_at_k
            )
            retrieval_recall, retrieval_recall_missing = self._encode_optional_metric(
                record.retrieval_recall_at_k
            )
            citation_precision, citation_precision_missing = self._encode_optional_metric(
                record.citation_precision
            )
            retrieval_latency, retrieval_latency_missing = self._encode_optional_metric(
                record.retrieval_latency_ms
            )
            row = [
                retrieval_precision,
                retrieval_precision_missing,
                retrieval_recall,
                retrieval_recall_missing,
                citation_precision,
                citation_precision_missing,
                record.supported_claim_ratio,
                record.context_token_count / 4096.0,
                record.generation_latency_ms / 10000.0,
                retrieval_latency / 1000.0,
                retrieval_latency_missing,
                record.risk_score,
            ]
            features.append(row)

            targets["hallucination"].append(int(record.hallucination_flag))
            targets["policy_violation"].append(int(record.policy_violation))
            targets["leakage"].append(int(record.leakage_detected))

        X = np.array(features, dtype=np.float32)
        y = {name: np.array(vals, dtype=np.int32) for name, vals in targets.items()}

        return X, y

    def train(self, records: list[EvaluationRecord]) -> dict[str, dict[str, float]]:
        """Train models."""
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, f1_score
        except ImportError:
            print("Warning: scikit-learn not available, skipping ML training.")
            return {}

        X, y_dict = self.extract_features(records)
        if len(X) < 10:
            return {}

        all_metrics = {}
        for target_name, y in y_dict.items():
            if y.sum() == 0 or y.sum() == len(y): continue

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config.test_size, random_state=self.config.random_state
            )

            model = GradientBoostingClassifier(n_estimators=100, random_state=self.config.random_state)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            all_metrics[target_name] = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            }
            self.models[target_name] = model

        self.metrics = all_metrics
        self._fitted = bool(self.models)
        return all_metrics

    def predict(self, records: list[EvaluationRecord]) -> dict[str, np.ndarray]:
        """Predict risk."""
        if not self._fitted: return {}
        X, _ = self.extract_features(records)
        return {name: model.predict_proba(X)[:, 1] for name, model in self.models.items()}

    def save_report(self, output_dir: str = "./output") -> Path:
        """Save report."""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        filepath = path / "ml_predictor_report.json"
        
        report = {
            "algorithm": self.config.algorithm,
            "features": self.FEATURE_NAMES,
            "metrics": self.metrics,
            "fitted": self._fitted,
        }
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)
        return filepath

    @staticmethod
    def _encode_optional_metric(value: Optional[float]) -> tuple[float, float]:
        if value is None:
            return 0.0, 1.0
        return float(value), 0.0
