from __future__ import annotations

from uuid import uuid4

import pytest

from rlrgf.models import Decision, EvaluationRecord
from rlrgf.predictor import FailureRiskPredictor


def test_predictor_features_distinguish_missing_from_zero() -> None:
    predictor = FailureRiskPredictor()
    records = [
        EvaluationRecord(
            query_id=uuid4(),
            decision=Decision.ACCEPT,
            retrieval_precision_at_k=None,
            retrieval_recall_at_k=None,
            citation_precision=None,
            retrieval_latency_ms=None,
        ),
        EvaluationRecord(
            query_id=uuid4(),
            decision=Decision.ACCEPT,
            retrieval_precision_at_k=0.0,
            retrieval_recall_at_k=0.0,
            citation_precision=0.0,
            retrieval_latency_ms=0.0,
        ),
    ]

    features, _ = predictor.extract_features(records)

    assert features[0, 1] == pytest.approx(1.0)  # retrieval_precision_missing
    assert features[1, 1] == pytest.approx(0.0)
    assert features[0, 3] == pytest.approx(1.0)  # retrieval_recall_missing
    assert features[1, 3] == pytest.approx(0.0)
    assert features[0, 5] == pytest.approx(1.0)  # citation_precision_missing
    assert features[1, 5] == pytest.approx(0.0)
    assert features[0, 10] == pytest.approx(1.0)  # retrieval_latency_missing
    assert features[1, 10] == pytest.approx(0.0)
