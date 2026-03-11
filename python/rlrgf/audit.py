"""
Audit Logger - Records structured interaction events.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class AuditLogger:
    """
    Append-only audit logger for RLRGF.
    """

    def __init__(self, log_dir: str = "./output/audit"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "audit.jsonl"

    def log_event(self, event_type: str, component: str, details: dict[str, Any]) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "component": component,
            "details": details,
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def log_retrieval(
        self, query_id: str, result_count: int, latency_ms: Optional[float]
    ) -> None:
        self.log_event("retrieval", "retrieval", {"query_id": query_id, "count": result_count, "latency": latency_ms})

    def log_council_evaluation(self, query_id: str, decision: str, risk_score: float) -> None:
        self.log_event("council_eval", "council", {"query_id": query_id, "decision": decision, "risk": risk_score})

    def log_failure(self, query_id: str, failure_type: str, details: dict[str, Any]) -> None:
        self.log_event("failure", "classifier", {"query_id": query_id, "type": failure_type, "details": details})
