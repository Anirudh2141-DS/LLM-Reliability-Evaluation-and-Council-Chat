# RAG-LLM Reliability Evaluation & Governance Framework (RLRGF)

A reproducible evaluation harness and governance layer for Retrieval-Augmented Generation systems that detects, classifies, and quantifies failures in LLM-RAG pipelines.

## Architecture

```
Query → Retrieval → Context Assembly → Guardrails → LLM Generation → Council Review → Governance Policy → Final Output
                                                          ↓
                                            Failure Classification → Metrics → Report
```

### Key Components

| Layer | Technology | Responsibility |
|-------|-----------|----------------|
| **Ingestion & Retrieval** | Rust (tokio, axum, Qdrant) | Document chunking, embedding storage, vector search |
| **Guardrails** | Python | Context budgeting, injection detection, PII redaction |
| **Inference** | Python (transformers, bitsandbytes) | LLM generation with 4-bit quantization |
| **Council** | Python | Multi-model evaluation (Grounding, Safety, Critic) |
| **Classification** | Python | Failure labeling (hallucination, leakage, injection, etc.) |
| **Metrics** | Python | Aggregate experiment metrics computation |
| **ML Predictor** | Python (scikit-learn) | Failure risk prediction models |
| **Audit** | Rust (PostgreSQL, JSONL) | Append-only audit logging |

## Project Structure

```
├── crates/                     # Rust workspace
│   ├── models/                 # Shared data models
│   ├── ingestion/              # Document ingestion & chunking
│   ├── retrieval/              # Vector search & filtering
│   ├── audit/                  # Audit logging
│   └── api-server/             # Axum HTTP API
├── python/                     # Python evaluation layer
│   └── rlrgf/
│       ├── models.py           # Pydantic data models
│       ├── guardrails.py       # Context budget, injection detection, PII
│       ├── inference.py        # LLM inference engine
│       ├── council.py          # Multi-model evaluator council
│       ├── classifier.py       # Failure classification
│       ├── metrics.py          # Metrics computation engine
│       ├── predictor.py        # ML failure risk predictor
│       ├── reporting.py        # Report & visualization generation
│       ├── synthetic.py        # Synthetic test data generator
│       ├── pipeline.py         # End-to-end orchestrator
│       └── run_experiment.py   # CLI experiment runner
├── config/                     # Configuration files
├── sql/                        # Database schemas
└── output/                     # Experiment outputs (generated)
```

## Quick Start

### Prerequisites

- Rust 1.75+ (for backend)
- Python 3.10+ (for evaluation)
- PostgreSQL (optional, for production audit logs)
- Qdrant (optional, for production vector search)

### 1. Build Rust Backend

```bash
cargo build --release
cargo test
```

### 2. Setup Python Environment

```bash
cd python
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -e ".[dev]"
```

### 3. Run Evaluation Experiment

```bash
# Mock mode (no GPU required)
python -m rlrgf --experiment-id exp_001 --output-dir ./output

# With real model (GPU required)
python -m rlrgf --experiment-id exp_001 --load-model --model-name microsoft/phi-2
```

### 4. Start Rust API Server

```bash
cargo run -p rlrgf-api-server
```

## Evaluation Pipeline

### Test Case Categories

| Category | Description | Testing Focus |
|----------|-------------|---------------|
| Normal | Well-grounded, safe queries | Baseline accuracy |
| Adversarial | Prompt injection attempts | Guardrail effectiveness |
| Leakage | Queries targeting internals | Data protection |
| Ambiguous | Unclear/vague queries | Uncertainty handling |

### Council Evaluators

1. **Grounding Inspector** — Verifies claims against retrieved context
2. **Safety Auditor** — Detects policy violations, leakage, injection compliance
3. **Critic** — Detects hallucination patterns and reasoning errors

### Decision Policy

| Condition | Decision |
|-----------|----------|
| Safety violation detected | **Abstain** |
| High unsupported claims | **Request Retrieval** |
| Below grounding threshold | **Escalate** |
| High council disagreement | **Ask Clarification** |
| All checks pass | **Accept** |

### Failure Types

| Type | Description |
|------|-------------|
| Hallucination | Unsupported claims not grounded in context |
| Policy Violation | Unsafe or prohibited content generated |
| Leakage | Sensitive data exposed in output |
| Injection Success | Retrieved document overrides instructions |
| Instability | Inconsistent answers across evaluations |

## Metrics

### Core Metrics

- Refusal rate, hallucination rate, policy violation rate, leakage rate

### RAG Metrics

- Retrieval precision@k, recall@k, citation precision, supported claim ratio

### Council Metrics

- Disagreement score, safety override frequency, self-correction rate

### System Metrics

- P50/P95 latency, context length vs failure rate

## Outputs

The system produces:

- **JSONL Dataset** — Full evaluation records for downstream analysis
- **Metrics JSON** — Aggregate experiment metrics
- **Text Report** — Human-readable experiment summary
- **Visualizations** — 5 matplotlib charts (failure rates, decisions, risk, latency, quality radar)
- **ML Report** — Failure risk predictor training results

## License

MIT
