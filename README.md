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

### 3b. Run Live Council Runtime (No Dashboard)

```bash
cd python
.venv\Scripts\activate

# Fast mode council run
python -m rlrgf.run_council_runtime --query "How should we secure a production RAG stack?" --mode fast

# Interactive lane: lean path, minimal seats, no heavy critique by default
python -m rlrgf.run_council_runtime --query "How should we secure a production RAG stack?" --mode fast --execution-mode interactive

# Benchmark lane: strict full council with full diagnostics
python -m rlrgf.run_council_runtime --query "How should we secure a production RAG stack?" --mode full --execution-mode benchmark

# Full trace JSON for debugging/integration
python -m rlrgf.run_council_runtime --query "How should we secure a production RAG stack?" --mode full --json

# Force real Hugging Face router calls
# Place Hugging Face token in:
# python\rlrgf\hf_token.txt
cd rlrgf
python run_council_runtime.py "Explain exponential backoff" --use-real-models
```

Runtime endpoint settings can be provided with:

- `COUNCIL_API_BASE_URL`
- `COUNCIL_API_KEY`
- `COUNCIL_MODEL_<SEAT_ID>` for per-seat model overrides (example: `COUNCIL_MODEL_LLAMA_3_8B`)
- `HF_TOKEN` (used as fallback only when `python\rlrgf\hf_token.txt` is missing)
- `COUNCIL_USE_REAL_MODELS` (`true`/`false` switch)
- `COUNCIL_EXECUTION_MODE` (`interactive` or `benchmark`)

Adapter behavior:

- default/offline path uses `MockCouncilInferenceAdapter`
- real path uses `HuggingFaceRouterInferenceAdapter` against `https://router.huggingface.co/v1`
- checked-in council seat defaults in `config/council_models.json` are aligned to router-validated chat-capable models for the current provider account; use `COUNCIL_MODEL_<SEAT_ID>` to override per seat
- live runtime uses async network calls for initial, critique, revision, and synthesis stages; the sync adapter API remains for compatibility only
- CLI prints selected adapter, `remote_requests`, token-found status, token source, and base URL (never token value)

Smoke harness:

```bash
cd python
.venv\Scripts\activate

python -m rlrgf.run_council_runtime_smoke --modes fast,full --prompt-pack ..\config\council_smoke_prompts.json

# Benchmark-only smoke validation
python -m rlrgf.run_council_runtime_smoke --modes full --execution-mode benchmark --prompt-pack ..\config\council_smoke_prompts.json

# Optional: force real router calls for smoke runs
python -m rlrgf.run_council_runtime_smoke --modes fast,full --prompt-pack ..\config\council_smoke_prompts.json --use-real-models
```

## Council Runtime Contract (Frozen Pre-Dashboard)

The live runtime now emits a stable typed trace contract (`contract_version: council_runtime_v1`) with:

- request envelope (`CouncilRequest`)
- per-round artifacts (`initial_answers`, `peer_critiques`, `revised_answers`, `final_synthesis`)
- per-seat scorecards (`ModelScoreCard`)
- normalized failure events (`FailureEvent`)
- typed transcript events (`TranscriptEntry`)
- observability block (`requested/effective mode`, `active models`, `escalation`, `chair`, `fallback`, `cache status`, `quorum`, `round stats`)

Key reliability guarantees:

- malformed JSON and empty responses are normalized to explicit failure flags
- synthesis retries backup chair before final fallback
- duplicate failure events are deduplicated
- scorecard build failures degrade gracefully into valid fallback scorecards
- cache entries are schema-versioned to avoid stale contract reuse
- `force_live_rerun` bypasses cache reads

Key observability fields:

- `request_wall_time_ms`: end-to-end wall-clock runtime for the request
- `total_latency_ms`: model-sum latency across transcript events; this can exceed wall-clock latency when fan-out is parallel
- `stage_latency_ms`: wall-clock breakdown for `initial_round`, `critique`, `revision`, and `aggregation`
- `number_of_models_requested`, `number_of_models_succeeded`, `number_of_models_failed`: quorum-relevant seat outcomes
- `critique_enabled`: `true` only when a critique stage actually ran
- `backend_type`: `mock` or `remote`
- `quorum_success`: whether the active lane met quorum after failures/degradation

Token loading and safety:

- `python\rlrgf\hf_token.txt` is repo-local and gitignored
- if the file is absent, `HF_TOKEN` is used instead
- token values are never printed by the CLI or dashboard telemetry

Live-network caveat:

- router/provider support is model-dependent; unsupported models surface as `unavailable_model`
- benchmark runs can degrade if one or more configured seats are unsupported by the current provider account
- provider credit exhaustion can surface as `HTTP 402` during live runs; treat that as an external account-limit condition rather than a council runtime defect
- use `number_of_models_failed`, `failure_flags`, and `stage_latency_ms` to distinguish provider-support failures from latency or parsing problems

## Pre-Dashboard Status

Handled failure/degradation cases include:

- one-model timeout with surviving quorum
- multiple fast-mode failures with controlled escalation
- partial critique/revision failures
- synthesis failure with fallback
- malformed/empty outputs across rounds
- unavailable models during escalation

Still intentionally pending before UI wiring:

- dashboard integration onto this runtime contract
- UI mapping for observability/round stats surfaces
- UX decisions for degraded quorum and fallback messaging

Experimental foundation:

- the repository now also includes a non-default agentic runtime foundation under `python/rlrgf/`
- this path is opt-in, conservative, and bounded; it is not the current dashboard/demo execution path
- the Streamlit dashboard remains evaluation-first and continues to use the stable council runtime by default

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
