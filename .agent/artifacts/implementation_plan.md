# RAG-LLM Reliability Evaluation & Governance Framework (RLRGF)
## Implementation Plan

### Phase 1: Project Scaffolding & Core Data Models
- [ ] Create Rust workspace (Cargo workspace for backend services)
- [ ] Create Python package structure (evaluation, inference, analytics)
- [ ] Define shared data models (Document, Chunk, RetrievalLog, ModelOutput, CouncilVerdict, EvaluationRecord)
- [ ] Set up configuration files

### Phase 2: Rust Backend — Corpus Ingestion & Indexing
- [ ] Document ingestion service (text, PDF, OCR, transcripts)
- [ ] Deterministic chunking (fixed token size, overlapping windows)
- [ ] Metadata extraction and normalization
- [ ] Qdrant vector DB integration for embeddings storage
- [ ] PostgreSQL metadata registry via sqlx
- [ ] Audit logging with tracing

### Phase 3: Rust Backend — Retrieval Engine
- [ ] Vector similarity search via Qdrant
- [ ] Metadata filtering (tags, doc_type, tenant_id)
- [ ] Top-k retrieval with similarity scores
- [ ] Retrieval logging (chunk IDs, scores, latency)
- [ ] Axum HTTP API endpoints

### Phase 4: Python Layer — Guardrail Processor
- [ ] Context budgeting (max tokens, per-document quota)
- [ ] Injection detection (system overrides, instruction hijacks)
- [ ] PII redaction
- [ ] Prompt construction with strict template
- [ ] Prompt hashing and storage

### Phase 5: Python Layer — LLM Inference & Council
- [ ] Primary model inference (HuggingFace transformers, 4-bit quantization)
- [ ] Generator evaluator
- [ ] Grounding Inspector evaluator
- [ ] Safety Auditor evaluator
- [ ] Critic evaluator
- [ ] Structured JSON evaluation output

### Phase 6: Decision Aggregation & Failure Classification
- [ ] Aggregation policy engine (accept/abstain/escalate)
- [ ] Failure classifier (hallucination, policy violation, leakage, injection, instability)
- [ ] Risk scoring

### Phase 7: Metrics Engine
- [ ] Core metrics (refusal, hallucination, policy violation, leakage rates)
- [ ] RAG metrics (precision@k, recall@k, citation precision, supported claim ratio)
- [ ] Council metrics (disagreement, safety override frequency, self-correction rate)
- [ ] System metrics (p50/p95 latency, context length vs failure rate)

### Phase 8: Reporting & Dataset Export
- [ ] JSONL dataset export
- [ ] Visualization generation (matplotlib)
- [ ] Evaluation report generation
- [ ] ML failure risk predictor (logistic regression, random forest, gradient boosting)

### Phase 9: Integration & Testing
- [ ] End-to-end pipeline integration
- [ ] Synthetic test data generation
- [ ] Experiment runner with reproducibility
- [ ] Configuration validation
