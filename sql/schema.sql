-- RLRGF Database Schema for PostgreSQL
-- Audit Logging & Metadata Registry
-- ─── Documents ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS documents (
    doc_id UUID PRIMARY KEY,
    doc_type VARCHAR(50) NOT NULL,
    source TEXT NOT NULL,
    title TEXT,
    tenant_id VARCHAR(100),
    tags JSONB DEFAULT '[]'::jsonb,
    raw_size_bytes BIGINT DEFAULT 0,
    checksum VARCHAR(64) NOT NULL,
    ingest_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_documents_tenant ON documents(tenant_id);
CREATE INDEX idx_documents_doc_type ON documents(doc_type);
CREATE INDEX idx_documents_tags ON documents USING GIN(tags);
-- ─── Chunks ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id UUID PRIMARY KEY,
    doc_id UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    token_count INTEGER NOT NULL DEFAULT 0,
    source_ref JSONB,
    pii_flags JSONB DEFAULT '[]'::jsonb,
    injection_flags JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_chunks_doc_id ON chunks(doc_id);
CREATE INDEX idx_chunks_index ON chunks(doc_id, chunk_index);
-- ─── Retrieval Logs ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS retrieval_logs (
    query_id UUID PRIMARY KEY,
    query_text TEXT NOT NULL,
    retrieved_chunk_ids UUID [] NOT NULL DEFAULT '{}',
    similarity_scores REAL [] NOT NULL DEFAULT '{}',
    filters_applied JSONB DEFAULT '[]'::jsonb,
    top_k INTEGER NOT NULL DEFAULT 5,
    retrieval_latency_ms REAL NOT NULL DEFAULT 0,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_retrieval_timestamp ON retrieval_logs(timestamp);
-- ─── Model Outputs ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS model_outputs (
    output_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_id UUID NOT NULL,
    prompt_hash VARCHAR(64) NOT NULL,
    model_name VARCHAR(200) NOT NULL,
    generated_answer TEXT NOT NULL,
    citations JSONB DEFAULT '[]'::jsonb,
    generation_latency_ms REAL NOT NULL DEFAULT 0,
    token_usage JSONB NOT NULL DEFAULT '{}'::jsonb,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_model_outputs_query ON model_outputs(query_id);
CREATE INDEX idx_model_outputs_model ON model_outputs(model_name);
-- ─── Council Verdicts ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS council_verdicts (
    verdict_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_id UUID NOT NULL,
    evaluator_role VARCHAR(50) NOT NULL,
    critic_score REAL DEFAULT 0,
    grounding_score REAL DEFAULT 0,
    safety_flag BOOLEAN DEFAULT FALSE,
    leakage_risk REAL DEFAULT 0,
    confidence_score REAL DEFAULT 0,
    reasoning TEXT,
    claims JSONB DEFAULT '[]'::jsonb,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_verdicts_query ON council_verdicts(query_id);
CREATE INDEX idx_verdicts_role ON council_verdicts(evaluator_role);
-- ─── Evaluation Records ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS evaluation_records (
    eval_id UUID PRIMARY KEY,
    query_id UUID NOT NULL,
    experiment_id VARCHAR(100),
    failure_type VARCHAR(50),
    risk_score REAL NOT NULL DEFAULT 0,
    decision VARCHAR(50) NOT NULL,
    supported_claim_ratio REAL DEFAULT 0,
    policy_violation BOOLEAN DEFAULT FALSE,
    hallucination_flag BOOLEAN DEFAULT FALSE,
    leakage_detected BOOLEAN DEFAULT FALSE,
    injection_success BOOLEAN DEFAULT FALSE,
    instability_detected BOOLEAN DEFAULT FALSE,
    retrieval_precision_at_k REAL DEFAULT 0,
    retrieval_recall_at_k REAL DEFAULT 0,
    citation_precision REAL DEFAULT 0,
    generation_latency_ms REAL DEFAULT 0,
    retrieval_latency_ms REAL DEFAULT 0,
    context_token_count INTEGER DEFAULT 0,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_eval_query ON evaluation_records(query_id);
CREATE INDEX idx_eval_experiment ON evaluation_records(experiment_id);
CREATE INDEX idx_eval_failure ON evaluation_records(failure_type);
CREATE INDEX idx_eval_decision ON evaluation_records(decision);
-- ─── Experiment Metrics ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS experiment_metrics (
    metric_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id VARCHAR(100) NOT NULL UNIQUE,
    total_queries INTEGER NOT NULL DEFAULT 0,
    refusal_rate REAL DEFAULT 0,
    hallucination_rate REAL DEFAULT 0,
    policy_violation_rate REAL DEFAULT 0,
    leakage_rate REAL DEFAULT 0,
    injection_success_rate REAL DEFAULT 0,
    instability_rate REAL DEFAULT 0,
    avg_retrieval_precision REAL DEFAULT 0,
    avg_retrieval_recall REAL DEFAULT 0,
    avg_citation_precision REAL DEFAULT 0,
    avg_supported_ratio REAL DEFAULT 0,
    council_disagreement REAL DEFAULT 0,
    safety_override_freq REAL DEFAULT 0,
    self_correction_rate REAL DEFAULT 0,
    p50_latency_ms REAL DEFAULT 0,
    p95_latency_ms REAL DEFAULT 0,
    avg_context_length REAL DEFAULT 0,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
-- ─── Audit Trail ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS audit_trail (
    audit_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type VARCHAR(100) NOT NULL,
    component VARCHAR(100) NOT NULL,
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_audit_event ON audit_trail(event_type);
CREATE INDEX idx_audit_timestamp ON audit_trail(timestamp);
CREATE INDEX idx_audit_component ON audit_trail(component);