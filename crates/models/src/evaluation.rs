use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::council::Decision;

/// Final evaluation record for a single query through the entire pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationRecord {
    pub eval_id: Uuid,
    pub query_id: Uuid,
    pub failure_type: Option<FailureType>,
    pub risk_score: f32,
    pub decision: Decision,
    pub supported_claim_ratio: f32,
    pub policy_violation: bool,
    pub hallucination_flag: bool,
    pub leakage_detected: bool,
    pub injection_success: bool,
    pub instability_detected: bool,
    pub retrieval_precision_at_k: f32,
    pub retrieval_recall_at_k: f32,
    pub citation_precision: f32,
    pub generation_latency_ms: f64,
    pub retrieval_latency_ms: f64,
    pub context_token_count: usize,
    pub timestamp: DateTime<Utc>,
    pub experiment_id: Option<String>,
}

/// Classification of failure types detected during evaluation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FailureType {
    Hallucination,
    PolicyViolation,
    Leakage,
    InjectionSuccess,
    Instability,
    MultipleFailures,
}

impl std::fmt::Display for FailureType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FailureType::Hallucination => write!(f, "hallucination"),
            FailureType::PolicyViolation => write!(f, "policy_violation"),
            FailureType::Leakage => write!(f, "leakage"),
            FailureType::InjectionSuccess => write!(f, "injection_success"),
            FailureType::Instability => write!(f, "instability"),
            FailureType::MultipleFailures => write!(f, "multiple_failures"),
        }
    }
}

/// Aggregate metrics across an entire experiment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentMetrics {
    pub experiment_id: String,
    pub total_queries: usize,
    pub refusal_rate: f32,
    pub hallucination_rate: f32,
    pub policy_violation_rate: f32,
    pub leakage_rate: f32,
    pub injection_success_rate: f32,
    pub instability_rate: f32,
    pub avg_retrieval_precision_at_k: f32,
    pub avg_retrieval_recall_at_k: f32,
    pub avg_citation_precision: f32,
    pub avg_supported_claim_ratio: f32,
    pub council_disagreement_avg: f32,
    pub safety_override_frequency: f32,
    pub self_correction_rate: f32,
    pub p50_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub avg_context_length_tokens: f32,
    pub timestamp: DateTime<Utc>,
}
