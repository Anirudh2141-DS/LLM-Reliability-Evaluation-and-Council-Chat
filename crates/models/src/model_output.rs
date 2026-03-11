use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Output from a single LLM generation step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelOutput {
    pub query_id: Uuid,
    pub prompt_hash: String,
    pub model_name: String,
    pub generated_answer: String,
    pub citations: Vec<Citation>,
    pub generation_latency_ms: f64,
    pub token_usage: TokenUsage,
    pub timestamp: DateTime<Utc>,
}

/// Citation referencing a specific chunk used in the answer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
    pub chunk_id: Uuid,
    pub chunk_index: usize,
    pub excerpt: String,
    pub relevance_score: f32,
}

/// Token usage statistics from a generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}
