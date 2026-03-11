use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Log entry for a single retrieval operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalLog {
    pub query_id: Uuid,
    pub query_text: String,
    pub retrieved_chunks: Vec<RetrievedChunkRef>,
    pub filters_applied: Vec<RetrievalFilter>,
    pub retrieval_latency_ms: f64,
    pub timestamp: DateTime<Utc>,
    pub top_k: usize,
}

/// Reference to a retrieved chunk with similarity score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievedChunkRef {
    pub chunk_id: Uuid,
    pub doc_id: Uuid,
    pub similarity_score: f32,
    pub chunk_text: String,
    pub chunk_index: usize,
}

/// Filters that can be applied during retrieval.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "filter_type", rename_all = "snake_case")]
pub enum RetrievalFilter {
    Tag {
        tag: String,
    },
    DocType {
        doc_type: String,
    },
    TenantId {
        tenant_id: String,
    },
    DateRange {
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    },
}

/// Query request sent to the retrieval engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalRequest {
    pub query_text: String,
    pub top_k: usize,
    pub filters: Vec<RetrievalFilter>,
    pub tenant_id: Option<String>,
}

/// Response from the retrieval engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalResponse {
    pub query_id: Uuid,
    pub chunks: Vec<RetrievedChunkRef>,
    pub latency_ms: f64,
}
