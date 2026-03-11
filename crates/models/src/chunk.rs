use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Represents a chunk of a document after deterministic chunking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub chunk_id: Uuid,
    pub doc_id: Uuid,
    pub chunk_index: usize,
    pub chunk_text: String,
    pub embedding: Option<Vec<f32>>,
    pub token_count: usize,
    pub source_reference: Option<SourceReference>,
    pub pii_flags: Vec<String>,
    pub injection_flags: Vec<String>,
    pub created_at: DateTime<Utc>,
}

/// Reference to the source location within the original document.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SourceReference {
    Page {
        page_number: usize,
    },
    Timestamp {
        start_seconds: f64,
        end_seconds: f64,
    },
    Frame {
        frame_index: usize,
    },
    LineRange {
        start_line: usize,
        end_line: usize,
    },
}

/// Configuration for the deterministic chunking algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkingConfig {
    /// Target number of tokens per chunk
    pub chunk_size_tokens: usize,
    /// Number of overlapping tokens between consecutive chunks
    pub overlap_tokens: usize,
    /// Minimum chunk size (chunks smaller than this are merged)
    pub min_chunk_tokens: usize,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            chunk_size_tokens: 256,
            overlap_tokens: 64,
            min_chunk_tokens: 32,
        }
    }
}
