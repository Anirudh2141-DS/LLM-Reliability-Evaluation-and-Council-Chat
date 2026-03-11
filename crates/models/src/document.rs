use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Represents a document ingested into the corpus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub doc_id: Uuid,
    pub doc_type: DocType,
    pub source: String,
    pub ingest_timestamp: DateTime<Utc>,
    pub tags: Vec<String>,
    pub tenant_id: Option<String>,
    pub title: Option<String>,
    pub raw_size_bytes: u64,
    pub checksum: String,
}

/// Supported document types for ingestion.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DocType {
    PlainText,
    Pdf,
    OcrOutput,
    Transcript,
    Html,
    Markdown,
}

impl std::fmt::Display for DocType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DocType::PlainText => write!(f, "plain_text"),
            DocType::Pdf => write!(f, "pdf"),
            DocType::OcrOutput => write!(f, "ocr_output"),
            DocType::Transcript => write!(f, "transcript"),
            DocType::Html => write!(f, "html"),
            DocType::Markdown => write!(f, "markdown"),
        }
    }
}

/// Request to ingest a new document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestRequest {
    pub source: String,
    pub doc_type: DocType,
    pub tags: Vec<String>,
    pub tenant_id: Option<String>,
    pub title: Option<String>,
    pub content: String,
}

/// Response after successful ingestion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestResponse {
    pub doc_id: Uuid,
    pub chunk_count: usize,
    pub total_tokens: usize,
    pub ingest_timestamp: DateTime<Utc>,
}
