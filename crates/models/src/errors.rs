use thiserror::Error;

/// Domain-level errors for the RLRGF system.
#[derive(Error, Debug)]
pub enum RlrgfError {
    #[error("Document not found: {doc_id}")]
    DocumentNotFound { doc_id: String },

    #[error("Chunk not found: {chunk_id}")]
    ChunkNotFound { chunk_id: String },

    #[error("Ingestion failed: {reason}")]
    IngestionFailed { reason: String },

    #[error("Retrieval failed: {reason}")]
    RetrievalFailed { reason: String },

    #[error("Embedding generation failed: {reason}")]
    EmbeddingFailed { reason: String },

    #[error("Database error: {0}")]
    DatabaseError(String),

    #[error("Vector DB error: {0}")]
    VectorDbError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Context budget exceeded: {used} tokens > {limit} tokens")]
    ContextBudgetExceeded { used: usize, limit: usize },

    #[error("Injection detected: {pattern}")]
    InjectionDetected { pattern: String },

    #[error("PII detected: {category}")]
    PiiDetected { category: String },

    #[error("Council evaluation failed: {reason}")]
    CouncilError { reason: String },

    #[error("Audit logging failed: {reason}")]
    AuditError { reason: String },
}
