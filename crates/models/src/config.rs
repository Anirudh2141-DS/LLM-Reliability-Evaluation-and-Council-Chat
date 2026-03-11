use serde::{Deserialize, Serialize};

use crate::chunk::ChunkingConfig;

/// Top-level configuration for the RLRGF system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemConfig {
    pub server: ServerConfig,
    pub database: DatabaseConfig,
    pub vector_db: VectorDbConfig,
    pub chunking: ChunkingConfig,
    pub retrieval: RetrievalConfig,
    pub guardrails: GuardrailConfig,
    pub council: CouncilConfig,
    pub experiment: ExperimentConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub log_level: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    pub url: String,
    pub max_connections: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorDbConfig {
    pub url: String,
    pub collection_name: String,
    pub embedding_dimension: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalConfig {
    pub default_top_k: usize,
    pub min_similarity_threshold: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardrailConfig {
    pub max_context_tokens: usize,
    pub per_document_token_quota: usize,
    pub enable_pii_redaction: bool,
    pub enable_injection_detection: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouncilConfig {
    /// Threshold below which supported_claim_ratio triggers retrieval request
    pub grounding_threshold: f32,
    /// Threshold above which disagreement triggers clarification
    pub disagreement_threshold: f32,
    /// Require all safety checks to pass
    pub strict_safety: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentConfig {
    pub experiment_id: String,
    pub output_dir: String,
    pub export_format: ExportFormat,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExportFormat {
    Jsonl,
    Csv,
    Both,
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            server: ServerConfig {
                host: "0.0.0.0".into(),
                port: 8080,
                log_level: "info".into(),
            },
            database: DatabaseConfig {
                url: "postgresql://localhost:5432/rlrgf".into(),
                max_connections: 10,
            },
            vector_db: VectorDbConfig {
                url: "http://localhost:6334".into(),
                collection_name: "rlrgf_chunks".into(),
                embedding_dimension: 384,
            },
            chunking: ChunkingConfig::default(),
            retrieval: RetrievalConfig {
                default_top_k: 5,
                min_similarity_threshold: 0.3,
            },
            guardrails: GuardrailConfig {
                max_context_tokens: 2048,
                per_document_token_quota: 512,
                enable_pii_redaction: true,
                enable_injection_detection: true,
            },
            council: CouncilConfig {
                grounding_threshold: 0.6,
                disagreement_threshold: 0.4,
                strict_safety: true,
            },
            experiment: ExperimentConfig {
                experiment_id: "default".into(),
                output_dir: "./output".into(),
                export_format: ExportFormat::Jsonl,
            },
        }
    }
}
