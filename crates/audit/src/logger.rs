use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::Result;
use chrono::Utc;
use serde::Serialize;
use tracing::{error, info};

#[cfg(feature = "postgres")]
use sqlx::{postgres::PgRow, PgPool};

/// Append-only audit logger for the RLRGF system.
/// Writes structured JSON lines to an audit log file and optionally to PostgreSQL.
pub struct AuditLogger {
    log_dir: PathBuf,
    file_handle: Mutex<Option<std::fs::File>>,
    #[cfg(feature = "postgres")]
    db_pool: Option<PgPool>,
}

/// Represents a single audit log entry.
#[derive(Debug, Clone, Serialize)]
pub struct AuditEntry {
    pub timestamp: String,
    pub event_type: String,
    pub component: String,
    pub details: serde_json::Value,
}

impl AuditLogger {
    pub fn new(log_dir: impl AsRef<Path>) -> Result<Self> {
        let log_dir = log_dir.as_ref().to_path_buf();
        fs::create_dir_all(&log_dir)?;

        let log_file = log_dir.join("audit.jsonl");
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_file)?;

        info!(path = %log_file.display(), "Audit logger initialized with file backup");

        Ok(Self {
            log_dir,
            file_handle: Mutex::new(Some(file)),
            #[cfg(feature = "postgres")]
            db_pool: None,
        })
    }

    #[cfg(feature = "postgres")]
    pub fn with_db(log_dir: impl AsRef<Path>, pool: PgPool) -> Result<Self> {
        let mut logger = Self::new(log_dir)?;
        logger.db_pool = Some(pool);
        info!("Audit logger initialized with PostgreSQL");
        Ok(logger)
    }

    /// Log an audit event.
    pub fn log_event(
        &self,
        event_type: &str,
        component: &str,
        details: serde_json::Value,
    ) -> Result<()> {
        let entry = AuditEntry {
            timestamp: Utc::now().to_rfc3339(),
            event_type: event_type.into(),
            component: component.into(),
            details: details.clone(),
        };

        // 1. Write to file (always as backup)
        let json = serde_json::to_string(&entry)?;
        if let Ok(mut handle) = self.file_handle.lock() {
            if let Some(ref mut file) = *handle {
                if let Err(e) = writeln!(file, "{}", json) {
                    error!("Failed to write to audit file: {}", e);
                }
                let _ = file.flush();
            }
        }

        // 2. Write to PostgreSQL if available
        #[cfg(feature = "postgres")]
        if let Some(ref pool) = self.db_pool {
            let pool = pool.clone();
            let event_type = event_type.to_string();
            let component = component.to_string();
            let details = details.clone();

            // Spawn background task to avoid blocking the caller
            tokio::spawn(async move {
                let res = sqlx::query(
                    "INSERT INTO audit_trail (event_type, component, details) VALUES ($1, $2, $3)",
                )
                .bind(event_type)
                .bind(component)
                .bind(details)
                .execute(&pool)
                .await;

                if let Err(e) = res {
                    error!("Failed to write to audit database: {}", e);
                }
            });
        }

        Ok(())
    }

    /// Log a document ingestion event.
    pub fn log_ingestion(&self, doc_id: &str, chunk_count: usize, source: &str) -> Result<()> {
        self.log_event(
            "document_ingested",
            "ingestion",
            serde_json::json!({
                "doc_id": doc_id,
                "chunk_count": chunk_count,
                "source": source
            }),
        )
    }

    /// Log a retrieval event.
    pub fn log_retrieval(
        &self,
        query_id: &str,
        result_count: usize,
        latency_ms: f64,
    ) -> Result<()> {
        self.log_event(
            "retrieval_executed",
            "retrieval",
            serde_json::json!({
                "query_id": query_id,
                "result_count": result_count,
                "latency_ms": latency_ms
            }),
        )
    }

    /// Log a council evaluation event.
    pub fn log_council_evaluation(
        &self,
        query_id: &str,
        decision: &str,
        risk_score: f32,
    ) -> Result<()> {
        self.log_event(
            "council_evaluation",
            "council",
            serde_json::json!({
                "query_id": query_id,
                "decision": decision,
                "risk_score": risk_score
            }),
        )
    }

    /// Log a failure classification event.
    pub fn log_failure(
        &self,
        query_id: &str,
        failure_type: &str,
        details: serde_json::Value,
    ) -> Result<()> {
        self.log_event(
            "failure_classified",
            "classifier",
            serde_json::json!({
                "query_id": query_id,
                "failure_type": failure_type,
                "details": details
            }),
        )
    }

    /// Get the audit log directory path.
    pub fn log_dir(&self) -> &Path {
        &self.log_dir
    }
}
