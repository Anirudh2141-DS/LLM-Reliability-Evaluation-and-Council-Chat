use anyhow::Result;
use std::sync::Arc;

use rlrgf_audit::AuditLogger;
use rlrgf_ingestion::IngestionService;
use rlrgf_models::{ChunkingConfig, SystemConfig};
use rlrgf_retrieval::{DocumentStore, RetrievalEngine};

#[cfg(feature = "postgres")]
use sqlx::postgres::PgPoolOptions;

/// Shared application state accessible by all handlers.
#[derive(Clone)]
pub struct AppState {
    pub ingestion: Arc<IngestionService>,
    pub store: Arc<DocumentStore>,
    pub retrieval: Arc<RetrievalEngine>,
    pub audit: Arc<AuditLogger>,
    pub config: Arc<SystemConfig>,
}

impl AppState {
    pub async fn new(config: SystemConfig) -> Result<Self> {
        let config = Arc::new(config);

        let ingestion = Arc::new(IngestionService::new(config.chunking.clone()));
        let store = Arc::new(DocumentStore::new());
        let retrieval = Arc::new(RetrievalEngine::new(
            store.as_ref().clone(),
            config.retrieval.default_top_k,
        ));

        // Initialize audit logger with DB if configured
        #[cfg(feature = "postgres")]
        {
            match PgPoolOptions::new()
                .max_connections(config.database.max_connections)
                .connect(&config.database.url)
                .await
            {
                Ok(pool) => {
                    let audit = Arc::new(AuditLogger::with_db("./output/audit", pool)?);
                    return Ok(Self {
                        ingestion,
                        store,
                        retrieval,
                        audit,
                        config,
                    });
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to connect to database: {}. Falling back to file-only audit.",
                        e
                    );
                }
            }
        }

        let audit = Arc::new(AuditLogger::new("./output/audit")?);

        Ok(Self {
            ingestion,
            store,
            retrieval,
            audit,
            config,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::AppState;
    use rlrgf_models::{DocType, IngestRequest, RetrievalRequest, SystemConfig};

    #[tokio::test]
    async fn ingest_then_retrieve_roundtrip_uses_shared_store() {
        let state = AppState::new(SystemConfig::default())
            .await
            .expect("state initialization should succeed");

        let ingest = IngestRequest {
            source: "roundtrip.txt".into(),
            doc_type: DocType::PlainText,
            tags: vec!["roundtrip".into()],
            tenant_id: None,
            title: Some("Roundtrip".into()),
            content: "shared_store_roundtrip_token appears in retrieval results".into(),
        };

        let (doc, chunks, _) = state
            .ingestion
            .ingest(ingest)
            .expect("ingestion should succeed");
        state.store.insert_document(doc, chunks);

        let response = state.retrieval.retrieve(&RetrievalRequest {
            query_text: "shared_store_roundtrip_token".into(),
            top_k: 0,
            filters: vec![],
            tenant_id: None,
        });

        assert!(
            !response.chunks.is_empty(),
            "retrieval should see chunks inserted by ingestion"
        );
    }
}
