use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use serde::Serialize;
use uuid::Uuid;

use rlrgf_models::{IngestRequest, IngestResponse, RetrievalRequest, RetrievalResponse};

use crate::state::AppState;

/// Health check endpoint.
pub async fn health_check() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".into(),
        service: "rlrgf-api-server".into(),
        version: env!("CARGO_PKG_VERSION").into(),
    })
}

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub service: String,
    pub version: String,
}

/// Ingest a document.
pub async fn ingest_document(
    State(state): State<AppState>,
    Json(request): Json<IngestRequest>,
) -> Result<Json<IngestResponse>, (StatusCode, String)> {
    let (doc, chunks, response) = state
        .ingestion
        .ingest(request)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // Store the document and chunks
    state.store.insert_document(doc.clone(), chunks);

    // Audit log
    let _ = state
        .audit
        .log_ingestion(&doc.doc_id.to_string(), response.chunk_count, &doc.source);

    Ok(Json(response))
}

/// Retrieve relevant chunks for a query.
pub async fn retrieve(
    State(state): State<AppState>,
    Json(request): Json<RetrievalRequest>,
) -> Json<RetrievalResponse> {
    let response = state.retrieval.retrieve(&request);

    // Audit log
    let _ = state.audit.log_retrieval(
        &response.query_id.to_string(),
        response.chunks.len(),
        response.latency_ms,
    );

    Json(response)
}

/// List all ingested documents.
pub async fn list_documents(State(state): State<AppState>) -> Json<Vec<rlrgf_models::Document>> {
    let docs = state.store.get_all_documents();
    Json(docs)
}

/// Get chunks for a specific document.
pub async fn get_document_chunks(
    State(state): State<AppState>,
    Path(doc_id): Path<Uuid>,
) -> Result<Json<Vec<rlrgf_models::Chunk>>, (StatusCode, String)> {
    let chunks = state.store.get_chunks_for_doc(&doc_id);
    if chunks.is_empty() {
        // Check if document exists
        if state.store.get_document(&doc_id).is_none() {
            return Err((
                StatusCode::NOT_FOUND,
                format!("Document {} not found", doc_id),
            ));
        }
    }
    Ok(Json(chunks))
}

/// Get system statistics.
pub async fn system_stats(State(state): State<AppState>) -> Json<SystemStats> {
    Json(SystemStats {
        document_count: state.store.document_count(),
        chunk_count: state.store.chunk_count(),
    })
}

#[derive(Serialize)]
pub struct SystemStats {
    pub document_count: usize,
    pub chunk_count: usize,
}
