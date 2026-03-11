use std::time::Instant;

use chrono::Utc;
use tracing::info;
use uuid::Uuid;

use rlrgf_models::{
    RetrievalFilter, RetrievalLog, RetrievalRequest, RetrievalResponse, RetrievedChunkRef,
};

use crate::store::DocumentStore;

/// Retrieval engine that performs similarity search over the document store.
/// In production this would query Qdrant; here we use keyword-based matching.
pub struct RetrievalEngine {
    store: DocumentStore,
    default_top_k: usize,
}

impl RetrievalEngine {
    pub fn new(store: DocumentStore, default_top_k: usize) -> Self {
        Self {
            store,
            default_top_k,
        }
    }

    /// Execute a retrieval query against the chunk store.
    /// Uses simple keyword overlap scoring as a stand-in for vector similarity.
    pub fn retrieve(&self, request: &RetrievalRequest) -> RetrievalResponse {
        let start = Instant::now();
        let query_id = Uuid::new_v4();

        let query_terms: Vec<String> = request
            .query_text
            .to_lowercase()
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();

        let all_chunks = self.store.get_all_chunks();
        let top_k = self.effective_top_k(request.top_k);

        // Score each chunk by keyword overlap
        let mut scored: Vec<(f32, &rlrgf_models::Chunk)> = all_chunks
            .iter()
            .map(|chunk| {
                let chunk_lower = chunk.chunk_text.to_lowercase();
                let chunk_words: Vec<&str> = chunk_lower.split_whitespace().collect();
                let total = chunk_words.len().max(1) as f32;
                let matches = query_terms
                    .iter()
                    .filter(|qt| chunk_words.iter().any(|cw| cw.contains(qt.as_str())))
                    .count() as f32;
                let score = matches / total.sqrt();
                (score, chunk)
            })
            .filter(|(score, _)| *score > 0.0)
            .collect();

        // Apply filters
        let all_docs = self.store.get_all_documents();
        for filter in &request.filters {
            match filter {
                RetrievalFilter::Tag { tag } => {
                    let matching_docs: Vec<Uuid> = all_docs
                        .iter()
                        .filter(|d| d.tags.contains(tag))
                        .map(|d| d.doc_id)
                        .collect();
                    scored.retain(|(_, c)| matching_docs.contains(&c.doc_id));
                }
                RetrievalFilter::DocType { doc_type } => {
                    let matching_docs: Vec<Uuid> = all_docs
                        .iter()
                        .filter(|d| d.doc_type.to_string() == *doc_type)
                        .map(|d| d.doc_id)
                        .collect();
                    scored.retain(|(_, c)| matching_docs.contains(&c.doc_id));
                }
                RetrievalFilter::TenantId { tenant_id } => {
                    let matching_docs: Vec<Uuid> = all_docs
                        .iter()
                        .filter(|d| d.tenant_id.as_deref() == Some(tenant_id))
                        .map(|d| d.doc_id)
                        .collect();
                    scored.retain(|(_, c)| matching_docs.contains(&c.doc_id));
                }
                _ => {} // DateRange filter not implemented for in-memory store
            }
        }

        // Sort by score descending
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        let chunks: Vec<RetrievedChunkRef> = scored
            .iter()
            .map(|(score, chunk)| RetrievedChunkRef {
                chunk_id: chunk.chunk_id,
                doc_id: chunk.doc_id,
                similarity_score: *score,
                chunk_text: chunk.chunk_text.clone(),
                chunk_index: chunk.chunk_index,
            })
            .collect();

        let latency = start.elapsed().as_secs_f64() * 1000.0;

        info!(
            query_id = %query_id,
            results = chunks.len(),
            latency_ms = latency,
            "Retrieval complete"
        );

        RetrievalResponse {
            query_id,
            chunks,
            latency_ms: latency,
        }
    }

    /// Create a retrieval log entry from a completed retrieval.
    pub fn create_log(
        &self,
        request: &RetrievalRequest,
        response: &RetrievalResponse,
    ) -> RetrievalLog {
        RetrievalLog {
            query_id: response.query_id,
            query_text: request.query_text.clone(),
            retrieved_chunks: response.chunks.clone(),
            filters_applied: request.filters.clone(),
            retrieval_latency_ms: response.latency_ms,
            timestamp: Utc::now(),
            top_k: self.effective_top_k(request.top_k),
        }
    }

    fn effective_top_k(&self, requested_top_k: usize) -> usize {
        if requested_top_k > 0 {
            requested_top_k
        } else {
            self.default_top_k
        }
    }
}

#[cfg(test)]
mod tests {
    use super::RetrievalEngine;
    use crate::store::DocumentStore;
    use rlrgf_models::{RetrievalRequest, RetrievalResponse};
    use uuid::Uuid;

    #[test]
    fn create_log_uses_default_top_k_when_request_is_zero() {
        let engine = RetrievalEngine::new(DocumentStore::new(), 7);
        let request = RetrievalRequest {
            query_text: "test".into(),
            top_k: 0,
            filters: vec![],
            tenant_id: None,
        };
        let response = RetrievalResponse {
            query_id: Uuid::new_v4(),
            chunks: vec![],
            latency_ms: 1.2,
        };

        let log = engine.create_log(&request, &response);
        assert_eq!(log.top_k, 7);
    }

    #[test]
    fn create_log_keeps_explicit_top_k() {
        let engine = RetrievalEngine::new(DocumentStore::new(), 7);
        let request = RetrievalRequest {
            query_text: "test".into(),
            top_k: 3,
            filters: vec![],
            tenant_id: None,
        };
        let response = RetrievalResponse {
            query_id: Uuid::new_v4(),
            chunks: vec![],
            latency_ms: 1.2,
        };

        let log = engine.create_log(&request, &response);
        assert_eq!(log.top_k, 3);
    }
}
