use anyhow::Result;
use chrono::Utc;
use tracing::info;
use uuid::Uuid;

use rlrgf_models::{ChunkingConfig, Document, IngestRequest, IngestResponse};

use crate::chunker::DeterministicChunker;
use crate::hasher::content_hash;
use crate::normalizer::TextNormalizer;

/// The ingestion service coordinates normalization, chunking, and metadata creation.
pub struct IngestionService {
    normalizer: TextNormalizer,
    chunker: DeterministicChunker,
}

impl IngestionService {
    pub fn new(chunking_config: ChunkingConfig) -> Self {
        Self {
            normalizer: TextNormalizer::new(),
            chunker: DeterministicChunker::new(chunking_config),
        }
    }

    /// Ingest a document: normalize, chunk, and return the document + chunks.
    pub fn ingest(&self, request: IngestRequest) -> Result<(Document, Vec<Chunk>, IngestResponse)> {
        let doc_id = Uuid::new_v4();
        let now = Utc::now();

        // Step 1: Normalize the raw content
        let normalized = self.normalizer.normalize(&request.content);
        let checksum = content_hash(&normalized);
        let raw_size = request.content.len() as u64;

        info!(
            doc_id = %doc_id,
            source = %request.source,
            doc_type = %request.doc_type,
            raw_size = raw_size,
            "Starting ingestion"
        );

        // Step 2: Create document record
        let document = Document {
            doc_id,
            doc_type: request.doc_type,
            source: request.source,
            ingest_timestamp: now,
            tags: request.tags,
            tenant_id: request.tenant_id,
            title: request.title,
            raw_size_bytes: raw_size,
            checksum,
        };

        // Step 3: Chunk the normalized text
        let chunks = self.chunker.chunk(doc_id, &normalized);
        let total_tokens: usize = chunks.iter().map(|c| c.token_count).sum();

        let response = IngestResponse {
            doc_id,
            chunk_count: chunks.len(),
            total_tokens,
            ingest_timestamp: now,
        };

        info!(
            doc_id = %doc_id,
            chunk_count = chunks.len(),
            total_tokens = total_tokens,
            "Ingestion complete"
        );

        Ok((document, chunks, response))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlrgf_models::DocType;

    #[test]
    fn test_ingest_service() {
        let service = IngestionService::new(ChunkingConfig::default());
        let req = IngestRequest {
            source: "test_file.txt".into(),
            doc_type: DocType::PlainText,
            tags: vec!["test".into()],
            tenant_id: Some("tenant1".into()),
            title: Some("Test Document".into()),
            content: "This is a test document with enough words to create at least one chunk in the deterministic chunker. We need to pad it with sufficient content so the chunker actually produces meaningful output for our test assertions to verify.".into(),
        };

        let (doc, chunks, resp) = service.ingest(req).unwrap();
        assert!(!chunks.is_empty());
        assert_eq!(doc.doc_id, resp.doc_id);
        assert_eq!(chunks.len(), resp.chunk_count);
    }
}
