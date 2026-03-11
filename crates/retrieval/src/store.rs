use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use rlrgf_models::{Chunk, Document};
use uuid::Uuid;

/// In-memory store for documents and chunks.
/// For production, this would be backed by PostgreSQL + Qdrant.
#[derive(Clone)]
pub struct DocumentStore {
    documents: Arc<RwLock<HashMap<Uuid, Document>>>,
    chunks: Arc<RwLock<HashMap<Uuid, Chunk>>>,
    doc_chunks: Arc<RwLock<HashMap<Uuid, Vec<Uuid>>>>,
}

impl DocumentStore {
    pub fn new() -> Self {
        Self {
            documents: Arc::new(RwLock::new(HashMap::new())),
            chunks: Arc::new(RwLock::new(HashMap::new())),
            doc_chunks: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Store a document and its chunks.
    pub fn insert_document(&self, doc: Document, chunks: Vec<Chunk>) {
        let doc_id = doc.doc_id;
        let chunk_ids: Vec<Uuid> = chunks.iter().map(|c| c.chunk_id).collect();

        self.documents.write().unwrap().insert(doc_id, doc);

        let mut chunk_store = self.chunks.write().unwrap();
        for chunk in chunks {
            chunk_store.insert(chunk.chunk_id, chunk);
        }

        self.doc_chunks.write().unwrap().insert(doc_id, chunk_ids);
    }

    /// Get a document by ID.
    pub fn get_document(&self, doc_id: &Uuid) -> Option<Document> {
        self.documents.read().unwrap().get(doc_id).cloned()
    }

    /// Get all chunks for a document.
    pub fn get_chunks_for_doc(&self, doc_id: &Uuid) -> Vec<Chunk> {
        let doc_chunks = self.doc_chunks.read().unwrap();
        let chunk_store = self.chunks.read().unwrap();

        doc_chunks
            .get(doc_id)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| chunk_store.get(id).cloned())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get all chunks across all documents.
    pub fn get_all_chunks(&self) -> Vec<Chunk> {
        self.chunks.read().unwrap().values().cloned().collect()
    }

    /// Get all documents.
    pub fn get_all_documents(&self) -> Vec<Document> {
        self.documents.read().unwrap().values().cloned().collect()
    }

    /// Count documents.
    pub fn document_count(&self) -> usize {
        self.documents.read().unwrap().len()
    }

    /// Count chunks.
    pub fn chunk_count(&self) -> usize {
        self.chunks.read().unwrap().len()
    }
}

impl Default for DocumentStore {
    fn default() -> Self {
        Self::new()
    }
}
