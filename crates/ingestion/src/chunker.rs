use chrono::Utc;
use tracing::info;
use uuid::Uuid;

use rlrgf_models::{Chunk, ChunkingConfig, SourceReference};

/// Deterministic chunker that splits text into fixed-size token windows
/// with configurable overlap.
pub struct DeterministicChunker {
    config: ChunkingConfig,
}

impl DeterministicChunker {
    pub fn new(config: ChunkingConfig) -> Self {
        Self { config }
    }

    /// Chunk the given text into deterministic, overlapping windows.
    ///
    /// Uses whitespace tokenization as an approximation for token counting.
    /// For production use, integrate tiktoken or a model-specific tokenizer.
    pub fn chunk(&self, doc_id: Uuid, text: &str) -> Vec<Chunk> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let total_words = words.len();

        if total_words == 0 {
            return vec![];
        }

        let chunk_size = self.config.chunk_size_tokens;
        let overlap = self.config.overlap_tokens;
        let step = if chunk_size > overlap {
            chunk_size - overlap
        } else {
            1
        };

        let mut chunks = Vec::new();
        let mut start = 0;
        let mut chunk_index = 0;

        while start < total_words {
            let end = (start + chunk_size).min(total_words);
            let chunk_words = &words[start..end];
            let chunk_text = chunk_words.join(" ");
            let token_count = chunk_words.len();

            // Skip very small trailing chunks unless it's the only chunk
            if token_count < self.config.min_chunk_tokens && chunk_index > 0 {
                // Merge with previous chunk
                if let Some(last) = chunks.last_mut() {
                    let last_chunk: &mut Chunk = last;
                    last_chunk.chunk_text.push(' ');
                    last_chunk.chunk_text.push_str(&chunk_text);
                    last_chunk.token_count += token_count;
                }
                break;
            }

            let chunk = Chunk {
                chunk_id: Uuid::new_v4(),
                doc_id,
                chunk_index,
                chunk_text,
                embedding: None,
                token_count,
                source_reference: Some(SourceReference::LineRange {
                    start_line: start,
                    end_line: end,
                }),
                pii_flags: vec![],
                injection_flags: vec![],
                created_at: Utc::now(),
            };

            chunks.push(chunk);
            chunk_index += 1;
            start += step;

            if end == total_words {
                break;
            }
        }

        info!(
            doc_id = %doc_id,
            total_words = total_words,
            chunk_count = chunks.len(),
            "Chunking complete"
        );

        chunks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_chunking() {
        let config = ChunkingConfig {
            chunk_size_tokens: 10,
            overlap_tokens: 3,
            min_chunk_tokens: 2,
        };
        let chunker = DeterministicChunker::new(config);
        let doc_id = Uuid::new_v4();
        let text = "word ".repeat(25).trim().to_string();
        let chunks = chunker.chunk(doc_id, &text);

        assert!(!chunks.is_empty());
        // Every chunk should reference this document
        for chunk in &chunks {
            assert_eq!(chunk.doc_id, doc_id);
        }
    }

    #[test]
    fn test_empty_text() {
        let chunker = DeterministicChunker::new(ChunkingConfig::default());
        let chunks = chunker.chunk(Uuid::new_v4(), "");
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_deterministic_output() {
        let config = ChunkingConfig {
            chunk_size_tokens: 5,
            overlap_tokens: 2,
            min_chunk_tokens: 1,
        };
        let chunker = DeterministicChunker::new(config);
        let doc_id = Uuid::new_v4();
        let text = "alpha bravo charlie delta echo foxtrot golf hotel india juliet";

        let chunks1 = chunker.chunk(doc_id, text);
        let chunks2 = chunker.chunk(doc_id, text);

        assert_eq!(chunks1.len(), chunks2.len());
        for (a, b) in chunks1.iter().zip(chunks2.iter()) {
            assert_eq!(a.chunk_text, b.chunk_text);
            assert_eq!(a.chunk_index, b.chunk_index);
            assert_eq!(a.token_count, b.token_count);
        }
    }
}
