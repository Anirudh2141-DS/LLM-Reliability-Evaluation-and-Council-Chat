use sha2::{Digest, Sha256};

/// Compute a deterministic SHA-256 hash of the given content.
pub fn content_hash(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    hex::encode(hasher.finalize())
}

/// Compute a prompt hash for deduplication and caching.
pub fn prompt_hash(system_prompt: &str, user_query: &str, context: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"system:");
    hasher.update(system_prompt.as_bytes());
    hasher.update(b"|query:");
    hasher.update(user_query.as_bytes());
    hasher.update(b"|context:");
    hasher.update(context.as_bytes());
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_hash() {
        let h1 = content_hash("hello world");
        let h2 = content_hash("hello world");
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 64); // SHA-256 hex = 64 chars
    }

    #[test]
    fn test_different_content() {
        let h1 = content_hash("hello");
        let h2 = content_hash("world");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_prompt_hash_deterministic() {
        let h1 = prompt_hash("sys", "query", "ctx");
        let h2 = prompt_hash("sys", "query", "ctx");
        assert_eq!(h1, h2);
    }
}
