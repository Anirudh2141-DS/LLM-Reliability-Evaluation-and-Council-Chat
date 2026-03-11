"""
Guardrail Layer - Context processing, injection detection, and PII redaction.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .models import RetrievedChunkRef


@dataclass
class GuardrailConfig:
    """Configuration for the guardrail system."""
    max_context_tokens: int = 2048
    per_document_token_quota: int = 512
    enable_pii_redaction: bool = True
    enable_injection_detection: bool = True


@dataclass
class GuardrailResult:
    """Result of the guardrail processing step."""
    prompt: str
    processed_chunks: List[RetrievedChunkRef]
    injection_flags: List[str] = field(default_factory=list)
    prompt_hash: str = ""


class GuardrailProcessor:
    """
    Orchestrates pre-inference safety and budgeting checks.
    """

    # Common injection patterns
    INJECTION_PATTERNS = [
        re.compile(r"(?i)ignore\s+previous\s+instructions"),
        re.compile(r"(?i)you\s+are\s+now\s+an?\s+unrestricted"),
        re.compile(r"(?i)reveal\s+your\s+system\s+prompt"),
        re.compile(r"(?i)override\s+safety"),
        re.compile(r"(?i)forget\s+your\s+rules"),
        re.compile(r"(?i)disregard\s+all\s+guidelines"),
        re.compile(r"(?i)\[system\]"),
        re.compile(r"(?i)<<sys>>"),
        re.compile(r"(?i)thought\s+process:"),
    ]

    # Simple PII patterns
    PII_PATTERNS = {
        "EMAIL": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
        "PHONE": re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
        "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "API_KEY": re.compile(r"sk-[a-zA-Z0-9]{20,}"),
    }

    def __init__(self, config: Optional[GuardrailConfig] = None):
        self.config = config or GuardrailConfig()

    def process(
        self,
        query: str,
        chunks: List[RetrievedChunkRef],
    ) -> GuardrailResult:
        """Process query and context through guardrails."""
        # 1. Detect injections in the query
        injection_flags = []
        if self.config.enable_injection_detection:
            for pattern in self.INJECTION_PATTERNS:
                if pattern.search(query):
                    injection_flags.append(f"Query: {pattern.pattern}")

        # 2. redacting PII from the query if needed
        processed_query = query
        if self.config.enable_pii_redaction:
            processed_query = self._redact_pii(processed_query)

        # 3. Process and budget the context
        processed_chunks, budgeted_text = self._budget_context(chunks)

        # 4. Detect injections in context chunks
        if self.config.enable_injection_detection:
            for i, chunk in enumerate(processed_chunks):
                for pattern in self.INJECTION_PATTERNS:
                    if pattern.search(chunk.chunk_text):
                        injection_flags.append(f"Chunk[{i}]: {pattern.pattern}")

        # 5. Build final prompt
        prompt = self._build_prompt(processed_query, budgeted_text)
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()

        return GuardrailResult(
            prompt=prompt,
            processed_chunks=processed_chunks,
            injection_flags=injection_flags,
            prompt_hash=prompt_hash,
        )

    def _budget_context(
        self, chunks: List[RetrievedChunkRef]
    ) -> Tuple[List[RetrievedChunkRef], str]:
        """Apply token quotas and limits to retrieved context."""
        budgeted_chunks = []
        texts = []
        total_chars = 0
        char_limit = self.config.max_context_tokens * 4  # Rough char estimate

        for chunk in sorted(chunks, key=lambda x: x.similarity_score, reverse=True):
            text = chunk.chunk_text
            
            # Application of per-document quota (rough estimation in chars)
            quota_chars = self.config.per_document_token_quota * 4
            if len(text) > quota_chars:
                text = text[:quota_chars] + "..."
            
            if total_chars + len(text) > char_limit:
                remaining = char_limit - total_chars
                if remaining > 50:
                    text = text[:remaining] + "..."
                    texts.append(text)
                    budgeted_chunks.append(chunk)
                break
            
            # redact PII from chunks
            if self.config.enable_pii_redaction:
                text = self._redact_pii(text)
            
            texts.append(text)
            budgeted_chunks.append(chunk)
            total_chars += len(text)

        return budgeted_chunks, "\n\n".join(texts)

    def _redact_pii(self, text: str) -> str:
        """Redact PII patterns from text."""
        processed = text
        for label, pattern in self.PII_PATTERNS.items():
            processed = pattern.sub(f"[REDACTED_{label}]", processed)
        return processed

    def _build_prompt(self, query: str, context: str) -> str:
        """Assemble the RAG prompt."""
        return (
            "You are a helpful assistant. Use the following retrieved context to answer the user query. "
            "If the answer is not in the context, say you don't know based on the provided documents. "
            "Do not use external knowledge.\n\n"
            "CONTEXT:\n"
            f"{context}\n\n"
            "QUERY:\n"
            f"{query}\n\n"
            "ANSWER:"
        )
