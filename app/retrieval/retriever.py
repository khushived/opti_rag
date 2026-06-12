"""
app/retrieval/retriever.py
───────────────────────────
Orchestrates query-time retrieval:
  1. Embed the query text (via Ollama)
  2. Query the vector store for top-k similar chunks
  3. Separate text context from image context
  4. Return a ``RetrievalResult`` ready for the generation stage
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.core.config import get_settings
from app.core.logging import get_logger
from app.ingestion.embedder import embed_text_async
from app.retrieval.vector_store import VectorStore

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    """Structured output from a retrieval pass."""

    query: str
    query_embedding: np.ndarray
    text_chunks: list[dict[str, Any]]    # hit dicts with "text", "metadata", "score"
    image_chunk_ids: list[str]           # chunk_ids whose metadata["has_images"] is True
    sources: list[str]                   # unique source files referenced


class Retriever:
    """
    Combines Ollama embedding with ChromaDB retrieval.

    The same ``VectorStore`` instance should be shared across the application
    (created once during startup and injected here).
    """

    def __init__(self, vector_store: VectorStore) -> None:
        self._store = vector_store

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> RetrievalResult:
        """
        Embed *query* then fetch top-k chunks from the vector store.

        Args:
            query:  Natural language question.
            top_k:  Override the configured ``top_k_retrieval`` value.

        Returns:
            A ``RetrievalResult`` with separated text / image contexts.
        """
        settings = get_settings()
        k = top_k or settings.top_k_retrieval

        logger.info("Retrieving context", query=query[:80], top_k=k)

        # 1. Embed query
        query_embedding = await embed_text_async(query)

        # 2. Query vector store (sync call – run in thread pool to avoid blocking)
        loop = asyncio.get_event_loop()
        hits = await loop.run_in_executor(
            None, lambda: self._store.query(query_embedding, top_k=k)
        )

        # 3. Partition hits
        text_chunks: list[dict[str, Any]] = []
        image_chunk_ids: list[str] = []
        sources: set[str] = set()

        for hit in hits:
            sources.add(hit["metadata"].get("source", "unknown"))
            if hit["metadata"].get("has_images", False):
                image_chunk_ids.append(hit["chunk_id"])
            text_chunks.append(hit)

        logger.info(
            "Retrieval complete",
            hits=len(hits),
            text_chunks=len(text_chunks),
            image_chunks=len(image_chunk_ids),
        )

        return RetrievalResult(
            query=query,
            query_embedding=query_embedding,
            text_chunks=text_chunks,
            image_chunk_ids=image_chunk_ids,
            sources=sorted(sources),
        )
