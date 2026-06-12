"""
app/retrieval/vector_store.py
──────────────────────────────
ChromaDB wrapper for storing and querying document chunk embeddings.

Uses a persistent client so the index survives process restarts.
All operations are synchronous (ChromaDB's Python API is sync).
"""

from __future__ import annotations

from typing import Any, Optional

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings

from app.core.config import get_settings
from app.core.logging import get_logger
from app.ingestion.chunker import Chunk

logger = get_logger(__name__)


class VectorStore:
    """
    Thin wrapper around a ChromaDB collection.

    Handles:
    - Initialising a persistent ChromaDB client
    - Upserting chunks with pre-computed embeddings
    - Top-k nearest-neighbour queries
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._client = chromadb.PersistentClient(
            path=str(settings.chroma_persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},   # use cosine distance
        )
        logger.info(
            "VectorStore initialised",
            collection=settings.chroma_collection_name,
            persist_dir=str(settings.chroma_persist_dir),
            existing_docs=self._collection.count(),
        )

    # ─── Write ────────────────────────────────────────────────────────────────

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[np.ndarray],
    ) -> None:
        """
        Upsert chunks and their embeddings into ChromaDB.

        Args:
            chunks:     List of ``Chunk`` objects.
            embeddings: Corresponding embedding vectors (same order).

        Raises:
            ValueError: If lengths do not match.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                "must have the same length."
            )

        # Filter out pure-image chunks (no text to search over)
        text_pairs = [
            (c, e) for c, e in zip(chunks, embeddings) if c.text.strip()
        ]

        if not text_pairs:
            logger.warning("No text chunks to upsert (all chunks are image-only).")
            return

        ids = [c.chunk_id for c, _ in text_pairs]
        documents = [c.text for c, _ in text_pairs]
        metas = [
            {
                **c.metadata,
                "source": c.source,
                "has_images": len(c.images) > 0,
                "num_images": len(c.images),
            }
            for c, _ in text_pairs
        ]
        vecs = [e.tolist() for _, e in text_pairs]

        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=vecs,
            metadatas=metas,
        )
        logger.info("Upserted chunks", count=len(text_pairs))

    # ─── Read ─────────────────────────────────────────────────────────────────

    def query(
        self,
        embedding: np.ndarray,
        top_k: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the top-k most similar chunks for a query embedding.

        Args:
            embedding: Query vector (float32 numpy array).
            top_k:     Number of results; defaults to config ``top_k_retrieval``.

        Returns:
            List of dicts with keys: ``chunk_id``, ``text``, ``metadata``,
            ``distance``, ``score`` (= 1 - distance for cosine space).
        """
        settings = get_settings()
        k = top_k or settings.top_k_retrieval

        results = self._collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=min(k, self._collection.count() or 1),
            include=["documents", "metadatas", "distances"],
        )

        hits: list[dict[str, Any]] = []
        for chunk_id, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append(
                {
                    "chunk_id": chunk_id,
                    "text": doc,
                    "metadata": meta,
                    "distance": dist,
                    "score": 1.0 - dist,   # cosine similarity ≈ 1 - cosine distance
                }
            )

        logger.debug("Vector query complete", top_k=k, hits=len(hits))
        return hits

    # ─── Stats ────────────────────────────────────────────────────────────────

    def count(self) -> int:
        """Return total number of stored embeddings."""
        return self._collection.count()

    def delete_source(self, source: str) -> None:
        """Remove all chunks from a specific source document."""
        self._collection.delete(where={"source": source})
        logger.info("Deleted chunks for source", source=source)
