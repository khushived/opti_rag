"""
tests/test_retrieval.py
────────────────────────
Unit tests for the chunker and vector store retrieval logic.
ChromaDB is tested with an in-memory (ephemeral) client to avoid disk I/O.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from app.ingestion.chunker import Chunk, chunk_documents
from app.ingestion.document_loader import RawDocument


# ─── Chunker tests ────────────────────────────────────────────────────────────

def test_chunk_plain_text():
    docs = [
        RawDocument(
            source="test.txt",
            content_type="text",
            text="word " * 600,  # 600 words — should produce multiple chunks
            metadata={"filename": "test.txt"},
        )
    ]
    chunks = chunk_documents(docs)
    assert len(chunks) > 1
    for c in chunks:
        assert isinstance(c, Chunk)
        assert c.text.strip()


def test_chunk_empty_text():
    docs = [
        RawDocument(
            source="empty.txt",
            content_type="text",
            text="",
            metadata={},
        )
    ]
    chunks = chunk_documents(docs)
    assert chunks == []


def test_chunk_image_only():
    fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    docs = [
        RawDocument(
            source="chart.png",
            content_type="image",
            text="",
            images=[fake_png],
            metadata={"filename": "chart.png"},
        )
    ]
    chunks = chunk_documents(docs)
    assert len(chunks) == 1
    assert chunks[0].images == [fake_png]
    assert chunks[0].text == ""


def test_chunk_mixed_attaches_images_to_first_chunk():
    fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    docs = [
        RawDocument(
            source="report.pdf",
            content_type="mixed",
            text="word " * 600,
            images=[fake_png],
            metadata={"page": 1, "filename": "report.pdf"},
        )
    ]
    chunks = chunk_documents(docs)
    assert len(chunks) > 1
    # Only first chunk should have images
    assert chunks[0].images == [fake_png]
    for c in chunks[1:]:
        assert c.images == []


def test_chunk_ids_are_unique():
    docs = [
        RawDocument(
            source="doc.txt",
            content_type="text",
            text="word " * 1200,
            metadata={"page": 1},
        )
    ]
    chunks = chunk_documents(docs)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "Chunk IDs must be unique"


# ─── VectorStore tests ────────────────────────────────────────────────────────

def test_vector_store_upsert_and_query(tmp_path, monkeypatch):
    """Test VectorStore with a real ephemeral ChromaDB client."""
    import chromadb
    from app.retrieval.vector_store import VectorStore

    # Patch settings to use a temp dir and test collection
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("CHROMA_COLLECTION_NAME", "test_collection")
    monkeypatch.setenv("TOP_K_RETRIEVAL", "3")

    # Re-import settings with fresh cache
    from app.core.config import get_settings
    get_settings.cache_clear()

    store = VectorStore()

    chunks = [
        Chunk(
            chunk_id=f"doc:p1:c{i}",
            source="doc.txt",
            text=f"This is test chunk number {i} about machine learning.",
            metadata={"page": 1, "filename": "doc.txt"},
        )
        for i in range(5)
    ]
    dim = 768
    embeddings = [np.random.rand(dim).astype(np.float32) for _ in chunks]

    store.upsert_chunks(chunks, embeddings)
    assert store.count() == 5

    # Query with a random vector — should return 3 results
    query_vec = np.random.rand(dim).astype(np.float32)
    hits = store.query(query_vec, top_k=3)
    assert len(hits) == 3
    for h in hits:
        assert "text" in h
        assert "score" in h
        assert 0.0 <= h["score"] <= 1.5  # cosine similarity range

    get_settings.cache_clear()


def test_vector_store_skips_empty_text_chunks(tmp_path, monkeypatch):
    """Chunks with no text should be silently skipped."""
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma2"))
    monkeypatch.setenv("CHROMA_COLLECTION_NAME", "test_empty")

    from app.core.config import get_settings
    get_settings.cache_clear()

    from app.retrieval.vector_store import VectorStore
    store = VectorStore()

    chunks = [
        Chunk(chunk_id="img:p1:img0", source="a.png", text="", images=[b"..."], metadata={})
    ]
    embeddings = [np.zeros(768, dtype=np.float32)]

    store.upsert_chunks(chunks, embeddings)
    assert store.count() == 0  # image-only chunk should be skipped

    get_settings.cache_clear()
