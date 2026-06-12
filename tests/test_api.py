"""
tests/test_api.py
──────────────────
Integration tests for the FastAPI API layer.

Uses FastAPI's TestClient with all heavy dependencies (RabbitMQ, Redis,
ChromaDB, Ollama) mocked so tests run without any running services.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_vector_store():
    vs = MagicMock()
    vs.count.return_value = 42
    vs.upsert_chunks.return_value = None
    vs.query.return_value = [
        {
            "chunk_id": "doc:p1:c0",
            "text": "Paris is the capital of France.",
            "metadata": {"source": "france.pdf", "filename": "france.pdf", "page": 1, "has_images": False},
            "distance": 0.05,
            "score": 0.95,
        }
    ]
    return vs


@pytest.fixture
def mock_publisher():
    pub = AsyncMock()
    pub._worker_task = MagicMock()
    pub._worker_task.done.return_value = False
    pub.publish_query.return_value = {
        "answer": "Paris is the capital of France.",
        "sources": ["france.pdf"],
        "cache_hit": False,
        "latency_ms": 512.0,
        "model": "mistral",
        "is_multimodal": False,
        "context_chunks": 1,
    }
    return pub


@pytest.fixture
def mock_cache():
    cache = AsyncMock()
    cache.get_stats.return_value = {
        "hits": 10,
        "misses": 5,
        "total_lookups": 15,
        "hit_rate": "66.7%",
        "entry_count": 8,
        "threshold": 0.92,
        "ttl_seconds": 86400,
    }
    cache.flush.return_value = 8
    return cache


@pytest.fixture
def client(mock_vector_store, mock_publisher, mock_cache):
    """TestClient with all external services mocked."""
    from app.main import create_app

    app = create_app()

    # Pre-populate app.state before lifespan runs
    app.state.vector_store = mock_vector_store
    app.state.publisher = mock_publisher
    app.state.semantic_cache = mock_cache

    # Bypass the lifespan (don't connect to real services)
    with patch("app.main.VectorStore", return_value=mock_vector_store), \
         patch("app.main.aioredis.from_url", return_value=AsyncMock()), \
         patch("app.main.SemanticCache", return_value=mock_cache), \
         patch("app.main.LocalQueuePublisher", return_value=mock_publisher):
        with TestClient(app, raise_server_exceptions=True) as c:
            # Inject mocks into state
            c.app.state.vector_store = mock_vector_store
            c.app.state.publisher = mock_publisher
            c.app.state.semantic_cache = mock_cache
            yield c


# ─── Health tests ─────────────────────────────────────────────────────────────

def test_root_redirect(client):
    resp = client.get("/")
    assert resp.status_code == 200
    content_type = resp.headers.get("content-type", "")
    if "text/html" in content_type:
        assert b"OptiRAG" in resp.content
    else:
        assert "docs" in resp.json()


def test_health_endpoint_returns_200(client):
    with patch("redis.asyncio.from_url") as mock_from_url, \
         patch("httpx.AsyncClient") as mock_httpx:

        # Mock Redis ping
        mock_redis_instance = AsyncMock()
        mock_redis_instance.ping = AsyncMock()
        mock_from_url.return_value = mock_redis_instance

        # Mock Ollama response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_httpx.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )

        resp = client.get("/api/v1/health")

    # Health may be degraded due to mocked environment, but endpoint should respond
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "services" in data
    assert "vector_store_docs" in data


# ─── Query tests ──────────────────────────────────────────────────────────────

def test_query_returns_answer(client, mock_publisher):
    resp = client.post(
        "/api/v1/query",
        json={"query": "What is the capital of France?"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "Paris is the capital of France."
    assert data["cache_hit"] is False
    assert data["latency_ms"] > 0
    mock_publisher.publish_query.assert_called_once()


def test_query_too_short_rejected(client):
    resp = client.post("/api/v1/query", json={"query": "Hi"})
    assert resp.status_code == 422


def test_query_missing_body(client):
    resp = client.post("/api/v1/query", json={})
    assert resp.status_code == 422


def test_query_timeout_returns_504(client, mock_publisher):
    mock_publisher.publish_query.side_effect = TimeoutError("Worker timeout")
    resp = client.post(
        "/api/v1/query",
        json={"query": "What happened in the annual report?"},
    )
    assert resp.status_code == 504


# ─── Cache tests ──────────────────────────────────────────────────────────────

def test_cache_stats(client):
    resp = client.get("/api/v1/cache/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["hits"] == 10
    assert data["misses"] == 5
    assert data["hit_rate"] == "66.7%"


def test_cache_flush(client, mock_cache):
    resp = client.delete("/api/v1/cache/flush")
    assert resp.status_code == 200
    data = resp.json()
    assert data["deleted"] == 8
    mock_cache.flush.assert_called_once()


# ─── Ingestion tests ──────────────────────────────────────────────────────────

def test_ingest_text_file(client, tmp_path, mock_vector_store):
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("word " * 600)

    with patch("app.api.routes.load_document") as mock_load, \
         patch("app.api.routes.chunk_documents") as mock_chunk, \
         patch("app.api.routes.embed_texts_sync") as mock_embed:

        from app.ingestion.chunker import Chunk
        from app.ingestion.document_loader import RawDocument
        import numpy as np

        mock_load.return_value = [
            RawDocument(source="sample.txt", content_type="text", text="word " * 600, metadata={"filename": "sample.txt"})
        ]
        mock_chunks = [
            Chunk(chunk_id=f"sample.txt:p1:c{i}", source="sample.txt", text=f"chunk {i}", metadata={})
            for i in range(3)
        ]
        mock_chunk.return_value = mock_chunks
        mock_embed.return_value = [np.zeros(768) for _ in mock_chunks]

        with open(sample_file, "rb") as f:
            resp = client.post(
                "/api/v1/ingest",
                files={"file": ("sample.txt", f, "text/plain")},
            )

    assert resp.status_code == 201
    data = resp.json()
    assert data["status"] == "success"
    assert data["chunks_indexed"] == 3
