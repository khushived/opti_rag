"""
app/api/routes.py
──────────────────
FastAPI route definitions.

Endpoints:
  POST /ingest        — Upload and index a document
  POST /query         — Ask a question; returns grounded LLM answer
  GET  /health        — Liveness + dependency checks
  GET  /cache/stats   — Semantic cache hit/miss statistics
  DELETE /cache/flush — Clear the semantic cache (dev/test only)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse

from app.api.models import (
    CacheFlushResponse,
    CacheStatsResponse,
    ErrorResponse,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from app.core.config import get_settings
from app.core.logging import get_logger
from app.ingestion.chunker import chunk_documents
from app.ingestion.document_loader import load_document
from app.ingestion.embedder import embed_texts_sync

logger = get_logger(__name__)

router = APIRouter()


# ─── Dependency helpers ───────────────────────────────────────────────────────

def get_publisher(request: Request):
    return request.app.state.publisher


def get_vector_store(request: Request):
    return request.app.state.vector_store


def get_cache(request: Request):
    return request.app.state.semantic_cache


# ─── Health ───────────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Liveness and dependency health check",
)
async def health_check(request: Request):
    """
    Check the status of all downstream dependencies:
    Redis, RabbitMQ, Ollama, and ChromaDB.
    """
    import httpx
    import redis.asyncio as aioredis

    settings = get_settings()
    services: dict[str, str] = {}

    # Redis
    try:
        redis_client = aioredis.from_url(settings.redis_url)
        await redis_client.ping()
        await redis_client.aclose()
        services["redis"] = "ok"
    except Exception as exc:
        services["redis"] = f"error: {exc}"

    # Ollama
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
            services["ollama"] = "ok" if resp.status_code == 200 else f"http {resp.status_code}"
    except Exception as exc:
        services["ollama"] = f"error: {exc}"

    # Local Queue worker status
    publisher = request.app.state.publisher
    services["queue"] = (
        "ok"
        if publisher._worker_task and not publisher._worker_task.done()
        else "disconnected"
    )

    # Vector store
    vector_store = request.app.state.vector_store
    doc_count = vector_store.count()
    services["chromadb"] = "ok"

    overall = "healthy" if all(v == "ok" for v in services.values()) else "degraded"
    return HealthResponse(
        status=overall,
        services=services,
        vector_store_docs=doc_count,
    )


# ─── Document Ingestion ───────────────────────────────────────────────────────

@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Documents"],
    summary="Upload and index a document",
)
async def ingest_document(
    file: UploadFile = File(..., description="PDF, image, text, or DOCX file"),
    vector_store=Depends(get_vector_store),
):
    """
    Upload a document, extract content, chunk it, embed with Ollama,
    and store in ChromaDB.

    Supported formats: PDF, PNG, JPG, WEBP, TXT, MD, DOCX
    """
    settings = get_settings()
    t0 = time.monotonic()

    # Save upload to disk
    upload_path = settings.upload_dir / file.filename
    content = await file.read()
    upload_path.write_bytes(content)

    logger.info("File uploaded", filename=file.filename, size=len(content))

    try:
        # Load → chunk → embed → store
        raw_docs = load_document(upload_path)
        chunks = chunk_documents(raw_docs)

        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No processable content found in the uploaded file.",
            )

        # Embed text chunks only (image chunks have empty text)
        text_chunks = [c for c in chunks if c.text.strip()]
        if not text_chunks:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Document contains only images with no extractable text.",
            )

        embeddings = embed_texts_sync([c.text for c in text_chunks])
        vector_store.upsert_chunks(text_chunks, embeddings)

        elapsed = round((time.monotonic() - t0) * 1000, 1)
        logger.info(
            "Ingestion complete",
            filename=file.filename,
            chunks=len(text_chunks),
            latency_ms=elapsed,
        )
        return IngestResponse(
            status="success",
            filename=file.filename,
            chunks_indexed=len(text_chunks),
            message=f"Indexed {len(text_chunks)} chunks in {elapsed}ms.",
        )

    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc


# ─── Query ────────────────────────────────────────────────────────────────────

@router.post(
    "/query",
    response_model=QueryResponse,
    tags=["RAG"],
    summary="Ask a question against indexed documents",
)
async def query(
    body: QueryRequest,
    publisher=Depends(get_publisher),
):
    """
    Submit a natural language query.

    The request is processed via an in-process local queue.
    The RAG worker:
    1. Checks the semantic cache (Redis) → instant reply on hit
    2. Retrieves top-k relevant document chunks (ChromaDB)
    3. Generates an answer via Ollama (llava for vision, mistral for text)
    4. Caches the result for future similar queries

    Returns a grounded answer with source attribution.
    """
    try:
        result = await publisher.publish_query(
            query=body.query,
            session_id=body.session_id,
            multimodal=body.multimodal,
            top_k=body.top_k,
        )
    except TimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Query failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {exc}",
        ) from exc

    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result["error"],
        )

    return QueryResponse(**result)


# ─── Cache Stats ──────────────────────────────────────────────────────────────

@router.get(
    "/cache/stats",
    response_model=CacheStatsResponse,
    tags=["Cache"],
    summary="Semantic cache hit/miss statistics",
)
async def cache_stats(cache=Depends(get_cache)):
    """Return aggregated cache statistics including hit rate and entry count."""
    stats = await cache.get_stats()
    return CacheStatsResponse(**stats)


@router.delete(
    "/cache/flush",
    response_model=CacheFlushResponse,
    tags=["Cache"],
    summary="Flush all semantic cache entries",
)
async def cache_flush(cache=Depends(get_cache)):
    """
    Delete all entries from the semantic cache.
    Useful during development or after re-indexing documents.
    """
    deleted = await cache.flush()
    return CacheFlushResponse(
        deleted=deleted,
        message=f"Flushed {deleted} cache entries.",
    )
