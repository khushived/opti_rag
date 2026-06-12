"""
app/ingestion/embedder.py
─────────────────────────
Generates dense vector embeddings via Ollama's ``nomic-embed-text`` model.

Provides both synchronous (batch ingest) and asynchronous (query-time)
interfaces.  Uses ``tenacity`` for retry logic on transient HTTP failures.
"""

from __future__ import annotations

import asyncio
from typing import Sequence

import httpx
import numpy as np
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


# ─── Retry policy ─────────────────────────────────────────────────────────────

_RETRY_POLICY = dict(
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(4),
    reraise=True,
)


# ─── Synchronous embedder (used during batch ingestion) ───────────────────────

def embed_texts_sync(texts: list[str]) -> list[np.ndarray]:
    """
    Embed a batch of text strings using Ollama (synchronous).
    Falls back to zero vectors if Ollama is not available (demo mode).
    """
    settings = get_settings()
    url = f"{settings.ollama_base_url}/api/embed"

    vectors: list[np.ndarray] = []
    with httpx.Client(timeout=10) as client:
        for text in texts:
            if not text.strip():
                vectors.append(np.zeros(768, dtype=np.float32))
                continue

            try:
                response = client.post(
                    url,
                    json={"model": settings.ollama_embed_model, "input": text},
                )
                response.raise_for_status()
                data = response.json()
                embedding = data["embeddings"][0]
                vectors.append(np.array(embedding, dtype=np.float32))
            except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError):
                logger.warning("Ollama unavailable — using zero embedding (demo mode)")
                vectors.append(np.zeros(768, dtype=np.float32))

    logger.debug("Embedded texts (sync)", count=len(texts))
    return vectors


# ─── Asynchronous embedder (used at query time) ───────────────────────────────

async def embed_text_async(text: str) -> np.ndarray:
    """
    Embed a single text string asynchronously.
    Falls back to a zero vector if Ollama is not available.
    """
    settings = get_settings()
    url = f"{settings.ollama_base_url}/api/embed"

    if not text.strip():
        return np.zeros(768, dtype=np.float32)

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                url,
                json={"model": settings.ollama_embed_model, "input": text},
            )
            response.raise_for_status()
            data = response.json()

        embedding = data["embeddings"][0]
        logger.debug("Embedded text (async)", text_preview=text[:60])
        return np.array(embedding, dtype=np.float32)

    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError):
        logger.warning("Ollama unavailable — using zero embedding (demo mode)")
        return np.zeros(768, dtype=np.float32)


async def embed_texts_async(texts: list[str]) -> list[np.ndarray]:
    """Embed multiple texts concurrently."""
    tasks = [embed_text_async(t) for t in texts]
    return list(await asyncio.gather(*tasks))


# ─── Utility ──────────────────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def normalise(v: np.ndarray) -> np.ndarray:
    """Return L2-normalised vector."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v
