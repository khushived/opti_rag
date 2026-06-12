"""
app/queue/local_queue.py
─────────────────────────
Drop-in replacement for RabbitMQPublisher that uses Python's built-in
asyncio.Queue instead of RabbitMQ.

Why: RabbitMQ requires Erlang (a 148 MB runtime) just to install.
     This module gives you the exact same API — no external server needed.

How it works:
  - publish_query()  →  puts a job into an asyncio.Queue
  - A background worker coroutine reads from the queue, runs the RAG
    pipeline, and resolves a Future with the result.
  - The caller awaits the Future — identical behaviour to the RPC pattern.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

from app.core.logging import get_logger

logger = get_logger(__name__)


class LocalQueuePublisher:
    """
    In-process message queue that mimics the RabbitMQPublisher interface.

    The RAG pipeline runs in the same asyncio event loop — no separate
    process or external broker required.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        # Fake connection flags so health-check code stays the same
        self._connection = _FakeConnection()

    async def connect(self) -> None:
        """Start the background worker coroutine."""
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info("LocalQueuePublisher started (no RabbitMQ needed)")

    async def publish_query(
        self,
        query: str,
        session_id: Optional[str] = None,
        multimodal: bool = True,
        top_k: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Submit a query and wait for the result (same API as RabbitMQPublisher).
        """
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()

        await self._queue.put(
            {
                "query": query,
                "session_id": session_id,
                "multimodal": multimodal,
                "top_k": top_k,
                "future": future,
            }
        )

        logger.info("Query enqueued (local)", query_preview=query[:60])
        return await future

    async def close(self) -> None:
        """Cancel the worker gracefully."""
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("LocalQueuePublisher stopped")

    # ─── Worker loop ──────────────────────────────────────────────────────────

    async def _worker_loop(self) -> None:
        """
        Continuously drain the queue and run the RAG pipeline for each job.
        Imported lazily to avoid circular imports.
        """
        # Lazy imports — the consumer module imports vector store, retriever etc.
        from app.cache.semantic_cache import SemanticCache
        from app.generation.llm_client import generate
        from app.retrieval.retriever import Retriever
        from app.retrieval.vector_store import VectorStore

        import redis.asyncio as aioredis
        from app.core.config import get_settings

        settings = get_settings()

        # Initialise shared components once
        loop = asyncio.get_event_loop()
        vector_store: VectorStore = await loop.run_in_executor(None, VectorStore)
        retriever = Retriever(vector_store)

        redis_client = aioredis.from_url(
            settings.redis_url, decode_responses=False, max_connections=5
        )
        cache = SemanticCache(redis_client)

        logger.info("Local RAG worker ready")

        while True:
            job = await self._queue.get()
            future: asyncio.Future = job.pop("future")
            t0 = time.monotonic()

            try:
                result = await _run_rag_pipeline(
                    retriever=retriever,
                    cache=cache,
                    **job,
                )
                result["latency_ms"] = round((time.monotonic() - t0) * 1000, 1)
                if not future.done():
                    future.set_result(result)

            except Exception as exc:  # noqa: BLE001
                logger.exception("RAG pipeline error", error=str(exc))
                error_result = {
                    "answer": f"Pipeline error: {exc}",
                    "sources": [],
                    "cache_hit": False,
                    "latency_ms": round((time.monotonic() - t0) * 1000, 1),
                    "model": "error",
                    "is_multimodal": False,
                    "context_chunks": 0,
                }
                if not future.done():
                    future.set_result(error_result)

            self._queue.task_done()


async def _run_rag_pipeline(
    retriever,
    cache,
    query: str,
    session_id: Optional[str],
    multimodal: bool,
    top_k: Optional[int],
) -> dict[str, Any]:
    """Run the full RAG pipeline: cache → retrieve → generate → cache store."""

    # 1. Semantic cache check
    cache_entry = await cache.lookup(query)
    if cache_entry:
        return {
            "answer": cache_entry["answer"],
            "sources": [],
            "cache_hit": True,
            "similarity": cache_entry.get("similarity"),
            "cached_query": cache_entry.get("cached_query"),
            "model": "cache",
            "is_multimodal": False,
            "context_chunks": 0,
        }

    # 2. Retrieve context
    retrieval = await retriever.retrieve(query, top_k=top_k)

    # 3. Generate answer
    gen_result = await generate(
        query=query,
        context_chunks=retrieval.text_chunks,
        images=None,
    )

    # 4. Cache for future similar queries
    await cache.store(
        query=query,
        answer=gen_result.answer,
        embedding=retrieval.query_embedding,
    )

    return {
        "answer": gen_result.answer,
        "sources": retrieval.sources,
        "cache_hit": False,
        "model": gen_result.model,
        "is_multimodal": gen_result.is_multimodal,
        "context_chunks": len(retrieval.text_chunks),
    }


class _FakeConnection:
    """Mimics aio_pika connection so health-check code doesn't break."""
    is_closed = False
