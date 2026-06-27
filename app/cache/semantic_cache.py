"""
app/cache/semantic_cache.py
────────────────────────────
Redis-backed semantic cache for RAG query results.

How it works
────────────
1. Every cached entry is stored as a Redis Hash under the key:
       ``{prefix}{entry_id}``
   containing fields: ``query``, ``answer``, ``embedding`` (JSON list), ``timestamp``.

2. A Redis Sorted Set ``{prefix}index`` maps entry_id → 0 (score unused; 
   we store IDs for iteration).

3. On a cache LOOKUP:
   a. Embed the incoming query.
   b. Iterate over all cached embeddings (up to ``max_entries``).
   c. Compute cosine similarity to each cached embedding.
   d. If max similarity ≥ ``threshold`` → CACHE HIT → return cached answer.
   e. Otherwise → CACHE MISS.

4. On a cache STORE:
   - Serialise the embedding as a JSON list.
   - SET the Redis Hash with TTL.
   - Add the entry_id to the index sorted set.

Performance note
────────────────
For up to ~10 k entries the linear scan is fast (ms range).  For larger
deployments this should be replaced with a purpose-built vector index
(e.g. Redis Stack with HNSW, or a standalone Qdrant/Weaviate).
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from typing import Optional

import numpy as np
import redis.asyncio as aioredis

from app.core.config import get_settings
from app.core.logging import get_logger
from app.ingestion.embedder import cosine_similarity, embed_text_async

logger = get_logger(__name__)


class SemanticCache:
    """
    Async Redis semantic cache.

    Instantiate once at startup and pass around as a shared dependency.
    """

    def __init__(self, redis_client: aioredis.Redis) -> None:
        self._redis = redis_client
        settings = get_settings()
        self._prefix = settings.redis_cache_prefix
        self._threshold = settings.semantic_cache_threshold
        self._ttl = settings.redis_cache_ttl
        self._max_entries = settings.semantic_cache_max_entries
        self._index_key = f"{self._prefix}index"
        self._stats_key = f"{self._prefix}stats"

    # ─── Public API ───────────────────────────────────────────────────────────

    async def lookup(self, query: str) -> Optional[dict]:
        """
        Search the cache for a semantically similar previous query.

        Args:
            query: Incoming user query string.

        Returns:
            Dict with ``answer``, ``cached_query``, ``similarity``, and
            ``entry_id`` if a hit is found; ``None`` otherwise.
        """
        try:
            query_embedding = await embed_text_async(query)
        except Exception as exc:
            logger.warning("Failed to embed query for cache lookup", error=str(exc))
            return None

        entry_ids = await self._get_all_entry_ids()

        if not entry_ids:
            await self._increment_stat("misses")
            return None

        best_similarity = -1.0
        best_entry: Optional[dict] = None

        try:
            # Fetch all embeddings in a pipeline to reduce round-trips
            pipe = self._redis.pipeline(transaction=False)
            for eid in entry_ids:
                pipe.hgetall(f"{self._prefix}{eid}")
            results = await pipe.execute()
        except Exception as exc:
            logger.warning("Redis pipeline execution failed during lookup", error=str(exc))
            return None

        for eid, raw in zip(entry_ids, results):
            if not raw:
                continue
            try:
                cached_embedding = np.array(
                    json.loads(raw[b"embedding"]), dtype=np.float32
                )
                sim = cosine_similarity(query_embedding, cached_embedding)
                if sim > best_similarity:
                    best_similarity = sim
                    best_entry = {
                        "entry_id": eid,
                        "cached_query": raw.get(b"query", b"").decode("utf-8"),
                        "answer": raw.get(b"answer", b"").decode("utf-8"),
                        "similarity": sim,
                    }
            except Exception as exc:  # noqa: BLE001
                logger.warning("Cache entry decode error", entry_id=eid, error=str(exc))

        if best_entry and best_similarity >= self._threshold:
            await self._increment_stat("hits")
            logger.info(
                "Cache HIT",
                similarity=f"{best_similarity:.4f}",
                threshold=self._threshold,
                cached_query=best_entry["cached_query"][:60],
            )
            return best_entry

        await self._increment_stat("misses")
        logger.info(
            "Cache MISS",
            best_similarity=f"{best_similarity:.4f}" if best_entry else "n/a",
            threshold=self._threshold,
        )
        return None

    async def store(self, query: str, answer: str, embedding: Optional[np.ndarray] = None) -> str:
        """
        Store a query-answer pair in the cache.

        Args:
            query:     Original query string.
            answer:    LLM-generated answer.
            embedding: Pre-computed query embedding (avoids re-embedding).

        Returns:
            The new entry ID.
        """
        if embedding is None:
            try:
                embedding = await embed_text_async(query)
            except Exception as exc:
                logger.warning("Failed to embed query for cache store", error=str(exc))
                return ""

        entry_id = hashlib.sha256(query.encode()).hexdigest()[:16]

        payload = {
            "query": query,
            "answer": answer,
            "embedding": json.dumps(embedding.tolist()),
            "timestamp": str(time.time()),
        }

        try:
            pipe = self._redis.pipeline(transaction=True)
            pipe.hset(f"{self._prefix}{entry_id}", mapping=payload)
            pipe.expire(f"{self._prefix}{entry_id}", self._ttl)
            pipe.zadd(self._index_key, {entry_id: time.time()})
            await pipe.execute()

            # Trim index to max_entries (remove oldest)
            count = await self._redis.zcard(self._index_key)
            if count > self._max_entries:
                oldest_ids = await self._redis.zrange(
                    self._index_key, 0, count - self._max_entries - 1
                )
                if oldest_ids:
                    pipe = self._redis.pipeline(transaction=True)
                    for oid in oldest_ids:
                        pipe.delete(f"{self._prefix}{oid.decode()}")
                    pipe.zrem(self._index_key, *oldest_ids)
                    await pipe.execute()

            logger.debug("Cache STORE", entry_id=entry_id, query_preview=query[:60])
        except Exception as exc:
            logger.warning("Redis cache store operation failed", error=str(exc))

        return entry_id

    async def invalidate(self, entry_id: str) -> bool:
        """Delete a specific cache entry by ID."""
        try:
            deleted = await self._redis.delete(f"{self._prefix}{entry_id}")
            await self._redis.zrem(self._index_key, entry_id)
            return deleted > 0
        except Exception as exc:
            logger.warning("Redis invalidate failed", entry_id=entry_id, error=str(exc))
            return False

    async def flush(self) -> int:
        """Clear all cache entries. Returns number of keys deleted."""
        try:
            entry_ids = await self._get_all_entry_ids()
            if not entry_ids:
                return 0

            pipe = self._redis.pipeline(transaction=True)
            for eid in entry_ids:
                pipe.delete(f"{self._prefix}{eid}")
            pipe.delete(self._index_key)
            pipe.delete(self._stats_key)
            results = await pipe.execute()
            deleted = sum(1 for r in results[:-2] if r)
            logger.info("Cache flushed", deleted=deleted)
            return deleted
        except Exception as exc:
            logger.warning("Redis flush failed", error=str(exc))
            return 0

    async def get_stats(self) -> dict:
        """Return cache hit/miss statistics."""
        try:
            raw = await self._redis.hgetall(self._stats_key)
            hits = int(raw.get(b"hits", 0))
            misses = int(raw.get(b"misses", 0))
            total = hits + misses
            entry_count = await self._redis.zcard(self._index_key)
            return {
                "hits": hits,
                "misses": misses,
                "total_lookups": total,
                "hit_rate": f"{hits / total * 100:.1f}%" if total > 0 else "n/a",
                "entry_count": entry_count,
                "threshold": self._threshold,
                "ttl_seconds": self._ttl,
            }
        except Exception as exc:
            logger.warning("Failed to fetch Redis cache stats", error=str(exc))
            return {
                "hits": 0,
                "misses": 0,
                "total_lookups": 0,
                "hit_rate": "n/a",
                "entry_count": 0,
                "threshold": self._threshold,
                "ttl_seconds": self._ttl,
            }

    # ─── Internals ────────────────────────────────────────────────────────────

    async def _get_all_entry_ids(self) -> list[str]:
        """Return all entry IDs from the sorted set index."""
        try:
            raw_ids = await self._redis.zrange(self._index_key, 0, -1)
            return [eid.decode("utf-8") for eid in raw_ids]
        except Exception as exc:
            logger.warning("Failed to retrieve entry IDs from Redis", error=str(exc))
            return []

    async def _increment_stat(self, field: str) -> None:
        try:
            await self._redis.hincrby(self._stats_key, field, 1)
        except Exception as exc:
            logger.warning("Redis increment_stat failed", field=field, error=str(exc))
