"""
tests/test_cache.py
────────────────────
Unit tests for the semantic cache module.
Uses a local Redis instance (or fakeredis for isolation).
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_redis():
    """A minimal in-memory mock of the Redis async client."""
    store: dict = {}
    zsets: dict = {}

    redis = AsyncMock()

    async def hset(key, mapping=None, **kwargs):
        store[key] = mapping or {}

    async def hgetall(key):
        return {
            k.encode(): v.encode() if isinstance(v, str) else v
            for k, v in store.get(key, {}).items()
        }

    async def expire(key, ttl):
        pass

    async def zadd(key, mapping):
        zsets.setdefault(key, {}).update(mapping)

    async def zrange(key, start, end):
        ids = sorted(zsets.get(key, {}).keys())
        if end == -1:
            return [i.encode() for i in ids]
        return [i.encode() for i in ids[start : end + 1]]

    async def zcard(key):
        return len(zsets.get(key, {}))

    async def zrem(key, *members):
        for m in members:
            zsets.get(key, {}).pop(m if isinstance(m, str) else m.decode(), None)

    async def delete(*keys):
        for k in keys:
            store.pop(k, None)
        return len(keys)

    async def hincrby(key, field, amount):
        store.setdefault(key, {})[field] = str(
            int(store.get(key, {}).get(field, "0")) + amount
        )

    redis.hset = hset
    redis.hgetall = hgetall
    redis.expire = expire
    redis.zadd = zadd
    redis.zrange = zrange
    redis.zcard = zcard
    redis.zrem = zrem
    redis.delete = delete
    redis.hincrby = hincrby

    # pipeline mock
    class FakePipeline:
        def __init__(self):
            self.commands = []
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return False
        def hgetall(self, key):
            self.commands.append(("hgetall", key))
            return self
        def hset(self, key, mapping=None, **kwargs):
            self.commands.append(("hset", key, mapping))
            return self
        def expire(self, key, ttl):
            self.commands.append(("expire", key, ttl))
            return self
        def zadd(self, key, mapping):
            self.commands.append(("zadd", key, mapping))
            return self
        def delete(self, *keys):
            self.commands.append(("delete", keys))
            return self
        def zrem(self, key, *members):
            self.commands.append(("zrem", key, members))
            return self
        async def execute(self):
            results = []
            for cmd in self.commands:
                name = cmd[0]
                if name == "hgetall":
                    res = await redis.hgetall(cmd[1])
                    results.append(res)
                elif name == "hset":
                    await redis.hset(cmd[1], cmd[2])
                    results.append(1)
                elif name == "expire":
                    await redis.expire(cmd[1], cmd[2])
                    results.append(True)
                elif name == "zadd":
                    await redis.zadd(cmd[1], cmd[2])
                    results.append(1)
                elif name == "delete":
                    res = await redis.delete(*cmd[1])
                    results.append(res)
                elif name == "zrem":
                    await redis.zrem(cmd[1], *cmd[2])
                    results.append(1)
            self.commands.clear()
            return results

    redis.pipeline = MagicMock(return_value=FakePipeline())

    return redis


@pytest.fixture
def cache(mock_redis):
    from app.cache.semantic_cache import SemanticCache

    return SemanticCache(mock_redis)


# ─── Tests ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cache_miss_on_empty(cache):
    """Cache lookup on an empty store should return None."""
    with patch(
        "app.cache.semantic_cache.embed_text_async",
        AsyncMock(return_value=np.zeros(768, dtype=np.float32)),
    ):
        result = await cache.lookup("What is the revenue for Q3?")
    assert result is None


@pytest.mark.asyncio
async def test_cache_store_and_hit(cache, mock_redis):
    """A stored entry should be retrieved on a semantically identical query."""
    query = "What is the capital of France?"
    answer = "The capital of France is Paris."
    embedding = np.random.rand(768).astype(np.float32)
    embedding /= np.linalg.norm(embedding)

    # Manually populate the store to simulate a stored entry
    entry_id = "test_entry_01"
    entry_key = f"{cache._prefix}{entry_id}"

    mock_redis._store_data = {
        entry_key: {
            b"query": query.encode(),
            b"answer": answer.encode(),
            b"embedding": json.dumps(embedding.tolist()).encode(),
            b"timestamp": b"1234567890.0",
        }
    }
    mock_redis._zset_data = {cache._index_key: {entry_id: 0.0}}

    # Override hgetall to return from our fake store
    async def hgetall(key):
        return mock_redis._store_data.get(key, {})

    async def zrange(key, start, end):
        ids = list(mock_redis._zset_data.get(key, {}).keys())
        if end == -1:
            return [i.encode() for i in ids]
        return [i.encode() for i in ids[start : end + 1]]

    mock_redis.hgetall = hgetall
    mock_redis.zrange = zrange

    # Same embedding → similarity = 1.0 → should hit
    with patch(
        "app.cache.semantic_cache.embed_text_async",
        AsyncMock(return_value=embedding),
    ):
        result = await cache.lookup(query)

    assert result is not None
    assert result["answer"] == answer
    assert result["similarity"] >= cache._threshold


@pytest.mark.asyncio
async def test_cosine_similarity_utility():
    """Test the cosine similarity helper directly."""
    from app.ingestion.embedder import cosine_similarity

    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    assert cosine_similarity(a, b) == pytest.approx(1.0)

    c = np.array([0.0, 1.0, 0.0])
    assert cosine_similarity(a, c) == pytest.approx(0.0)

    d = np.array([-1.0, 0.0, 0.0])
    assert cosine_similarity(a, d) == pytest.approx(-1.0)


@pytest.mark.asyncio
async def test_cache_flush(cache):
    """flush() should remove all entries."""
    with patch(
        "app.cache.semantic_cache.embed_text_async",
        AsyncMock(return_value=np.zeros(768, dtype=np.float32)),
    ):
        deleted = await cache.flush()
    assert isinstance(deleted, int)


@pytest.mark.asyncio
async def test_cache_stats_empty(cache):
    stats = await cache.get_stats()
    assert "hits" in stats
    assert "misses" in stats
    assert "hit_rate" in stats
    assert "entry_count" in stats
