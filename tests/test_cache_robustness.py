"""
tests/test_cache_robustness.py
──────────────────────────────
Tests to ensure that SemanticCache gracefully handles Redis errors
(e.g., ConnectionError, TimeoutError, etc.) instead of raising exceptions.
"""

from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import pytest
from redis.exceptions import ConnectionError

from app.cache.semantic_cache import SemanticCache

@pytest.fixture
def erroring_redis():
    """A mock Redis client that raises ConnectionError on all async methods."""
    redis = MagicMock()
    
    # We want any method call to return an async mock that raises ConnectionError when awaited
    async def raise_conn_error(*args, **kwargs):
        raise ConnectionError("Mocked Redis connection failure")
        
    redis.zrange = raise_conn_error
    redis.zcard = raise_conn_error
    redis.zadd = raise_conn_error
    redis.zrem = raise_conn_error
    redis.delete = raise_conn_error
    redis.hset = raise_conn_error
    redis.hgetall = raise_conn_error
    redis.hincrby = raise_conn_error
    
    # Mock pipeline
    class ErroringPipeline:
        def __init__(self):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return False
        def hgetall(self, *args, **kwargs):
            return self
        def hset(self, *args, **kwargs):
            return self
        def expire(self, *args, **kwargs):
            return self
        def zadd(self, *args, **kwargs):
            return self
        def delete(self, *args, **kwargs):
            return self
        def zrem(self, *args, **kwargs):
            return self
        async def execute(self):
            raise ConnectionError("Mocked pipeline execute failure")
            
    redis.pipeline = MagicMock(return_value=ErroringPipeline())
    return redis

@pytest.fixture
def robust_cache(erroring_redis):
    return SemanticCache(erroring_redis)

@pytest.mark.asyncio
async def test_lookup_on_redis_error(robust_cache):
    """lookup should return None and handle the error when Redis throws an exception."""
    with patch(
        "app.cache.semantic_cache.embed_text_async",
        AsyncMock(return_value=np.zeros(768, dtype=np.float32)),
    ):
        result = await robust_cache.lookup("Test query")
    assert result is None

@pytest.mark.asyncio
async def test_store_on_redis_error(robust_cache):
    """store should complete without raising exceptions when Redis throws an exception."""
    embedding = np.zeros(768, dtype=np.float32)
    entry_id = await robust_cache.store("Test query", "Test answer", embedding)
    assert isinstance(entry_id, str)
    assert len(entry_id) == 16

@pytest.mark.asyncio
async def test_invalidate_on_redis_error(robust_cache):
    """invalidate should return False and handle the error when Redis throws an exception."""
    success = await robust_cache.invalidate("some_id")
    assert success is False

@pytest.mark.asyncio
async def test_flush_on_redis_error(robust_cache):
    """flush should return 0 and handle the error when Redis throws an exception."""
    deleted = await robust_cache.flush()
    assert deleted == 0

@pytest.mark.asyncio
async def test_get_stats_on_redis_error(robust_cache):
    """get_stats should return safe default stats and handle the error when Redis throws an exception."""
    stats = await robust_cache.get_stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 0
    assert stats["total_lookups"] == 0
    assert stats["hit_rate"] == "n/a"
    assert stats["entry_count"] == 0
