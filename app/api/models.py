"""
app/api/models.py
──────────────────
Pydantic v2 request / response schemas for the REST API.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


# ─── Ingest ───────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    status: str
    filename: str
    chunks_indexed: int
    message: str


# ─── Query ────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Natural language question to answer from indexed documents.",
        examples=["What are the main findings of the report?"],
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session identifier for grouping related queries.",
    )
    multimodal: bool = Field(
        default=True,
        description="Allow the system to use vision models when image context is available.",
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Override the number of retrieved chunks (1–20).",
    )

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        return v.strip()


class QueryResponse(BaseModel):
    answer: str = Field(description="LLM-generated answer grounded in indexed documents.")
    sources: list[str] = Field(default_factory=list, description="Source file paths referenced.")
    cache_hit: bool = Field(description="True if this response was served from semantic cache.")
    latency_ms: float = Field(description="Total response time in milliseconds.")
    model: Optional[str] = Field(default=None, description="Ollama model used for generation.")
    similarity: Optional[float] = Field(
        default=None, description="Cache similarity score (only on cache hits)."
    )
    cached_query: Optional[str] = Field(
        default=None, description="The previously cached query that triggered the hit."
    )
    is_multimodal: Optional[bool] = Field(
        default=None, description="Whether a vision model was used."
    )
    context_chunks: Optional[int] = Field(
        default=None, description="Number of document chunks used as context."
    )


# ─── Cache ────────────────────────────────────────────────────────────────────

class CacheStatsResponse(BaseModel):
    hits: int
    misses: int
    total_lookups: int
    hit_rate: str
    entry_count: int
    threshold: float
    ttl_seconds: int


class CacheFlushResponse(BaseModel):
    deleted: int
    message: str


# ─── Health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    services: dict[str, str]
    vector_store_docs: int


# ─── Error ────────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    detail: str
    error_type: Optional[str] = None
