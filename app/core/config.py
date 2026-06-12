"""
app/core/config.py
──────────────────
Centralised application settings loaded from environment variables / .env file.
Uses pydantic-settings for type-safe configuration with validation.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ────────────────────────────────────────────────────────────
    app_name: str = Field(default="OptiRAG", description="Application name")
    app_env: Literal["development", "production", "testing"] = Field(
        default="development"
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO"
    )

    # ── Ollama ─────────────────────────────────────────────────────────────────
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_embed_model: str = Field(default="nomic-embed-text")
    ollama_text_model: str = Field(default="mistral")
    ollama_vision_model: str = Field(default="llava")
    ollama_timeout: int = Field(default=120, description="HTTP timeout in seconds")

    # ── Redis ──────────────────────────────────────────────────────────────────
    redis_url: str = Field(default="redis://localhost:6379/0")
    redis_cache_ttl: int = Field(default=86400, description="Cache TTL in seconds")
    redis_cache_prefix: str = Field(default="opti_rag:cache:")

    # ── Semantic Cache ─────────────────────────────────────────────────────────
    semantic_cache_threshold: float = Field(
        default=0.92,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold for cache hits",
    )
    semantic_cache_max_entries: int = Field(default=10_000)

    # ── ChromaDB ───────────────────────────────────────────────────────────────
    chroma_persist_dir: Path = Field(default=Path("./data/chroma_db"))
    chroma_collection_name: str = Field(default="rag_documents")

    # ── Document Ingestion ─────────────────────────────────────────────────────
    upload_dir: Path = Field(default=Path("./data/uploads"))
    chunk_size: int = Field(default=512, description="Approximate tokens per chunk")
    chunk_overlap: int = Field(default=64)
    top_k_retrieval: int = Field(default=5)

    # ── Generation ─────────────────────────────────────────────────────────────
    max_new_tokens: int = Field(default=1024)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    system_prompt: str = Field(
        default=(
            "You are a helpful assistant. Answer questions based only on the provided "
            "context. If the context does not contain enough information, say so clearly."
        )
    )

    @field_validator("redis_url", mode="before")
    @classmethod
    def validate_redis_url(cls, v: Any) -> str:
        if not v or not isinstance(v, str):
            return "redis://localhost:6379/0"
        v = v.strip().strip("\"'")
        if not (v.startswith("redis://") or v.startswith("rediss://") or v.startswith("unix://")):
            if not v:
                return "redis://localhost:6379/0"
            return f"redis://{v}"
        return v

    @field_validator("chroma_persist_dir", "upload_dir", mode="after")
    @classmethod
    def ensure_dir_exists(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
