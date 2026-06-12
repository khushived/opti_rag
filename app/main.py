"""
app/main.py
────────────
FastAPI application entrypoint.

Manages the full lifecycle of shared resources:
  - Local query queue publisher
  - ChromaDB vector store
  - Redis semantic cache

Run with:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.cache.semantic_cache import SemanticCache
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.queue.local_queue import LocalQueuePublisher
from app.retrieval.vector_store import VectorStore

setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start-up and tear-down of all shared resources."""
    settings = get_settings()
    logger.info("Starting OptiRAG", env=settings.app_env)

    # ── Vector Store (sync init, run in thread) ────────────────────────────
    import asyncio

    loop = asyncio.get_event_loop()
    app.state.vector_store = await loop.run_in_executor(None, VectorStore)

    # ── Redis + Semantic Cache ─────────────────────────────────────────────
    redis_client = aioredis.from_url(
        settings.redis_url, decode_responses=False, max_connections=20
    )
    app.state.redis = redis_client
    app.state.semantic_cache = SemanticCache(redis_client)

    # ── Local in-process Queue (no RabbitMQ broker needed) ────────────────────
    publisher = LocalQueuePublisher()
    await publisher.connect()
    app.state.publisher = publisher

    logger.info("All services initialised — OptiRAG is ready")
    yield

    # ── Shutdown ───────────────────────────────────────────────────────────
    logger.info("Shutting down OptiRAG...")
    await publisher.close()
    await redis_client.aclose()
    logger.info("Shutdown complete")


# ─── Application factory ──────────────────────────────────────────────────────

def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="OptiRAG",
        description=(
            "Multi-Modal RAG system with semantic caching.\n\n"
            "Supports text and image documents. Queries are processed via an in-process local queue, "
            "embedded with Ollama, retrieved from ChromaDB, and answered by a local LLM. "
            "Semantically similar queries are served instantly from a Redis cache."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # CORS (permissive for development; tighten in production)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.app_env == "development" else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount routes
    app.include_router(router, prefix="/api/v1")

    # Serve the UI at root
    @app.get("/", include_in_schema=False)
    async def root():
        from fastapi.responses import FileResponse
        import os
        ui_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "index.html")
        if os.path.exists(ui_path):
            return FileResponse(ui_path, media_type="text/html")
        return JSONResponse({"message": "OptiRAG API", "docs": "/docs", "health": "/api/v1/health"})

    return app


app = create_app()
