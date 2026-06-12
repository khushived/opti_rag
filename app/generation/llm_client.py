"""
app/generation/llm_client.py
─────────────────────────────
Ollama LLM client supporting:
  - Text-only generation (mistral)
  - Multi-modal generation with vision (llava) when images are present

Both endpoints use Ollama's /api/chat interface for a consistent message format.
Streaming is supported for real-time token delivery.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import get_settings
from app.core.logging import get_logger
from app.ingestion.document_loader import image_to_base64

logger = get_logger(__name__)

_RETRY_POLICY = dict(
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    stop=stop_after_attempt(3),
    reraise=True,
)


@dataclass
class GenerationResult:
    """Structured output from an LLM generation call."""

    answer: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    is_multimodal: bool = False


def _build_context_prompt(query: str, context_chunks: list[dict]) -> str:
    """Assemble a RAG prompt from retrieved chunks."""
    settings = get_settings()
    context_parts = [
        f"[Source {i + 1}: {c['metadata'].get('filename', c['metadata'].get('source', '?'))} "
        f"(page {c['metadata'].get('page', '?')}, score {c['score']:.3f})]\n{c['text']}"
        for i, c in enumerate(context_chunks)
        if c.get("text", "").strip()
    ]

    context_block = "\n\n---\n\n".join(context_parts) if context_parts else "No context available."

    return (
        f"{settings.system_prompt}\n\n"
        f"## Context\n\n{context_block}\n\n"
        f"## Question\n\n{query}\n\n"
        f"## Answer"
    )


# ─── Text-only generation ─────────────────────────────────────────────────────

async def generate_text(
    query: str,
    context_chunks: list[dict],
) -> GenerationResult:
    """
    Generate a text response from ``mistral`` using retrieved context.
    Falls back to a mock answer if Ollama is not running.
    """
    settings = get_settings()
    prompt = _build_context_prompt(query, context_chunks)

    logger.info("Generating text response", model=settings.ollama_text_model)

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                f"{settings.ollama_base_url}/api/chat",
                json={
                    "model": settings.ollama_text_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "options": {
                        "temperature": settings.temperature,
                        "num_predict": settings.max_new_tokens,
                    },
                    "stream": False,
                },
            )
            response.raise_for_status()
            data = response.json()

        answer = data["message"]["content"]
        usage = data.get("usage", {})
        logger.info("Text generation complete", tokens=usage.get("completion_tokens", 0))
        return GenerationResult(
            answer=answer,
            model=settings.ollama_text_model,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            is_multimodal=False,
        )

    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as exc:
        logger.warning("Ollama not available, using mock answer", error=str(exc))
        # Build a helpful mock answer from the retrieved context
        if context_chunks:
            context_text = "\n\n".join(
                f"• {c['text'][:300]}" for c in context_chunks[:3] if c.get("text")
            )
            mock_answer = (
                f"[Demo mode — Ollama not running]\n\n"
                f"Based on the retrieved context, here is what I found:\n\n"
                f"{context_text}\n\n"
                f"To get AI-generated answers, install Ollama from https://ollama.com "
                f"and run: ollama pull mistral"
            )
        else:
            mock_answer = (
                "[Demo mode — Ollama not running]\n\n"
                "No documents have been indexed yet. "
                "Upload a document via POST /api/v1/ingest first, then ask questions.\n\n"
                "To get AI-generated answers, install Ollama from https://ollama.com "
                "and run: ollama pull mistral"
            )
        return GenerationResult(
            answer=mock_answer,
            model="mock (ollama-unavailable)",
            is_multimodal=False,
        )


# ─── Multi-modal generation ───────────────────────────────────────────────────

@retry(**_RETRY_POLICY)
async def generate_multimodal(
    query: str,
    context_chunks: list[dict],
    images: list[bytes],
) -> GenerationResult:
    """
    Generate a response from ``llava`` with text context AND images.

    Args:
        query:          User's question.
        context_chunks: Retrieved text hit dicts.
        images:         List of raw PNG bytes to include in the prompt.

    Returns:
        ``GenerationResult`` with the LLM answer.
    """
    settings = get_settings()
    prompt = _build_context_prompt(query, context_chunks)
    b64_images = [image_to_base64(img) for img in images[:4]]  # cap at 4 images

    logger.info(
        "Generating multimodal response",
        model=settings.ollama_vision_model,
        num_images=len(b64_images),
    )

    async with httpx.AsyncClient(timeout=settings.ollama_timeout) as client:
        response = await client.post(
            f"{settings.ollama_base_url}/api/chat",
            json={
                "model": settings.ollama_vision_model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": b64_images,
                    }
                ],
                "options": {
                    "temperature": settings.temperature,
                    "num_predict": settings.max_new_tokens,
                },
                "stream": False,
            },
        )
        response.raise_for_status()
        data = response.json()

    answer = data["message"]["content"]
    usage = data.get("usage", {})

    logger.info("Multimodal generation complete", tokens=usage.get("completion_tokens", 0))
    return GenerationResult(
        answer=answer,
        model=settings.ollama_vision_model,
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        is_multimodal=True,
    )


# ─── Streaming variant ────────────────────────────────────────────────────────

async def generate_text_stream(
    query: str,
    context_chunks: list[dict],
) -> AsyncIterator[str]:
    """
    Stream text tokens from Ollama (yields partial strings).

    Usage::

        async for token in generate_text_stream(query, chunks):
            print(token, end="", flush=True)
    """
    settings = get_settings()
    prompt = _build_context_prompt(query, context_chunks)

    async with httpx.AsyncClient(timeout=settings.ollama_timeout) as client:
        async with client.stream(
            "POST",
            f"{settings.ollama_base_url}/api/chat",
            json={
                "model": settings.ollama_text_model,
                "messages": [{"role": "user", "content": prompt}],
                "options": {"temperature": settings.temperature},
                "stream": True,
            },
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token
                if chunk.get("done", False):
                    break


# ─── Routing helper ───────────────────────────────────────────────────────────

async def generate(
    query: str,
    context_chunks: list[dict],
    images: Optional[list[bytes]] = None,
) -> GenerationResult:
    """
    Route to multi-modal or text-only generation based on image presence.

    Args:
        query:          User's question.
        context_chunks: Retrieved text context.
        images:         Raw PNG bytes (if any) from retrieved image chunks.

    Returns:
        ``GenerationResult``.
    """
    if images:
        return await generate_multimodal(query, context_chunks, images)
    return await generate_text(query, context_chunks)
