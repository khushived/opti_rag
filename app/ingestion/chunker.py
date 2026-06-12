"""
app/ingestion/chunker.py
────────────────────────
Splits RawDocuments into smaller, overlapping text chunks suitable for
embedding and storage in the vector store.

Strategy:
  - Word-level sliding window (respects ``chunk_size`` and ``chunk_overlap``
    settings, treating words as atomic units for simplicity).
  - Image-bearing pages produce a chunk per image with surrounding text as context.
  - Each chunk carries forward all source metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from app.core.config import get_settings
from app.core.logging import get_logger
from app.ingestion.document_loader import RawDocument

logger = get_logger(__name__)


@dataclass
class Chunk:
    """A text chunk ready for embedding and vector storage."""

    chunk_id: str                              # "<source>:<page>:<index>"
    source: str                                # original file path
    text: str                                  # chunk text content
    images: list[bytes] = field(default_factory=list)  # associated images (may be empty)
    metadata: dict = field(default_factory=dict)


def _split_words(text: str, size: int, overlap: int) -> list[str]:
    """
    Split *text* into overlapping windows of approximately *size* words.

    Args:
        text:    Input text.
        size:    Target window size in words.
        overlap: Number of words shared between consecutive windows.

    Returns:
        List of text strings.
    """
    words = text.split()
    if not words:
        return []

    step = max(1, size - overlap)
    windows: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        windows.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += step

    return windows


def chunk_documents(documents: list[RawDocument]) -> list[Chunk]:
    """
    Convert a list of ``RawDocument`` objects into embedding-ready ``Chunk``s.

    Rules:
    - Pure text docs → sliding-window text chunks (no images).
    - Pure image docs → one chunk per image with empty text.
    - Mixed docs     → one chunk per image; text splits independently and
                       images are attached to the first text chunk on the page.

    Args:
        documents: Output of ``load_document()``.

    Returns:
        List of ``Chunk`` objects.
    """
    settings = get_settings()
    chunk_size = settings.chunk_size
    chunk_overlap = settings.chunk_overlap

    chunks: list[Chunk] = []

    for doc in documents:
        page = doc.metadata.get("page", 1)
        source = doc.source

        if doc.content_type == "text":
            # ── Text-only path ─────────────────────────────────────────────────
            windows = _split_words(doc.text, chunk_size, chunk_overlap)
            for idx, window in enumerate(windows):
                chunks.append(
                    Chunk(
                        chunk_id=f"{source}:p{page}:c{idx}",
                        source=source,
                        text=window,
                        images=[],
                        metadata={**doc.metadata, "chunk_index": idx},
                    )
                )

        elif doc.content_type == "image":
            # ── Image-only path ────────────────────────────────────────────────
            for idx, img_bytes in enumerate(doc.images):
                chunks.append(
                    Chunk(
                        chunk_id=f"{source}:p{page}:img{idx}",
                        source=source,
                        text="",          # no textual content
                        images=[img_bytes],
                        metadata={**doc.metadata, "image_index": idx, "is_image": True},
                    )
                )

        else:
            # ── Mixed path (text + images on same page) ────────────────────────
            # Text windows
            windows = _split_words(doc.text, chunk_size, chunk_overlap)
            for idx, window in enumerate(windows):
                # Attach all images to the first text window on the page
                imgs = doc.images if idx == 0 else []
                chunks.append(
                    Chunk(
                        chunk_id=f"{source}:p{page}:c{idx}",
                        source=source,
                        text=window,
                        images=imgs,
                        metadata={**doc.metadata, "chunk_index": idx},
                    )
                )

            # If there are images and no text at all, emit image-only chunks
            if not windows:
                for idx, img_bytes in enumerate(doc.images):
                    chunks.append(
                        Chunk(
                            chunk_id=f"{source}:p{page}:img{idx}",
                            source=source,
                            text="",
                            images=[img_bytes],
                            metadata={
                                **doc.metadata,
                                "image_index": idx,
                                "is_image": True,
                            },
                        )
                    )

    logger.info(
        "Chunking complete",
        input_docs=len(documents),
        output_chunks=len(chunks),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return chunks
