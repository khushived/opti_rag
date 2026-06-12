"""
app/ingestion/document_loader.py
─────────────────────────────────
Multi-modal document loader supporting:
  - PDF  → text pages + embedded images (via PyMuPDF)
  - Images (PNG / JPG / WEBP) → raw bytes + optional OCR placeholder
  - Plain text / Markdown → raw text
  - DOCX → paragraphs

Each loader returns a list of ``RawDocument`` objects containing text content
and optional image bytes for downstream processing.
"""

from __future__ import annotations

import base64
import io
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from PIL import Image

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RawDocument:
    """A single logical unit extracted from a source file."""

    source: str                             # file path or URL
    content_type: str                       # "text" | "image" | "mixed"
    text: str = ""                          # extracted / associated text
    images: list[bytes] = field(default_factory=list)  # raw PNG bytes
    metadata: dict = field(default_factory=dict)


# ─── PDF Loader ───────────────────────────────────────────────────────────────

def _extract_images_from_page(page: fitz.Page) -> list[bytes]:
    """Extract all embedded raster images from a PDF page as PNG bytes."""
    doc = page.parent
    image_list = page.get_images(full=True)
    png_images: list[bytes] = []

    for img_info in image_list:
        xref = img_info[0]
        try:
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]
            img_ext = base_image.get("ext", "png")

            # Normalise to PNG via Pillow
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            png_images.append(buf.getvalue())
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to extract image from PDF", xref=xref, error=str(exc))

    return png_images


def load_pdf(path: Path) -> list[RawDocument]:
    """Load a PDF file, returning one RawDocument per page."""
    docs: list[RawDocument] = []
    logger.info("Loading PDF", path=str(path))

    with fitz.open(str(path)) as pdf:
        total_pages = pdf.page_count
        for page_num in range(total_pages):
            page = pdf[page_num]
            text = page.get_text("text").strip()
            images = _extract_images_from_page(page)

            content_type = "mixed" if (text and images) else ("image" if images else "text")
            docs.append(
                RawDocument(
                    source=str(path),
                    content_type=content_type,
                    text=text,
                    images=images,
                    metadata={
                        "page": page_num + 1,
                        "total_pages": total_pages,
                        "filename": path.name,
                    },
                )
            )

    logger.info("PDF loaded", path=str(path), pages=len(docs))
    return docs


# ─── Image Loader ─────────────────────────────────────────────────────────────

def load_image(path: Path) -> list[RawDocument]:
    """Load a standalone image file."""
    logger.info("Loading image", path=str(path))
    with open(path, "rb") as f:
        raw = f.read()

    # Normalise to PNG
    buf = io.BytesIO()
    Image.open(io.BytesIO(raw)).convert("RGB").save(buf, format="PNG")

    return [
        RawDocument(
            source=str(path),
            content_type="image",
            text="",
            images=[buf.getvalue()],
            metadata={"filename": path.name},
        )
    ]


# ─── Text / Markdown Loader ───────────────────────────────────────────────────

def load_text(path: Path) -> list[RawDocument]:
    """Load a plain text or markdown file."""
    logger.info("Loading text file", path=str(path))
    text = path.read_text(encoding="utf-8", errors="replace")
    return [
        RawDocument(
            source=str(path),
            content_type="text",
            text=text,
            metadata={"filename": path.name},
        )
    ]


# ─── DOCX Loader ──────────────────────────────────────────────────────────────

def load_docx(path: Path) -> list[RawDocument]:
    """Load a DOCX file, returning one RawDocument per paragraph group."""
    try:
        from docx import Document  # type: ignore[import-untyped]
    except ImportError:
        logger.error("python-docx not installed; cannot load DOCX")
        return []

    logger.info("Loading DOCX", path=str(path))
    doc = Document(str(path))
    full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [
        RawDocument(
            source=str(path),
            content_type="text",
            text=full_text,
            metadata={"filename": path.name},
        )
    ]


# ─── Dispatcher ───────────────────────────────────────────────────────────────

_LOADERS = {
    ".pdf": load_pdf,
    ".png": load_image,
    ".jpg": load_image,
    ".jpeg": load_image,
    ".webp": load_image,
    ".txt": load_text,
    ".md": load_text,
    ".markdown": load_text,
    ".docx": load_docx,
}


def load_document(path: str | Path) -> list[RawDocument]:
    """
    Auto-detect file type and load accordingly.

    Args:
        path: Absolute or relative path to the document.

    Returns:
        List of ``RawDocument`` instances.

    Raises:
        ValueError: If the file extension is not supported.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Document not found: {p}")

    ext = p.suffix.lower()
    loader = _LOADERS.get(ext)
    if loader is None:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {sorted(_LOADERS.keys())}"
        )

    return loader(p)


def image_to_base64(image_bytes: bytes) -> str:
    """Encode raw PNG bytes to a base64 string (for Ollama vision API)."""
    return base64.b64encode(image_bytes).decode("utf-8")
