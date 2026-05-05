"""
rag/ingest.py – build or update the ChromaDB vector index.

Usage (CLI):
    python -m rag.ingest                  # ingest everything under data/raw/
    python -m rag.ingest --path data/raw/national/nppf.pdf

The module is also importable:
    from rag.ingest import ingest_path, ingest_all
"""

from __future__ import annotations

import argparse
import hashlib
import io
import logging
import re
from pathlib import Path
from typing import Generator

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from tqdm import tqdm

from rag import config

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers – text extraction
# ---------------------------------------------------------------------------

def _extract_pdf(path: Path) -> str:
    """Return all text from a PDF file."""
    import pypdf

    reader = pypdf.PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def _extract_html(path: Path) -> str:
    """Return visible text from an HTML file."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_bytes(), "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def _extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(path)
    if suffix in {".html", ".htm"}:
        return _extract_html(path)
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="replace")
    raise ValueError(f"Unsupported file type: {suffix}")


# ---------------------------------------------------------------------------
# Helpers – chunking
# ---------------------------------------------------------------------------

def _chunk_text(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[str]:
    """Split *text* into word-count-bounded chunks with word-level overlap."""
    words = text.split()
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


# ---------------------------------------------------------------------------
# Helpers – document ID
# ---------------------------------------------------------------------------

def _doc_id(path: Path, chunk_index: int) -> str:
    """Stable, unique ID for a chunk derived from the file path and index."""
    stem = hashlib.md5(str(path).encode()).hexdigest()[:12]
    return f"{stem}-{chunk_index}"


# ---------------------------------------------------------------------------
# Helpers – metadata
# ---------------------------------------------------------------------------

_CATEGORY_MAP = {
    "national": "national",
    "local": "local",
    "appeals": "appeals",
}


def _category(path: Path) -> str:
    for part in path.parts:
        if part in _CATEGORY_MAP:
            return _CATEGORY_MAP[part]
    return "other"


# ---------------------------------------------------------------------------
# Core ingest
# ---------------------------------------------------------------------------

def _get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(config.CHROMA_PERSIST_DIR))
    ef = OpenAIEmbeddingFunction(
        api_key=config.OPENAI_API_KEY,
        model_name=config.EMBEDDING_MODEL,
    )
    return client.get_or_create_collection(
        name=config.CHROMA_COLLECTION,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def ingest_path(path: Path, collection: chromadb.Collection | None = None) -> int:
    """Ingest a single file. Returns the number of chunks added."""
    if collection is None:
        collection = _get_collection()

    log.info("Extracting text from %s", path)
    try:
        text = _extract_text(path)
    except Exception as exc:
        log.warning("Skipping %s: %s", path, exc)
        return 0

    chunks = _chunk_text(text)
    if not chunks:
        return 0

    ids = [_doc_id(path, i) for i in range(len(chunks))]
    metadatas = [
        {
            "source": str(path),
            "filename": path.name,
            "category": _category(path),
            "chunk_index": i,
            "total_chunks": len(chunks),
        }
        for i in range(len(chunks))
    ]

    # Upsert so re-running is idempotent.
    collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)
    log.info("Upserted %d chunks from %s", len(chunks), path)
    return len(chunks)


def ingest_all(root: Path = config.DATA_RAW) -> int:
    """Walk *root* and ingest every supported file. Returns total chunks added."""
    supported = {".pdf", ".html", ".htm", ".txt", ".md"}
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in supported]

    if not files:
        log.warning("No supported files found under %s", root)
        return 0

    collection = _get_collection()
    total = 0
    for path in tqdm(files, desc="Ingesting documents"):
        total += ingest_path(path, collection)

    log.info("Ingest complete. Total chunks: %d", total)
    return total


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Ingest documents into the vector store.")
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Path to a specific file or directory (default: data/raw/)",
    )
    args = parser.parse_args()

    target: Path = args.path or config.DATA_RAW
    if target.is_file():
        ingest_path(target)
    else:
        ingest_all(target)
