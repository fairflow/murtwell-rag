"""
tests/test_ingest.py

Unit tests for rag/ingest.py.
All tests are offline – no OpenAI calls are made.
A deterministic fake embedding function is used in place of OpenAIEmbeddingFunction.
"""

import hashlib
import uuid
from pathlib import Path

import chromadb
import pytest

from rag.ingest import _category, _chunk_text, _doc_id, ingest_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeEF:
    """8-dimensional deterministic embedding – no network calls."""

    DIM = 8

    def name(self) -> str:  # required by newer ChromaDB
        return "fake-ef"

    def _encode(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            h = int(hashlib.md5(text.encode()).hexdigest(), 16)
            vec = [((h >> (i * 4)) & 0xF) / 15.0 for i in range(self.DIM)]
            out.append(vec)
        return out

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self._encode(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return self._encode(input)

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self._encode(input)


def _mem_collection() -> chromadb.Collection:
    # Unique name per call: EphemeralClient shares state within a process.
    client = chromadb.EphemeralClient()
    return client.create_collection(uuid.uuid4().hex, embedding_function=_FakeEF())


# ---------------------------------------------------------------------------
# _chunk_text
# ---------------------------------------------------------------------------

def test_chunk_text_single_chunk():
    text = "word " * 100  # 100 words, well within default 600
    chunks = _chunk_text(text, chunk_size=600, overlap=50)
    assert len(chunks) == 1


def test_chunk_text_multiple_chunks():
    text = " ".join(str(i) for i in range(1000))  # 1000 words
    chunks = _chunk_text(text, chunk_size=300, overlap=50)
    assert len(chunks) > 1


def test_chunk_text_overlap():
    words = [str(i) for i in range(700)]
    text = " ".join(words)
    chunks = _chunk_text(text, chunk_size=300, overlap=50)
    # The tail of chunk 0 and the head of chunk 1 should share words
    end_of_first = set(chunks[0].split()[-50:])
    start_of_second = set(chunks[1].split()[:50])
    assert end_of_first & start_of_second  # non-empty intersection


def test_chunk_text_empty():
    assert _chunk_text("") == []


def test_chunk_text_preserves_all_words():
    # Total unique words across all chunks should equal the original word count.
    # (With overlap, words appear more than once, but all originals must appear.)
    words = [str(i) for i in range(500)]
    text = " ".join(words)
    chunks = _chunk_text(text, chunk_size=200, overlap=40)
    all_words_in_chunks = " ".join(chunks).split()
    assert set(words) == set(all_words_in_chunks)


# ---------------------------------------------------------------------------
# _doc_id
# ---------------------------------------------------------------------------

def test_doc_id_stable():
    p = Path("/some/file.pdf")
    assert _doc_id(p, 0) == _doc_id(p, 0)


def test_doc_id_chunk_index_distinguishes():
    p = Path("/some/file.pdf")
    assert _doc_id(p, 0) != _doc_id(p, 1)


def test_doc_id_different_files():
    assert _doc_id(Path("/a.pdf"), 0) != _doc_id(Path("/b.pdf"), 0)


# ---------------------------------------------------------------------------
# _category
# ---------------------------------------------------------------------------

def test_category_national():
    assert _category(Path("/data/raw/national/nppf.pdf")) == "national"


def test_category_local():
    assert _category(Path("/data/raw/local/southwark.pdf")) == "local"


def test_category_appeals():
    assert _category(Path("/data/raw/appeals/abc123.pdf")) == "appeals"


def test_category_unknown():
    assert _category(Path("/somewhere/else/doc.pdf")) == "other"


# ---------------------------------------------------------------------------
# ingest_path
# ---------------------------------------------------------------------------

def test_ingest_path_txt(tmp_path):
    doc = tmp_path / "national" / "test.txt"
    doc.parent.mkdir()
    doc.write_text("This is a test planning document. " * 20)

    col = _mem_collection()
    n = ingest_path(doc, collection=col)

    assert n >= 1
    assert col.count() == n


def test_ingest_path_idempotent(tmp_path):
    doc = tmp_path / "national" / "test.txt"
    doc.parent.mkdir()
    doc.write_text("A planning document about green belt. " * 20)

    col = _mem_collection()
    n1 = ingest_path(doc, collection=col)
    n2 = ingest_path(doc, collection=col)  # second run: upsert, not duplicate

    assert n1 == n2
    assert col.count() == n1


def test_ingest_path_unsupported_extension(tmp_path):
    doc = tmp_path / "file.xyz"
    doc.write_text("ignored")

    col = _mem_collection()
    n = ingest_path(doc, collection=col)
    assert n == 0
    assert col.count() == 0


def test_ingest_path_metadata(tmp_path):
    doc = tmp_path / "appeals" / "decision.txt"
    doc.parent.mkdir()
    doc.write_text("Appeal decision regarding green belt development. " * 20)

    col = _mem_collection()
    ingest_path(doc, collection=col)

    results = col.get(include=["metadatas"])
    for meta in results["metadatas"]:
        assert meta["category"] == "appeals"
        assert meta["filename"] == "decision.txt"
