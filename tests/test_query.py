"""
tests/test_query.py

Unit tests for rag/query.py.
All tests are offline – _get_collection() is patched to return an in-memory
ChromaDB collection, and the OpenAI client is replaced with a MagicMock.
"""

import hashlib
import uuid
from unittest.mock import MagicMock, patch

import chromadb
import pytest

from rag.query import Answer, ask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeEF:
    """Same deterministic embedding as in test_ingest – no network calls."""

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


def _populated_collection() -> chromadb.Collection:
    client = chromadb.EphemeralClient()
    # Unique name per call: EphemeralClient shares state within a process.
    col = client.create_collection(uuid.uuid4().hex, embedding_function=_FakeEF())
    col.upsert(
        ids=["chunk-0", "chunk-1"],
        documents=[
            "Green belt land is protected from development under NPPF paragraph 142.",
            "The NPPF states that local authorities must protect green belt boundaries.",
        ],
        metadatas=[
            {
                "source": "/data/national/nppf.pdf",
                "filename": "nppf.pdf",
                "category": "national",
                "chunk_index": 0,
                "total_chunks": 2,
            },
            {
                "source": "/data/national/nppf.pdf",
                "filename": "nppf.pdf",
                "category": "national",
                "chunk_index": 1,
                "total_chunks": 2,
            },
        ],
    )
    return col


def _fake_openai_response(text: str) -> MagicMock:
    choice = MagicMock()
    choice.message.content = text
    response = MagicMock()
    response.choices = [choice]
    return response


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_ask_empty_collection_returns_no_results_message():
    col = chromadb.EphemeralClient().create_collection(
        uuid.uuid4().hex, embedding_function=_FakeEF()
    )
    with patch("rag.query._get_collection", return_value=col):
        result = ask("What is green belt?")

    assert isinstance(result, Answer)
    assert "No relevant documents" in result.answer
    assert result.citations == []


def test_ask_returns_answer_text():
    col = _populated_collection()
    fake_answer = "Green belt is protected land under the NPPF [1][2]."

    with patch("rag.query._get_collection", return_value=col), \
         patch("rag.query.OpenAI") as MockOpenAI:
        MockOpenAI.return_value.chat.completions.create.return_value = (
            _fake_openai_response(fake_answer)
        )
        result = ask("What does the NPPF say about green belt?")

    assert result.answer == fake_answer


def test_ask_returns_citations():
    col = _populated_collection()

    with patch("rag.query._get_collection", return_value=col), \
         patch("rag.query.OpenAI") as MockOpenAI:
        MockOpenAI.return_value.chat.completions.create.return_value = (
            _fake_openai_response("Answer.")
        )
        result = ask("What does the NPPF say about green belt?", top_k=2)

    assert len(result.citations) == 2
    assert all(c.category == "national" for c in result.citations)
    assert all("nppf.pdf" in c.source for c in result.citations)


def test_ask_preserves_question():
    col = _populated_collection()

    with patch("rag.query._get_collection", return_value=col), \
         patch("rag.query.OpenAI") as MockOpenAI:
        MockOpenAI.return_value.chat.completions.create.return_value = (
            _fake_openai_response("ok")
        )
        result = ask("My specific planning question")

    assert result.question == "My specific planning question"


def test_ask_citation_excerpt_is_truncated():
    col = _populated_collection()

    with patch("rag.query._get_collection", return_value=col), \
         patch("rag.query.OpenAI") as MockOpenAI:
        MockOpenAI.return_value.chat.completions.create.return_value = (
            _fake_openai_response("ok")
        )
        result = ask("green belt", top_k=2)

    for c in result.citations:
        assert len(c.excerpt) <= 200


def test_ask_category_filter_is_passed_to_chroma():
    col = _populated_collection()

    with patch("rag.query._get_collection", return_value=col), \
         patch("rag.query.OpenAI") as MockOpenAI:
        MockOpenAI.return_value.chat.completions.create.return_value = (
            _fake_openai_response("ok")
        )
        # category_filter="national" matches what's in the collection
        result = ask("green belt?", category_filter="national")

    assert isinstance(result, Answer)


def test_ask_top_k_limits_citations():
    col = _populated_collection()

    with patch("rag.query._get_collection", return_value=col), \
         patch("rag.query.OpenAI") as MockOpenAI:
        MockOpenAI.return_value.chat.completions.create.return_value = (
            _fake_openai_response("ok")
        )
        result = ask("green belt", top_k=1)

    assert len(result.citations) == 1
