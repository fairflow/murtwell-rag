"""
rag/query.py – retrieval + LLM call.

Usage (CLI):
    python -m rag.query "What does the NPPF say about green belt?"

Importable:
    from rag.query import ask
    result = ask("What does the NPPF say about green belt?")
    print(result.answer)
    for c in result.citations:
        print(c)
"""

from __future__ import annotations

import argparse
import logging
import textwrap
from dataclasses import dataclass, field

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI

from rag import config

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Citation:
    source: str
    category: str
    chunk_index: int
    excerpt: str  # first ~200 chars of the chunk


@dataclass
class Answer:
    question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are an expert in UK planning law and policy. "
    "Answer the user's question using ONLY the context passages provided. "
    "If the answer is not contained in the context, say so clearly. "
    "Where relevant, cite the passage label (e.g. [1]) in your answer."
)

_USER_TEMPLATE = """\
Context passages:
{context}

Question: {question}
"""


def _build_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk["metadata"].get("filename", "unknown")
        category = chunk["metadata"].get("category", "")
        parts.append(f"[{i}] ({category} / {source})\n{chunk['document']}")
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Core retrieval + generation
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


def ask(
    question: str,
    top_k: int = config.TOP_K,
    category_filter: str | None = None,
) -> Answer:
    """
    Retrieve relevant chunks and generate a grounded answer.

    Args:
        question: Natural language question.
        top_k: Number of chunks to retrieve.
        category_filter: Optional – restrict retrieval to 'national', 'local', or 'appeals'.
    """
    collection = _get_collection()

    where: dict | None = None
    if category_filter:
        where = {"category": {"$eq": category_filter}}

    results = collection.query(
        query_texts=[question],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    if not docs:
        return Answer(
            question=question,
            answer="No relevant documents were found in the index. Please ingest some documents first.",
        )

    chunks = [{"document": d, "metadata": m} for d, m in zip(docs, metas)]
    context_str = _build_context(chunks)

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": _USER_TEMPLATE.format(
                    context=context_str, question=question
                ),
            },
        ],
    )

    answer_text = response.choices[0].message.content or ""

    citations = [
        Citation(
            source=m.get("source", ""),
            category=m.get("category", ""),
            chunk_index=m.get("chunk_index", 0),
            excerpt=textwrap.shorten(d, width=200, placeholder="…"),
        )
        for d, m in zip(docs, metas)
    ]

    return Answer(question=question, answer=answer_text, citations=citations)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Query the planning RAG system.")
    parser.add_argument("question", nargs="+", help="Question to ask")
    parser.add_argument(
        "--category",
        choices=["national", "local", "appeals"],
        default=None,
        help="Restrict retrieval to a document category",
    )
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    args = parser.parse_args()

    q = " ".join(args.question)
    result = ask(q, top_k=args.top_k, category_filter=args.category)

    print(f"\n{'='*60}")
    print(f"Q: {result.question}")
    print(f"{'='*60}")
    print(result.answer)
    print(f"\n--- Sources ---")
    for i, c in enumerate(result.citations, start=1):
        print(f"[{i}] {c.category} / {c.source} (chunk {c.chunk_index})")
        print(f"    {c.excerpt}")
