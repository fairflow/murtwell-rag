"""
rag/server.py – thin FastAPI wrapper around query.ask().

Run:
    uvicorn rag.server:app --reload

Endpoints:
    POST /ask   { "question": "...", "top_k": 5, "category": null }
    GET  /health
"""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag.query import Answer, Citation, ask

app = FastAPI(
    title="Murtwell RAG – Planning Law",
    description="Retrieval-Augmented Generation over UK planning law and policy documents.",
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(..., min_length=5, description="Natural language question")
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve")
    category: Literal["national", "local", "appeals"] | None = Field(
        None, description="Restrict retrieval to a document category"
    )


class CitationOut(BaseModel):
    source: str
    category: str
    chunk_index: int
    excerpt: str


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[CitationOut]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest) -> AskResponse:
    try:
        result: Answer = ask(
            question=req.question,
            top_k=req.top_k,
            category_filter=req.category,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AskResponse(
        question=result.question,
        answer=result.answer,
        citations=[
            CitationOut(
                source=c.source,
                category=c.category,
                chunk_index=c.chunk_index,
                excerpt=c.excerpt,
            )
            for c in result.citations
        ],
    )
