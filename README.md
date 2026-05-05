# Murtwell RAG – Planning Law Retrieval-Augmented Generation

This repo explores a minimal Retrieval-Augmented Generation (RAG) stack for UK planning law and related documents (national policy, local plans, appeal decisions). The goal is to be able to:

- Ask natural language questions about planning, and
- Get grounded answers with citations from real documents.

It is deliberately small and modular so it can later be:

- Reused for other domains (e.g. Miolingo content), and/or
- Wrapped in a "Murtwell Planning Network" frontend.

## Layout

```text
murtwell-rag/
  data/
    raw/           # original PDFs / HTML / text
      national/
      local/
      appeals/
    processed/     # optional cached text+chunks

  rag/
    __init__.py
    config.py      # paths, model names, etc.
    ingest.py      # build/update the vector index
    query.py       # retrieval + LLM call
    server.py      # optional: FastAPI HTTP API

  notebooks/
    01-ingest-playground.ipynb
    02-query-playground.ipynb
```

## High-level design

1. **Ingest**
   - Walk `data/raw/**` and extract text from PDFs/other sources.
   - Chunk text into manageable sizes (e.g. 300–800 words with overlap).
   - Embed each chunk using an embedding model (initially a hosted model like `text-embedding-3-large`).
   - Store embeddings + text + metadata in a vector store (initially ChromaDB).

2. **Query**
   - Encode a user's question with the same embedding model.
   - Retrieve the top-k most similar chunks from the vector store.
   - Build a prompt that:
     - Includes the retrieved chunks as "Context" with labels, and
     - Poses the user's question at the end.
   - Send that prompt to a hosted LLM (e.g. GPT-4.x) and return the answer.

3. **Serve (optional)**
   - Wrap `query.ask()` in a small FastAPI app in `server.py` to expose `/ask`.
   - Later, plug a lightweight web UI ("Murtwell Planning Network") or other clients into that endpoint.

## Public data sources (initial ideas)

The UK already exposes some relevant planning-related open data:

- **Planning Data (DLUHC)** – <https://www.planning.data.gov.uk/>
  - Central portal for open planning datasets in England.
  - Covers things like brownfield land, local planning authorities, some policy and designations.
  - Good candidate for structured data (e.g. site constraints); less about full-text law, but useful context.

- **Live tables on planning application statistics (GOV.UK)** – <https://www.gov.uk/government/statistical-data-sets/live-tables-on-planning-application-statistics>
  - Aggregate statistics on planning applications; more for analysis than text RAG, but useful for background.

- **Searchland Planning Applications API** – <https://searchland.co.uk/our-apis/planning-applications>
  - Commercial API providing planning decisions (approvals, refusals, appeals) across UK councils.
  - Not open data, but indicative of what a comprehensive planning-application dataset looks like.

- **PlanningResource Casebook** – <https://www.planningresource.co.uk/casebook>
  - Summaries of key decisions (appeals, court judgments, etc.).
  - Not necessarily open for bulk ingestion, but useful as a reference for which cases matter.

For **full-text legal and policy documents**, the initial corpus will likely come from:

- Statute and regulations (e.g. GPDO, primary legislation) via legislation.gov.uk.
- National Planning Policy Framework (NPPF) and associated guidance (GOV.UK).
- Local plan documents downloaded from local authority websites (PDF/HTML).
- Selected Planning Inspectorate decision letters (PDF/text) where terms allow.

Later, if needed, we can:

- Add scraping/harvesting scripts for specific councils.
- Integrate any future structured/open APIs that provide full-text decisions.

## Status

- Repo live on GitHub at `fairflow/murtwell-rag`.
- Core implementation in place: `rag/config.py`, `rag/ingest.py`, `rag/query.py`, `rag/server.py`.
- Notebooks ready under `notebooks/`.

### Next steps

1. Copy a `.env` from `.env.example` and add your `OPENAI_API_KEY`.
2. `pip install -r requirements.txt`
3. Drop documents (e.g. NPPF PDF, a local plan) into `data/raw/national/` or `data/raw/local/`.
4. Run ingest: `python -m rag.ingest`
5. Ask a question: `python -m rag.query "What does the NPPF say about green belt?"`
6. (Optional) Start the API: `uvicorn rag.server:app --reload`
