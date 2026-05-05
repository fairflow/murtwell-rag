"""
rag/config.py – central configuration for paths, model names, and chunking.
All other modules import from here; override via environment variables or .env.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

# Sub-categories that mirror the README layout
RAW_NATIONAL = DATA_RAW / "national"
RAW_LOCAL = DATA_RAW / "local"
RAW_APPEALS = DATA_RAW / "appeals"

# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------
CHROMA_PERSIST_DIR = DATA_PROCESSED / "chroma"
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "planning_docs")

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # required at runtime
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))      # words
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))  # words

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K = int(os.getenv("TOP_K", "5"))
