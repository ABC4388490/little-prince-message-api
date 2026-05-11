"""Paths and tunables for the RAG stack (env + defaults)."""

from __future__ import annotations

import os

# Package directory (…/message-api/rag)
RAG_DIR = os.path.dirname(os.path.abspath(__file__))

INDEX_PATH = os.path.join(RAG_DIR, "index.faiss")
CHUNK_META_PATH = os.path.join(RAG_DIR, "chunk_meta.json")
CHUNK_ORDER_PATH = os.path.join(RAG_DIR, "chunk_order.json")
DATA_PATH = os.path.join(RAG_DIR, "data.json")

BI_ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_FAISS_TOP_K = 10
DEFAULT_RERANK_FINAL_K = 3


def faiss_retrieval_top_k() -> int:
    raw = os.environ.get("RAG_FAISS_TOP_K", "").strip()
    if not raw:
        return DEFAULT_FAISS_TOP_K
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_FAISS_TOP_K


def rerank_output_top_k() -> int:
    raw = os.environ.get("RAG_FINAL_TOP_K", "").strip()
    if not raw:
        return DEFAULT_RERANK_FINAL_K
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_RERANK_FINAL_K
