"""Process-wide lazy default retriever (Flask / CLI)."""

from __future__ import annotations

import logging
from typing import Any

from rag.config import faiss_retrieval_top_k
from rag.trace import rag_debug

logger = logging.getLogger(__name__)

_retriever: Any = None  # FaissRetriever | None, False = attempted load failed


def get_retriever():
    """Return cached FaissRetriever or None."""
    global _retriever
    if _retriever is False:
        return None
    if _retriever is not None:
        return _retriever
    from rag.retriever.faiss_retriever import FaissRetriever

    r = FaissRetriever.try_load_default_files()
    if r is None:
        _retriever = False
        return None
    _retriever = r
    return r


def retrieve_candidates(query: str, top_k: int | None = None) -> list[dict[str, Any]]:
    r = get_retriever()
    if r is None:
        rag_debug(logger, "retrieve: no index/meta (0 hits)")
        return []
    k = top_k if top_k is not None else faiss_retrieval_top_k()
    return r.retrieve_candidates(query, k)


def retrieve(query: str, top_k: int = 2) -> list[str]:
    return [c["text"] for c in retrieve_candidates(query, top_k=top_k)]
