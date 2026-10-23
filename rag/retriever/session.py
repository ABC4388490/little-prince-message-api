"""Process-wide lazy default retriever (Flask / CLI)."""

from __future__ import annotations

import logging
import json
import os
import re
from typing import Any

from rag.config import CHUNK_META_PATH, faiss_retrieval_top_k
from rag.chunk_meta import ensure_chunk_metadata
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
    if os.environ.get("RAG_RETRIEVAL_MODE", "").strip().lower() == "lexical":
        return lexical_candidates(query, top_k)
    r = get_retriever()
    if r is None:
        rag_debug(logger, "retrieve: no index/meta (0 hits)")
        return []
    k = top_k if top_k is not None else faiss_retrieval_top_k()
    return r.retrieve_candidates(query, k)


def lexical_candidates(query: str, top_k: int | None = None) -> list[dict[str, Any]]:
    """Retrieve the same cited corpus without loading a local embedding model."""
    with open(CHUNK_META_PATH, encoding="utf-8") as source:
        rows = json.load(source)
    return rank_lexical_rows(query, rows, top_k)


def rank_lexical_rows(query: str, rows: list[dict[str, Any]], top_k: int | None = None) -> list[dict[str, Any]]:
    """Rank supplied rows without changing the corpus or loading a model."""
    parts = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", str(query).lower())
    tokens = set()
    for part in parts:
        if re.fullmatch(r"[a-z0-9_]+", part):
            tokens.add(part)
        else:
            for size in (2, 3, 4):
                tokens.update(part[i:i + size] for i in range(len(part) - size + 1))
    if not tokens:
        return []
    scored = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        original_text = str(row.get("text") or row.get("content") or "")
        text = original_text.lower()
        title = str(row.get("chapter") or "").lower()
        score = sum(len(token) * (int(token in text) + 3 * int(token in title)) for token in tokens)
        if score > 0:
            scored.append((score, ensure_chunk_metadata({**row, "text": original_text})))
    scored.sort(key=lambda item: item[0], reverse=True)
    limit = max(1, top_k if top_k is not None else faiss_retrieval_top_k())
    return [row for _, row in scored[:limit]]


def retrieve(query: str, top_k: int = 2) -> list[str]:
    return [c["text"] for c in retrieve_candidates(query, top_k=top_k)]
