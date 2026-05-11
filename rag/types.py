"""Shared typing for RAG stages (test-friendly dict shapes)."""

from __future__ import annotations

from typing import Any, TypedDict


class ChunkRecord(TypedDict, total=False):
    chunk_id: str
    source_name: str
    text: str


class SearchHit(TypedDict, total=False):
    row_index: int
    chunk_id: str
    source_name: str
    text: str
    score: float
    rerank_score: float


# Plain dicts flow through retriever → reranker; SearchHit documents expected keys.
ChunkDict = dict[str, Any]
