"""
Canonical chunk metadata for the RAG pipeline (storage, retrieval, rerank, LLM, API).

Required for indexing / retrieval:
  - chunk_id: stable id (e.g. passage_21_02, char_fox_01)
  - source_name: logical source / corpus label (e.g. little_prince_kb)
  - text: passage body used for embedding (often includes chapter/topic/role keywords)

Extended fields (optional, preserved when present):
  - content: 【原文】+【释义】 or analysis body
  - chapter: chapter title or 跨章 / 全书
  - topic: list of theme tags (e.g. 爱, 孤独)
  - role: list of characters
  - chunk_type: passage | character | theme

Legacy JSON may use `source` instead of `source_name`; readers normalize both.
"""

from __future__ import annotations

from typing import Any

DEFAULT_SOURCE_NAME = "little_prince_kb"


def default_source_name_from_payload(payload: dict) -> str:
    v = payload.get("default_source_name") or payload.get("default_source")
    if v is None or str(v).strip() == "":
        return DEFAULT_SOURCE_NAME
    return str(v).strip()


def source_name_from_dict(row: dict, *, default: str = DEFAULT_SOURCE_NAME) -> str:
    """Resolve source_name from a chunk row or candidate dict (supports legacy `source`)."""
    for key in ("source_name", "source"):
        v = row.get(key)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return default


def ensure_chunk_metadata(d: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy with chunk_id and source_name guaranteed (for API / logging)."""
    out = dict(d)
    cid = str(out.get("chunk_id", "") or "").strip()
    out["chunk_id"] = cid
    out["source_name"] = source_name_from_dict(out)
    return out


def passage_body(d: dict[str, Any]) -> str:
    """Text shown to the LLM: prefer content, else text."""
    content = str(d.get("content", "") or "").strip()
    if content:
        return content
    return str(d.get("text", "") or "").strip()
