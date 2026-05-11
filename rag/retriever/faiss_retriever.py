"""
FAISS retriever: bi-encoder query embedding + IP search + chunk metadata join.

Stateless class is easy to test with a tiny in-memory store and fake meta.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

import faiss
import numpy as np

from rag.chunk_meta import DEFAULT_SOURCE_NAME, source_name_from_dict
from rag.config import CHUNK_META_PATH, CHUNK_ORDER_PATH, INDEX_PATH
from rag.embedding.bi_encoder import SentenceBiEncoder
from rag.trace import rag_debug
from rag.vectorstore.faiss_ip import FaissIPVectorStore

logger = logging.getLogger(__name__)


def _load_meta_rows() -> Optional[list[dict[str, str]]]:
    if os.path.isfile(CHUNK_META_PATH):
        try:
            with open(CHUNK_META_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if not isinstance(raw, list) or not raw:
                return None
            out: list[dict[str, str]] = []
            for i, row in enumerate(raw):
                if isinstance(row, dict):
                    t = str(row.get("text", "")).strip()
                    if not t:
                        continue
                    cid = str(row.get("chunk_id") or f"kb_{i:03d}").strip()
                    src = source_name_from_dict(row, default=DEFAULT_SOURCE_NAME)
                    out.append({"chunk_id": cid, "source_name": src, "text": t})
                elif isinstance(row, str) and row.strip():
                    out.append(
                        {
                            "chunk_id": f"kb_{i:03d}",
                            "source_name": DEFAULT_SOURCE_NAME,
                            "text": row.strip(),
                        }
                    )
            return out or None
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to read chunk_meta.json: %s", exc)
    if os.path.isfile(CHUNK_ORDER_PATH):
        try:
            with open(CHUNK_ORDER_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if not isinstance(raw, list) or not raw:
                return None
            return [
                {"chunk_id": f"kb_{i:03d}", "source_name": DEFAULT_SOURCE_NAME, "text": str(x)}
                for i, x in enumerate(raw)
                if str(x).strip()
            ]
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to read chunk_order.json: %s", exc)
    return None


class FaissRetriever:
    def __init__(
        self,
        encoder: SentenceBiEncoder,
        store: FaissIPVectorStore,
        meta: list[dict[str, str]],
    ) -> None:
        if int(store.ntotal) != len(meta):
            raise ValueError(f"index ntotal {store.ntotal} != len(meta) {len(meta)}")
        self._encoder = encoder
        self._store = store
        self._meta = meta

    @classmethod
    def try_load_default_files(
        cls,
        encoder: SentenceBiEncoder | None = None,
    ) -> FaissRetriever | None:
        """Load from default `rag/` paths; returns None if index or meta missing/invalid."""
        if not os.path.isfile(INDEX_PATH):
            logger.debug("RAG index not found (run python -m rag.embed).")
            return None
        meta = _load_meta_rows()
        if not meta:
            logger.debug("RAG chunk metadata missing.")
            return None
        try:
            store = FaissIPVectorStore.load(INDEX_PATH)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to load FAISS: %s", exc)
            return None
        if int(store.ntotal) != len(meta):
            logger.warning(
                "RAG index row count (%s) != metadata rows (%s).",
                store.ntotal,
                len(meta),
            )
            return None
        return cls(encoder or SentenceBiEncoder(), store, meta)

    def retrieve_candidates(self, query: str, top_k: int) -> list[dict[str, Any]]:
        q = " ".join(str(query or "").strip().split())
        if not q:
            return []
        k = max(1, int(top_k))
        n = len(self._meta)
        if n == 0:
            return []
        k = min(k, n)

        vec = self._encoder.encode([q])
        faiss.normalize_L2(vec)
        scores, indices = self._store.search(vec, k)

        out: list[dict[str, Any]] = []
        seen: set[int] = set()
        for rank in range(k):
            idx = int(indices[0][rank])
            if idx < 0 or idx >= n or idx in seen:
                continue
            seen.add(idx)
            row = self._meta[idx]
            out.append(
                {
                    "row_index": idx,
                    "chunk_id": row["chunk_id"],
                    "source_name": row["source_name"],
                    "text": row["text"],
                    "score": float(scores[0][rank]),
                }
            )
        if out:
            hits = ", ".join(f"{h['chunk_id']}:{h['score']:.3f}" for h in out)
            rag_debug(logger, "faiss top-%s: %s", len(out), hits)
        return out
