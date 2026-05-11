"""
Cross-encoder reranking (pluggable model chain + optional custom scorer).

Env: RAG_RERANKER_MODELS, RAG_RERANKER_MODEL, RAG_DISABLE_RERANK, RAG_FINAL_TOP_K (via rag.config).
"""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from typing import Any, Callable, Optional

from rag.config import rerank_output_top_k as _final_k_from_config
from rag.trace import rag_debug

logger = logging.getLogger(__name__)


def _trace_rerank_result(
    candidates: list[dict[str, Any]],
    result: list[dict[str, Any]],
    *,
    mode: str,
) -> None:
    ids = [str(r.get("chunk_id", "")) for r in result]
    rag_debug(
        logger,
        "rerank %s: in=%s out=%s [%s]",
        mode,
        len(candidates),
        len(result),
        ",".join(ids),
    )

DEFAULT_MODEL_CHAIN: tuple[str, ...] = (
    "BAAI/bge-reranker-base",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)

_custom_scorer: Optional[Callable[[str, list[str]], list[float]]] = None

_default_reranker: Optional["CrossEncoderReranker"] = None


def set_rerank_scorer(fn: Optional[Callable[[str, list[str]], list[float]]]) -> None:
    global _custom_scorer, _default_reranker
    _custom_scorer = fn
    _default_reranker = None


def rerank_model_chain() -> list[str]:
    raw = os.environ.get("RAG_RERANKER_MODELS", "").strip()
    if raw:
        return [p.strip() for p in raw.split(",") if p.strip()]
    single = os.environ.get("RAG_RERANKER_MODEL", "").strip()
    if single:
        return [single]
    return list(DEFAULT_MODEL_CHAIN)


def active_reranker_model_name() -> str | None:
    return default_reranker().active_model_name


def default_reranker() -> CrossEncoderReranker:
    global _default_reranker
    if _default_reranker is None:
        _default_reranker = CrossEncoderReranker()
    return _default_reranker


def rerank_output_top_k() -> int:
    return _final_k_from_config()


class CrossEncoderReranker:
    """Rerank (query, passage) pairs; each instance has its own CrossEncoder lazy cache."""

    def __init__(self, *, model_chain: list[str] | None = None) -> None:
        self._model_chain = list(model_chain) if model_chain is not None else rerank_model_chain()
        self._ce: Any = None
        self._ce_failed = False
        self._active_model: str | None = None

    @property
    def active_model_name(self) -> str | None:
        return self._active_model

    def _get_cross_encoder(self):
        if _custom_scorer is not None:
            return None
        if self._ce_failed:
            return None
        if self._ce is not None:
            return self._ce

        from sentence_transformers import CrossEncoder

        for name in self._model_chain:
            try:
                model = CrossEncoder(name)
                self._ce = model
                self._active_model = name
                logger.debug("RAG reranker loaded: %s", name)
                return model
            except Exception as exc:  # pragma: no cover
                logger.warning("Reranker unavailable, skipping %s: %s", name, exc)

        logger.warning("No reranker model could be loaded; falling back to FAISS order.")
        self._ce_failed = True
        self._active_model = None
        return None

    @staticmethod
    def _rerank_disabled() -> bool:
        v = os.environ.get("RAG_DISABLE_RERANK", "").strip().lower()
        return v in ("1", "true", "yes", "on")

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        final_k: int | None = None,
    ) -> list[dict[str, Any]]:
        fk = final_k if final_k is not None else _final_k_from_config()
        fk = max(1, int(fk))
        if not candidates:
            _trace_rerank_result(candidates, [], mode="no_candidates")
            return []
        if fk >= len(candidates):
            fk = len(candidates)

        if self._rerank_disabled():
            out = [deepcopy(c) for c in candidates[:fk]]
            _trace_rerank_result(candidates, out, mode="rerank_disabled_faiss_order")
            return out

        q = str(query or "").strip()
        if not q:
            out = [deepcopy(c) for c in candidates[:fk]]
            _trace_rerank_result(candidates, out, mode="empty_query_faiss_order")
            return out

        passages = [str(c.get("text", "")) for c in candidates]

        if _custom_scorer is not None:
            try:
                scores = _custom_scorer(q, passages)
                if len(scores) != len(candidates):
                    raise ValueError("custom scorer length mismatch")
            except Exception as exc:  # pragma: no cover
                logger.warning("Custom rerank scorer failed: %s", exc)
                out = [deepcopy(c) for c in candidates[:fk]]
                _trace_rerank_result(candidates, out, mode="custom_scorer_failed_faiss_order")
                return out
        else:
            ce = self._get_cross_encoder()
            if ce is None:
                out = [deepcopy(c) for c in candidates[:fk]]
                _trace_rerank_result(candidates, out, mode="no_cross_encoder_faiss_order")
                return out
            pairs = [(q, p) for p in passages]
            try:
                scores = ce.predict(pairs, show_progress_bar=False)
            except Exception as exc:  # pragma: no cover
                logger.warning("Rerank predict failed: %s", exc)
                out = [deepcopy(c) for c in candidates[:fk]]
                _trace_rerank_result(candidates, out, mode="predict_failed_faiss_order")
                return out

        ranked = sorted(
            zip(scores, range(len(candidates))),
            key=lambda x: float(x[0]),
            reverse=True,
        )
        out = []
        for score, pos in ranked[:fk]:
            row = deepcopy(candidates[pos])
            row["rerank_score"] = float(score)
            out.append(row)
        _trace_rerank_result(candidates, out, mode="cross_encoder_ranked")
        return out


def rerank(query: str, candidates: list[dict[str, Any]], final_k: int | None = None) -> list[dict[str, Any]]:
    return default_reranker().rerank(query, candidates, final_k=final_k)
