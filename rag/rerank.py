"""Backward-compatible imports for reranking (implementation in `rag.reranker`)."""

from rag.reranker.cross_encoder import (
    active_reranker_model_name,
    default_reranker,
    rerank,
    rerank_model_chain,
    rerank_output_top_k,
    set_rerank_scorer,
)

__all__ = [
    "rerank",
    "rerank_model_chain",
    "rerank_output_top_k",
    "active_reranker_model_name",
    "set_rerank_scorer",
    "default_reranker",
]
