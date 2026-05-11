"""Backward-compatible imports for FAISS retrieval (implementation in `rag.retriever`)."""

from rag.config import faiss_retrieval_top_k
from rag.retriever.session import get_retriever, retrieve, retrieve_candidates

__all__ = [
    "retrieve_candidates",
    "retrieve",
    "faiss_retrieval_top_k",
    "get_retriever",
]
