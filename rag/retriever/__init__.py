"""Retrieve candidates from a vector index + bi-encoder query."""

from rag.config import faiss_retrieval_top_k
from rag.retriever.session import get_retriever, retrieve, retrieve_candidates


def __getattr__(name: str):
    if name == "FaissRetriever":
        from rag.retriever.faiss_retriever import FaissRetriever

        return FaissRetriever
    raise AttributeError(name)

__all__ = [
    "FaissRetriever",
    "get_retriever",
    "retrieve_candidates",
    "retrieve",
    "faiss_retrieval_top_k",
]
