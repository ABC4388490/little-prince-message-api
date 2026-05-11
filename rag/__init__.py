"""
RAG stack (modular):

- ``rag.embedding`` — bi-encoder
- ``rag.vectorstore`` — FAISS IP index
- ``rag.retriever`` — query → candidates
- ``rag.reranker`` — cross-encoder rerank
- ``rag.prompts`` — grounding block + citations
- ``rag.generator`` — DeepSeek HTTP client
- ``rag.pipeline`` — orchestrator

Legacy: ``from rag.retrieve import retrieve`` (avoid importing heavy deps via ``import rag`` alone).
"""


def __getattr__(name: str):
    if name == "retrieve":
        from rag.retrieve import retrieve

        return retrieve
    if name == "retrieve_candidates":
        from rag.retrieve import retrieve_candidates

        return retrieve_candidates
    raise AttributeError(name)


__all__ = ["retrieve", "retrieve_candidates"]
