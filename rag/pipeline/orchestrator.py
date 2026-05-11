"""
Orchestrate retrieval stages: FAISS candidates → rerank → normalized chunk dicts.

Optional `RAGOrchestrator` bundles retrieval + prompt building for higher-level flows.
"""

from __future__ import annotations

from typing import Any

from rag.chunk_meta import ensure_chunk_metadata
from rag.config import faiss_retrieval_top_k, rerank_output_top_k
from rag.prompts.rag_prompt import RAGPromptBuilder
from rag.reranker.cross_encoder import CrossEncoderReranker, default_reranker
from rag.retriever.session import retrieve_candidates


class RetrievalPipeline:
    """Composable: retriever session + reranker + metadata normalization."""

    def __init__(self, reranker: CrossEncoderReranker | None = None) -> None:
        self._reranker = reranker if reranker is not None else default_reranker()

    def run(self, query: str) -> list[dict[str, Any]]:
        k_faiss = faiss_retrieval_top_k()
        k_out = rerank_output_top_k()
        candidates = retrieve_candidates(query, top_k=k_faiss)
        selected = self._reranker.rerank(query, candidates, final_k=k_out)
        return [ensure_chunk_metadata(x) for x in selected]


class RAGOrchestrator:
    """Retrieval pipeline + prompt/citation builder (no LLM call)."""

    def __init__(
        self,
        retrieval: RetrievalPipeline | None = None,
        prompt_builder: RAGPromptBuilder | None = None,
    ) -> None:
        self.retrieval = retrieval if retrieval is not None else RetrievalPipeline()
        self.prompts = prompt_builder if prompt_builder is not None else RAGPromptBuilder()

    def retrieve_and_rank(self, query: str) -> list[dict[str, Any]]:
        return self.retrieval.run(query)

    def build_prompt(self, selected: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        return self.prompts.build(selected)


_default_orch: RAGOrchestrator | None = None


def default_rag_orchestrator() -> RAGOrchestrator:
    global _default_orch
    if _default_orch is None:
        _default_orch = RAGOrchestrator()
    return _default_orch


def run_retrieval_pipeline(query: str) -> list[dict[str, Any]]:
    """Backward-compatible entry: same as `RetrievalPipeline().run(query)`."""
    return RetrievalPipeline().run(query)
