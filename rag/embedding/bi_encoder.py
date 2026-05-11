"""Dense sentence embeddings (independent of FAISS / retrieval)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class SentenceBiEncoder:
    """
    Lazy-loaded MiniLM (or any) SentenceTransformer.
    Testable by mocking `encode` or injecting a fake model in subclasses.
    """

    def __init__(self, model_name: str | None = None) -> None:
        from rag.config import BI_ENCODER_MODEL

        self.model_name = (model_name or BI_ENCODER_MODEL).strip()
        self._model: Optional["SentenceTransformer"] = None

    def _model_or_load(self) -> "SentenceTransformer":
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str], *, show_progress_bar: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        arr = self._model_or_load().encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=show_progress_bar,
        )
        return np.asarray(arr, dtype=np.float32)
