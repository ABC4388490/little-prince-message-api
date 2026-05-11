"""
FAISS inner-product store on L2-normalized vectors (cosine similarity).

No sentence-transformers dependency — only numpy + faiss.
"""

from __future__ import annotations

import os
from typing import Any

import faiss
import numpy as np


class FaissIPVectorStore:
    def __init__(self, index: faiss.Index) -> None:
        self._index = index

    @property
    def ntotal(self) -> int:
        return int(self._index.ntotal)

    @classmethod
    def from_normalized_matrix(cls, embeddings: np.ndarray) -> FaissIPVectorStore:
        """Build a flat IP index from a (n, dim) float32 matrix (rows should be L2-normalized)."""
        mat = np.asarray(embeddings, dtype=np.float32)
        if mat.ndim != 2 or mat.shape[0] == 0:
            raise ValueError("embeddings must be 2-D with n > 0")
        dim = mat.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(mat)
        return cls(index)

    def search(self, query_vectors: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        query_vectors: (batch, dim) float32, L2-normalized per row.
        Returns (scores, indices) each (batch, k).
        """
        q = np.asarray(query_vectors, dtype=np.float32)
        k = max(1, min(int(k), self.ntotal))
        scores, indices = self._index.search(q, k)
        return scores, indices

    def save(self, path: str) -> None:
        """Serialize index to bytes on disk (Windows-safe for non-ASCII paths)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        index_arr = faiss.serialize_index(self._index)
        if not isinstance(index_arr, np.ndarray):
            index_arr = np.asarray(index_arr, dtype=np.uint8)
        with open(path, "wb") as f:
            f.write(index_arr.tobytes())

    @classmethod
    def load(cls, path: str) -> FaissIPVectorStore:
        with open(path, "rb") as f:
            buf = f.read()
        index = faiss.deserialize_index(np.frombuffer(buf, dtype=np.uint8))
        return cls(index)
