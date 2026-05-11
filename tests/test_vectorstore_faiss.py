import unittest

import faiss
import numpy as np

from rag.vectorstore.faiss_ip import FaissIPVectorStore


class TestFaissIPVectorStore(unittest.TestCase):
    def test_search_returns_best_neighbor(self) -> None:
        emb = np.eye(5, dtype=np.float32)
        faiss.normalize_L2(emb)
        store = FaissIPVectorStore.from_normalized_matrix(emb)
        q = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        faiss.normalize_L2(q)
        scores, indices = store.search(q, 2)
        self.assertEqual(int(indices[0][0]), 0)
        self.assertGreater(float(scores[0][0]), 0.0)

    def test_ntotal_matches_rows(self) -> None:
        emb = np.random.randn(3, 8).astype(np.float32)
        faiss.normalize_L2(emb)
        store = FaissIPVectorStore.from_normalized_matrix(emb)
        self.assertEqual(store.ntotal, 3)
