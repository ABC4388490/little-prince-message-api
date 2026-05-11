import unittest
from unittest.mock import patch

from rag.pipeline.orchestrator import RetrievalPipeline


class _PassthroughReranker:
    def rerank(self, query, candidates, final_k=None):
        fk = 3 if final_k is None else int(final_k)
        return list(candidates)[:fk]


class TestRetrievalPipeline(unittest.TestCase):
    @patch("rag.pipeline.orchestrator.retrieve_candidates")
    def test_run_normalizes_metadata(self, mock_rc) -> None:
        mock_rc.return_value = [
            {
                "row_index": 0,
                "chunk_id": "kb_000",
                "source_name": "little_prince_kb",
                "text": "hello",
                "score": 0.9,
            },
            {
                "row_index": 1,
                "chunk_id": "kb_001",
                "source_name": "little_prince_kb",
                "text": "world",
                "score": 0.8,
            },
        ]
        pipe = RetrievalPipeline(reranker=_PassthroughReranker())
        out = pipe.run("q")
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["chunk_id"], "kb_000")
        self.assertEqual(out[0]["source_name"], "little_prince_kb")
