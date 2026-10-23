import unittest

from rag.corpus import normalize_records


class TestCorpusNormalize(unittest.TestCase):
    def test_preserves_extended_fields(self) -> None:
        payload = {
            "default_source_name": "little_prince_kb",
            "chunks": [
                {
                    "chunk_id": "passage_01_01",
                    "text": "index body",
                    "content": "【原文】短引\n【释义】释义",
                    "chapter": "第一章",
                    "topic": ["童年"],
                    "role": ["飞行员"],
                    "chunk_type": "passage",
                }
            ],
        }
        rows = normalize_records(payload)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["chunk_id"], "passage_01_01")
        self.assertEqual(rows[0]["content"], "【原文】短引\n【释义】释义")
        self.assertEqual(rows[0]["chapter"], "第一章")
        self.assertEqual(rows[0]["topic"], ["童年"])
        self.assertEqual(rows[0]["role"], ["飞行员"])
        self.assertEqual(rows[0]["chunk_type"], "passage")
