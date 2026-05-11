import unittest

from rag.prompts.rag_prompt import RAGPromptBuilder


class TestRAGPromptBuilder(unittest.TestCase):
    def test_build_contains_chunk_metadata(self) -> None:
        b = RAGPromptBuilder()
        block, cites = b.build(
            [{"chunk_id": "kb_000", "source_name": "little_prince_kb", "text": "玫瑰与狐狸。"}]
        )
        self.assertIn("chunk_id=kb_000", block)
        self.assertIn("source_name=little_prince_kb", block)
        self.assertIn("玫瑰与狐狸", block)
        self.assertEqual(len(cites), 1)
        self.assertEqual(cites[0]["chunk_id"], "kb_000")
        self.assertEqual(cites[0]["source_name"], "little_prince_kb")

    def test_empty_selection(self) -> None:
        b = RAGPromptBuilder()
        block, cites = b.build([])
        self.assertEqual(block, "")
        self.assertEqual(cites, [])
