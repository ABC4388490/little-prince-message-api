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

    def test_build_extended_metadata_and_content(self) -> None:
        b = RAGPromptBuilder()
        block, cites = b.build(
            [
                {
                    "chunk_id": "passage_21_01",
                    "source_name": "little_prince_kb",
                    "text": "fallback",
                    "content": "【原文】请你驯服我吧。\n【释义】狐狸请求建立联系。",
                    "chapter": "第二十一章 狐狸",
                    "topic": ["驯服", "爱"],
                    "role": ["狐狸", "小王子"],
                    "chunk_type": "passage",
                }
            ]
        )
        self.assertIn("chapter=第二十一章 狐狸", block)
        self.assertIn("topic=驯服,爱", block)
        self.assertIn("role=狐狸,小王子", block)
        self.assertIn("chunk_type=passage", block)
        self.assertIn("【原文】请你驯服我吧", block)
        self.assertNotIn("fallback", block)
        self.assertIn("【释义】", cites[0]["text"])

    def test_empty_selection(self) -> None:
        b = RAGPromptBuilder()
        block, cites = b.build([])
        self.assertEqual(block, "")
        self.assertEqual(cites, [])
