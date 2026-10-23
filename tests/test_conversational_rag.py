"""Pipeline checks; real reply quality is assessed separately with live calls."""
import os
from unittest.mock import patch

from app.services.chat_pipeline import run_chat
from app.services.agent_service import decide_strategy
from rag.chat_context import supplement_chat_context
from rag.prompts.rag_prompt import RAGPromptBuilder


def test_chat_keeps_machine_citations_separate_from_conversation_context():
    row = {"chunk_id": "kb_001", "source_name": "little_prince_kb", "chapter": "第二十一章", "text": "狐狸说驯养就是建立联系。", "source_url": "https://example.org/book"}
    block, citations = RAGPromptBuilder().build([row], conversational=True)
    assert "狐狸说驯养就是建立联系" in block
    assert "kb_001" not in block and "little_prince_kb" not in block
    assert "合理解释" in block and "不能加引号冒充原话" in block
    assert citations[0]["chunk_id"] == "kb_001"
    assert citations[0]["source_url"] == "https://example.org/book"


def test_strict_analysis_still_requires_auditable_inline_sources():
    block, citations = RAGPromptBuilder().build([{"chunk_id": "kb_001", "source_name": "little_prince_kb", "text": "建立联系"}])
    assert "仅依据摘录" in block and "强制引用" in block
    assert "kb_001" in block and "little_prince_kb" in block
    assert citations


def test_live_pipeline_uses_conversation_grounding_for_short_followup():
    history = [{"role": "user", "content": "狐狸说的驯养是什么意思？"}, {"role": "assistant", "content": "建立联系。"}, {"role": "user", "content": "为什么呢？"}]
    with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-only", "RAG_RETRIEVAL_MODE": "lexical"}), patch("app.services.llm_service.DeepSeekChatGenerator.generate", return_value="因为相处让彼此变得熟悉。") as generate:
        result = run_chat(history)
    system = generate.call_args.args[0][0]["content"]
    assert "【聊天参考】" in system and "仅依据摘录作答" not in system
    assert "kb_001" not in system and "chat_21_" not in system
    assert result["citations"] and result["analysis"] == ""
    assert generate.call_args.args[0][-3:] == history


def test_verified_context_explains_patience_without_inventing_a_duration():
    rows = supplement_chat_context("狐狸为什么说驯养要有耐心？", [])
    assert any("每天靠近一点" in row["text"] for row in rows)
    rows = supplement_chat_context("狐狸和小王子相处多少天？", [])
    assert any("多少天" in row["text"] for row in rows)
    assert all(row["chunk_type"] == "chapter_summary" and row["source_url"].startswith("https://gutenberg.net.au/") for row in rows)


def test_unrelated_chat_does_not_gain_a_book_citation():
    assert supplement_chat_context("火锅辣椒", []) == []
    with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-only"}), patch("app.services.rag_service.default_rag_orchestrator", side_effect=AssertionError("No book retrieval")), patch("app.services.llm_service.DeepSeekChatGenerator.generate", return_value="哈哈，下次少放点辣。"):
        result = run_chat([{"role": "user", "content": "先不聊这个，我今天吃的面太辣了。"}])
    assert result["citations"] == []


def test_empty_selection_does_not_force_a_reference_or_refusal():
    assert RAGPromptBuilder().build([], conversational=True) == ("", [])


def test_explanatory_and_source_followups_keep_the_book_topic_until_a_switch():
    history = [{"role": "user", "content": "狐狸为什么会喜欢麦田？"}, {"role": "assistant", "content": "麦田会使他想起朋友。"}]
    for text in ["为什么这样就特别了？", "这段在书的哪里？", "你刚才说的都是书里的原话吗？"]:
        history.append({"role": "user", "content": text})
        assert decide_strategy(text, history).use_rag
        history.append({"role": "assistant", "content": "这是第二十一章的内容梗概。"})
    history += [{"role": "user", "content": "今天吃火锅太辣了"}, {"role": "assistant", "content": "哈哈，下次微辣。"}, {"role": "user", "content": "为什么会这样呢？"}]
    assert not decide_strategy(history[-1]["content"], history).use_rag
