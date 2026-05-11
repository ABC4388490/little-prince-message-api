"""
Interpretation-only pipeline: RAG on the given reply text + structured LLM (conclusion + analysis).
Does not use agent strategy or merge_rag (avoids double RAG / wrong routing).
"""

from __future__ import annotations

import logging
from typing import Any

from app.db import utc_iso_now
from app.services.llm_service import complete_chat_structured, safe_message_text
from rag.pipeline import default_rag_orchestrator
from rag.trace import rag_debug

logger = logging.getLogger(__name__)

MERGED_SYSTEM_CAP = 3200

# Prepended to RAG grounding block. complete_chat_structured appends JSON output format to first system.
INTERPRET_SYSTEM_PREFIX = "\n".join(
    [
        "你是《小王子》世界里的温柔旁白者，用中文说话。",
        "用户会给你一段「小王子的回信」；知识库摘录已附在下方，请只作为参照，不要逐字复述。",
        "你的解读要：语气柔软、有星星与玫瑰的意象、少训诫、少列表。",
        "先有一语中的的总结，再有一小段分析；具体输出格式由系统末尾的【输出格式】约定（JSON 键 conclusion / analysis）。",
    ]
)


def run_analyze(text: str) -> dict[str, Any]:
    """
    Retrieve against `text`, inject grounding, call structured LLM.
    """
    body = safe_message_text(str(text or ""), limit=1200)
    if not body:
        raise ValueError("text is required")

    citations: list[dict[str, Any]] = []
    rag_block = ""
    try:
        orch = default_rag_orchestrator()
        selected = orch.retrieve_and_rank(body)
        rag_block, citations = orch.build_prompt(selected)
    except Exception:
        logger.exception("run_analyze RAG failed (text=%.80s)", body)

    if rag_block.strip():
        merged = INTERPRET_SYSTEM_PREFIX + "\n\n" + rag_block
        system_content = safe_message_text(merged, limit=MERGED_SYSTEM_CAP)
    else:
        system_content = safe_message_text(INTERPRET_SYSTEM_PREFIX, limit=MERGED_SYSTEM_CAP)

    user_msg = (
        "请围绕下面这封「小王子回信」作答；若摘录为空，仍请基于《小王子》气质做温柔解读：\n\n"
        f"「{body}」"
    )
    user_msg = safe_message_text(user_msg, limit=1200)

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_msg},
    ]

    cite = ",".join(f"{c.get('label')}:{c.get('chunk_id')}" for c in citations) if citations else "-"
    rag_debug(logger, "run_analyze: citations=%s system_chars=%s", cite, len(system_content))

    structured = complete_chat_structured(messages)

    return {
        "strategy": "interpret",
        "citations": citations,
        "conclusion": structured["conclusion"],
        "analysis": structured["analysis"],
        "assistant": {
            "role": "assistant",
            "content": structured["assistant_plain"],
            "createdAt": utc_iso_now(),
        },
    }
