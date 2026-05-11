"""
Shared chat + RAG pipeline (Flask and FastAPI).

Returns strategy, citations, structured conclusion/analysis, and assistant block.
"""

from __future__ import annotations

from typing import Any

from app.db import utc_iso_now
from app.services.agent_service import decide_strategy, merge_agent_tone
from app.services.llm_service import complete_chat_structured
from app.services.rag_service import merge_rag


def run_chat(messages: list[dict[str, str]]) -> dict[str, Any]:
    user_turns = [
        m
        for m in messages
        if isinstance(m, dict) and m.get("role") == "user" and str(m.get("content", "")).strip()
    ]
    last_user = str(user_turns[-1].get("content", "")) if user_turns else ""
    decision = decide_strategy(last_user)

    msgs = merge_agent_tone(messages, decision)
    msgs, citations = merge_rag(msgs, use_rag=decision.use_rag)
    structured = complete_chat_structured(msgs)

    return {
        "strategy": decision.strategy,
        "citations": citations,
        "conclusion": structured["conclusion"],
        "analysis": structured["analysis"],
        "assistant": {
            "role": "assistant",
            "content": structured["assistant_plain"],
            "createdAt": utc_iso_now(),
        },
    }
