"""
Shared chat + RAG pipeline (Flask and FastAPI).

Returns a plain conversational reply; conclusion/analysis remain compatibility fields.
"""

from __future__ import annotations

from typing import Any

from app.db import utc_iso_now
from app.services.agent_service import decide_strategy, merge_agent_tone
from app.services.llm_service import complete_chat
from app.services.rag_service import merge_rag


def run_chat(messages: list[dict[str, str]]) -> dict[str, Any]:
    user_turns = [
        m
        for m in messages
        if isinstance(m, dict) and m.get("role") == "user" and str(m.get("content", "")).strip()
    ]
    last_user = str(user_turns[-1].get("content", "")) if user_turns else ""
    decision = decide_strategy(last_user, messages)

    msgs = merge_agent_tone(messages, decision)
    msgs, citations = merge_rag(msgs, use_rag=decision.use_rag)
    reply = complete_chat(msgs, skip_rag=True, strict=True)

    return {
        "strategy": decision.strategy,
        "citations": citations,
        "conclusion": reply,
        "analysis": "",
        "assistant": {
            "role": "assistant",
            "content": reply,
            "createdAt": utc_iso_now(),
        },
    }
