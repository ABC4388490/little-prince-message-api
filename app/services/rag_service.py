from __future__ import annotations

import logging
from typing import Any, Sequence

from app.services.llm_service import SYSTEM_PROMPT, safe_message_text
from rag.pipeline import default_rag_orchestrator
from rag.trace import rag_debug

logger = logging.getLogger(__name__)


def merge_rag(
    messages: Sequence[dict[str, str]],
    *,
    use_rag: bool = True,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """
    When use_rag: retrieval pipeline → prompt builder → inject grounding into first system.
    Returns (messages, citations) for API; citations empty when RAG off or failed.
    """
    msgs: list[dict[str, str]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", "")).strip()
        content = str(m.get("content", ""))
        if role not in ("system", "user", "assistant"):
            continue
        msgs.append({"role": role, "content": content})

    if not use_rag:
        return msgs, []

    user_msgs = [m for m in msgs if m.get("role") == "user" and m.get("content", "").strip()]
    if not user_msgs:
        return msgs, []

    query = user_msgs[-1]["content"]
    citations: list[dict[str, Any]] = []
    try:
        orch = default_rag_orchestrator()
        selected = orch.retrieve_and_rank(query)
        rag_block, citations = orch.build_prompt(selected)
    except Exception:
        logger.exception("merge_rag failed (query=%.80s)", query)
        return msgs, []

    if not rag_block.strip():
        return msgs, []

    merged_cap = 3200
    first_sys = next((i for i, m in enumerate(msgs) if m.get("role") == "system"), None)
    if first_sys is not None:
        merged = msgs[first_sys]["content"] + "\n\n" + rag_block
        msgs[first_sys] = {"role": "system", "content": safe_message_text(merged, limit=merged_cap)}
    else:
        msgs.insert(
            0,
            {
                "role": "system",
                "content": safe_message_text(SYSTEM_PROMPT + "\n\n" + rag_block, limit=merged_cap),
            },
        )

    cite = ",".join(f"{c.get('label')}:{c.get('chunk_id')}" for c in citations) if citations else "-"
    sys_len = len(next((m["content"] for m in msgs if m.get("role") == "system"), ""))
    rag_debug(logger, "merge_rag ok: citations=%s system_chars=%s", cite, sys_len)

    return msgs, citations
