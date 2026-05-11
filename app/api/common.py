from __future__ import annotations

from typing import Any

from flask import Request

from app.services.llm_service import safe_message_text


def validate_messages_list(msgs: Any) -> list[dict[str, str]]:
    """Normalize and validate OpenAI-style messages (shared by Flask and FastAPI)."""
    if not isinstance(msgs, list) or not msgs:
        raise ValueError("messages[] is required")
    msgs = msgs[-24:]
    out: list[dict[str, str]] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "").strip()
        content = str(m.get("content") or "")
        if role not in ("system", "user", "assistant"):
            continue
        content = safe_message_text(content, limit=1200 if role == "system" else 900)
        if not content:
            continue
        out.append({"role": role, "content": content})
    if not out:
        raise ValueError("messages[] is empty")
    return out


def require_messages_json(req: Request) -> list[dict[str, str]]:
    data = req.get_json(silent=True) or {}
    msgs = data.get("messages")
    return validate_messages_list(msgs)


def require_visitor_id(req: Request) -> str:
    visitor_id = (
        req.args.get("visitorId")
        or (req.get_json(silent=True) or {}).get("visitorId")
        or req.headers.get("X-Visitor-Id")
        or ""
    )
    visitor_id = str(visitor_id).strip()
    if not visitor_id:
        raise ValueError("visitorId is required")
    return visitor_id[:128]
