from __future__ import annotations

from dataclasses import asdict
from typing import Any

from flask import Blueprint, jsonify, request

from app.api.common import require_messages_json
from app.db import connect_sqlite, parse_float, utc_iso_now
from app.models import Message
from app.services.llm_service import complete_chat, safe_message_text

bp = Blueprint("messages", __name__, url_prefix="/api")


@bp.get("/messages")
def list_messages() -> Any:
    with connect_sqlite() as conn:
        rows = conn.execute(
            """
            SELECT id, content, posX, posY, createdAt, reply, replyCreatedAt
            FROM messages
            ORDER BY id ASC
            """
        ).fetchall()
    messages = [
        Message(
            id=int(r["id"]),
            content=str(r["content"]),
            posX=float(r["posX"]),
            posY=float(r["posY"]),
            reply=str(r["reply"] or ""),
            createdAt=str(r["createdAt"]),
            replyCreatedAt=str(r["replyCreatedAt"] or r["createdAt"]),
        )
        for r in rows
    ]
    return jsonify([asdict(m) for m in messages])


@bp.post("/messages")
def create_message() -> Any:
    data = request.get_json(silent=True) or {}
    try:
        messages = require_messages_json(request)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    pos_x = parse_float(data.get("posX"))
    pos_y = parse_float(data.get("posY"))
    if pos_x is None or pos_y is None:
        return jsonify({"error": "posX and posY must be numbers"}), 400
    pos_x = max(0.0, min(100.0, pos_x))
    pos_y = max(0.0, min(100.0, pos_y))

    user_msgs = [m for m in messages if m.get("role") == "user" and m.get("content")]
    content = safe_message_text(user_msgs[-1]["content"] if user_msgs else "", limit=900)
    if not content:
        return jsonify({"error": "messages[] must contain a user message"}), 400

    created_at = utc_iso_now()
    reply = complete_chat(messages)
    reply_created_at = utc_iso_now()

    with connect_sqlite() as conn:
        cur = conn.execute(
            """
            INSERT INTO messages (content, posX, posY, createdAt, reply, replyCreatedAt)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (content, pos_x, pos_y, created_at, reply, reply_created_at),
        )
        new_id = int(cur.lastrowid)

    msg = Message(
        id=new_id,
        content=content,
        posX=pos_x,
        posY=pos_y,
        reply=reply,
        createdAt=created_at,
        replyCreatedAt=reply_created_at,
    )
    return jsonify(asdict(msg)), 201
