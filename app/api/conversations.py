from __future__ import annotations

from typing import Any

from flask import Blueprint, jsonify, request

from app.api.common import require_messages_json
from app.auth import require_session_visitor
from app.db import connect_pg, db_url, parse_float, utc_iso_now
from app.services.llm_service import complete_chat, safe_message_text

bp = Blueprint("conversations", __name__, url_prefix="/api")


@bp.after_request
def private_response(response):
    response.headers["Cache-Control"] = "no-store"
    return response


def require_conversation_owner(conn, conversation_id, visitor_id):
    from werkzeug.exceptions import NotFound

    row = conn.execute(
        "SELECT 1 FROM conversations WHERE id = %s AND visitor_id = %s",
        (conversation_id, visitor_id),
    ).fetchone()
    if not row:
        raise NotFound("conversation not found")


@bp.get("/conversations/me")
def get_or_create_conversation() -> Any:
    try:
        visitor_id = require_session_visitor()
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if not db_url():
        return jsonify({"error": "DATABASE_URL is not configured"}), 503

    with connect_pg() as conn:
        row = conn.execute(
            """
            INSERT INTO conversations (visitor_id) VALUES (%s)
            ON CONFLICT (visitor_id) DO UPDATE SET visitor_id = EXCLUDED.visitor_id
            RETURNING id
            """,
            (visitor_id,),
        ).fetchone()
        conn.commit()
        return jsonify({"conversationId": int(row[0])})



@bp.get("/conversations/<int:conversation_id>/messages")
def list_conversation_messages(conversation_id: int) -> Any:
    visitor_id = require_session_visitor()
    if not db_url():
        return jsonify({"error": "DATABASE_URL is not configured"}), 503

    with connect_pg() as conn:
        require_conversation_owner(conn, conversation_id, visitor_id)
        rows = conn.execute(
            """
            SELECT id, role, content, pos_x, pos_y, created_at
            FROM messages_v2
            WHERE conversation_id = %s
            ORDER BY id ASC
            """,
            (conversation_id,),
        ).fetchall()
    items = [
        {
            "id": int(r[0]),
            "role": str(r[1]),
            "content": str(r[2]),
            "posX": (float(r[3]) if r[3] is not None else None),
            "posY": (float(r[4]) if r[4] is not None else None),
            "createdAt": (r[5].isoformat() if hasattr(r[5], "isoformat") else str(r[5])),
        }
        for r in rows
    ]
    return jsonify(items)


@bp.post("/conversations/<int:conversation_id>/messages")
def post_conversation_message(conversation_id: int) -> Any:
    visitor_id = require_session_visitor()
    if not db_url():
        return jsonify({"error": "DATABASE_URL is not configured"}), 503

    data = request.get_json(silent=True) or {}
    try:
        messages = require_messages_json(request)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    pos_x = parse_float(data.get("posX"))
    pos_y = parse_float(data.get("posY"))
    if pos_x is not None:
        pos_x = max(0.0, min(100.0, pos_x))
    if pos_y is not None:
        pos_y = max(0.0, min(100.0, pos_y))

    user_msgs = [m for m in messages if m.get("role") == "user" and m.get("content")]
    content = safe_message_text(user_msgs[-1]["content"] if user_msgs else "", limit=900)
    if not content:
        return jsonify({"error": "messages[] must contain a user message"}), 400

    # Reject foreign IDs before spending an LLM request.
    with connect_pg() as conn:
        require_conversation_owner(conn, conversation_id, visitor_id)

    reply = complete_chat(messages)

    with connect_pg() as conn:
        require_conversation_owner(conn, conversation_id, visitor_id)

        cur_user = conn.execute(
            """
            INSERT INTO messages_v2 (conversation_id, role, content, pos_x, pos_y)
            VALUES (%s, 'user', %s, %s, %s)
            RETURNING id, created_at
            """,
            (conversation_id, content, pos_x, pos_y),
        )
        user_row = cur_user.fetchone()

        cur_asst = conn.execute(
            """
            INSERT INTO messages_v2 (conversation_id, role, content)
            VALUES (%s, 'assistant', %s)
            RETURNING id, created_at
            """,
            (conversation_id, reply),
        )
        asst_row = cur_asst.fetchone()
        conn.commit()

    def _iso(v: Any) -> str:
        return v.isoformat() if hasattr(v, "isoformat") else str(v)

    return jsonify(
        {
            "user": {
                "id": int(user_row[0]),
                "role": "user",
                "content": content,
                "createdAt": _iso(user_row[1]),
            },
            "assistant": {
                "id": int(asst_row[0]),
                "role": "assistant",
                "content": reply,
                "createdAt": _iso(asst_row[1]),
            },
        }
    ), 201
