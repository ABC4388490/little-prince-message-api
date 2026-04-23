from __future__ import annotations

import json
import os
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Optional, Sequence
from urllib import error as urlerror
from urllib import request as urlrequest

from flask import Flask, jsonify, request
from flask_cors import CORS

try:
    import psycopg
except Exception:  # pragma: no cover
    psycopg = None  # type: ignore


SYSTEM_PROMPT = "\n".join(
    [
        "你是小王子的星球管家，住在 B612。",
        "你的语气温柔、童真、克制，像夜里轻声说话。",
        "请用富有画面感的比喻写 2-4 句短句，给用户一段带哲理的回应。",
        "结构建议：先共情，再给一小句启发，最后留一句柔和的提问。",
        "避免说教、避免口号、避免网络热词、避免 emoji。",
    ]
)

FALLBACK_REPLY = (
    "我听见了你的心事。就像黄昏里的风，不必一下子把答案带来，"
    "先把今天最在意的一件小事轻轻放在掌心里看看——"
    "它也许就是通往星光的第一步。"
)


def _load_local_env_file() -> None:
    """
    Load key=value pairs from message-api/.env into process env.
    Existing environment variables are preserved.
    """
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        return
    raw_text = ""
    for enc in ("utf-8", "utf-8-sig", "utf-16", "gbk"):
        try:
            with open(env_path, "r", encoding=enc) as f:
                raw_text = f.read()
            if raw_text:
                break
        except Exception:
            continue
    if not raw_text:
        return
    for raw in raw_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and (key not in os.environ or not str(os.environ.get(key, "")).strip()):
            os.environ[key] = value


_load_local_env_file()


@dataclass(frozen=True)
class Message:
    id: int
    content: str
    posX: float
    posY: float
    reply: str
    createdAt: str
    replyCreatedAt: str


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _db_url() -> str:
    return os.environ.get("DATABASE_URL", "").strip()


def _db_path() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "messages.db")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
    names = {str(c["name"]) for c in cols}
    if column not in names:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def _init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                posX REAL NOT NULL,
                posY REAL NOT NULL,
                createdAt TEXT NOT NULL
            )
            """
        )
        # 兼容已有旧库：在线增列，不破坏历史数据
        _ensure_column(conn, "messages", "reply", "TEXT")
        _ensure_column(conn, "messages", "replyCreatedAt", "TEXT")


def _pg_connect():
    if not _db_url():
        raise RuntimeError("DATABASE_URL is not set")
    if psycopg is None:
        raise RuntimeError("psycopg is not installed")
    return psycopg.connect(_db_url())


def _pg_init_db() -> None:
    """
    Create tables for conversation memory (Postgres).

    Railway typically provides DATABASE_URL for Postgres.
    """
    if not _db_url():
        return
    with _pg_connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id BIGSERIAL PRIMARY KEY,
                visitor_id TEXT NOT NULL UNIQUE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages_v2 (
                id BIGSERIAL PRIMARY KEY,
                conversation_id BIGINT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                pos_x REAL,
                pos_y REAL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )
        # online migration for older deployments
        conn.execute("ALTER TABLE messages_v2 ADD COLUMN IF NOT EXISTS pos_x REAL;")
        conn.execute("ALTER TABLE messages_v2 ADD COLUMN IF NOT EXISTS pos_y REAL;")
        conn.commit()


def _parse_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        f = float(value)
        if f != f:
            return None
        if f in (float("inf"), float("-inf")):
            return None
        return f
    except Exception:
        return None


def _safe_message_text(text: str, limit: int = 1800) -> str:
    clean = " ".join(str(text or "").strip().split())
    return clean[:limit]


def _call_deepseek(user_text: str, context: Optional[Sequence[dict[str, str]]] = None) -> str:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        return FALLBACK_REPLY

    url = os.environ.get("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions").strip()
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat").strip()

    base_messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if context:
        # context is a list like: [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]
        base_messages.extend(context)
    base_messages.append({"role": "user", "content": user_text})

    payload = {
        "model": model,
        "temperature": 0.8,
        "messages": base_messages,
    }
    req = urlrequest.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urlrequest.urlopen(req, timeout=20) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            data = json.loads(body)
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            content = _safe_message_text(content, limit=220)
            return content or FALLBACK_REPLY
    except (urlerror.URLError, TimeoutError, json.JSONDecodeError, KeyError, IndexError, ValueError):
        return FALLBACK_REPLY


app = Flask(__name__)
cors_origins = os.environ.get("CORS_ORIGINS", "*").strip()
origins = "*"
if cors_origins and cors_origins != "*":
    origins = [o.strip() for o in cors_origins.split(",") if o.strip()]
CORS(app, resources={r"/api/*": {"origins": origins}})
_init_db()
_pg_init_db()


@app.get("/health")
def health() -> Any:
    return jsonify({"ok": True})


@app.get("/api/messages")
def list_messages() -> Any:
    with _connect() as conn:
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


@app.post("/api/messages")
def create_message() -> Any:
    data = request.get_json(silent=True) or {}
    content = _safe_message_text(str(data.get("content") or ""), limit=900)
    posX = _parse_float(data.get("posX"))
    posY = _parse_float(data.get("posY"))

    if not content:
        return jsonify({"error": "content is required"}), 400
    if posX is None or posY is None:
        return jsonify({"error": "posX and posY must be numbers"}), 400

    posX = max(0.0, min(100.0, posX))
    posY = max(0.0, min(100.0, posY))

    created_at = _utc_iso_now()
    reply = _call_deepseek(content)
    reply_created_at = _utc_iso_now()

    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO messages (content, posX, posY, createdAt, reply, replyCreatedAt)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (content, posX, posY, created_at, reply, reply_created_at),
        )
        new_id = int(cur.lastrowid)

    msg = Message(
        id=new_id,
        content=content,
        posX=posX,
        posY=posY,
        reply=reply,
        createdAt=created_at,
        replyCreatedAt=reply_created_at,
    )
    return jsonify(asdict(msg)), 201


def _require_visitor_id() -> str:
    visitor_id = (
        request.args.get("visitorId")
        or (request.get_json(silent=True) or {}).get("visitorId")
        or request.headers.get("X-Visitor-Id")
        or ""
    )
    visitor_id = str(visitor_id).strip()
    if not visitor_id:
        raise ValueError("visitorId is required")
    # Accept UUID-like strings; don't hard-fail on older ids, but cap length.
    return visitor_id[:128]


@app.get("/api/conversations/me")
def get_or_create_conversation() -> Any:
    try:
        visitor_id = _require_visitor_id()
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if not _db_url():
        return jsonify({"error": "DATABASE_URL is not configured"}), 503

    with _pg_connect() as conn:
        row = conn.execute(
            "SELECT id FROM conversations WHERE visitor_id = %s",
            (visitor_id,),
        ).fetchone()
        if row:
            return jsonify({"conversationId": int(row[0])})

        cur = conn.execute(
            "INSERT INTO conversations (visitor_id) VALUES (%s) RETURNING id",
            (visitor_id,),
        )
        new_id = int(cur.fetchone()[0])
        conn.commit()
        return jsonify({"conversationId": new_id}), 201


@app.get("/api/conversations/<int:conversation_id>/messages")
def list_conversation_messages(conversation_id: int) -> Any:
    if not _db_url():
        return jsonify({"error": "DATABASE_URL is not configured"}), 503

    with _pg_connect() as conn:
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


def _build_context_for_llm(conversation_id: int, max_pairs: int = 6) -> list[dict[str, str]]:
    """
    Return last N messages (user/assistant) as LLM context.
    """
    with _pg_connect() as conn:
        rows = conn.execute(
            """
            SELECT role, content
            FROM messages_v2
            WHERE conversation_id = %s
            ORDER BY id DESC
            LIMIT %s
            """,
            (conversation_id, max_pairs * 2),
        ).fetchall()
    rows = list(reversed(rows))
    return [{"role": str(r[0]), "content": _safe_message_text(str(r[1]), limit=220)} for r in rows]


@app.post("/api/conversations/<int:conversation_id>/messages")
def post_conversation_message(conversation_id: int) -> Any:
    if not _db_url():
        return jsonify({"error": "DATABASE_URL is not configured"}), 503

    data = request.get_json(silent=True) or {}
    content = _safe_message_text(str(data.get("content") or ""), limit=900)
    if not content:
        return jsonify({"error": "content is required"}), 400

    pos_x = _parse_float(data.get("posX"))
    pos_y = _parse_float(data.get("posY"))
    if pos_x is not None:
        pos_x = max(0.0, min(100.0, pos_x))
    if pos_y is not None:
        pos_y = max(0.0, min(100.0, pos_y))

    # Create assistant reply with context.
    context = _build_context_for_llm(conversation_id, max_pairs=6)
    reply = _call_deepseek(content, context=context)

    with _pg_connect() as conn:
        # Ensure conversation exists
        exists = conn.execute("SELECT 1 FROM conversations WHERE id = %s", (conversation_id,)).fetchone()
        if not exists:
            return jsonify({"error": "conversation not found"}), 404

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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

