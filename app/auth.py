from __future__ import annotations

import hashlib
import re
import secrets
import time
import uuid

from flask import request
from werkzeug.exceptions import ServiceUnavailable, Unauthorized

from app.db import connect_pg, connect_sqlite, db_url

SESSION_SECONDS = 365 * 24 * 60 * 60


def create_session() -> dict:
    token = secrets.token_urlsafe(32)
    visitor_id = "session-" + uuid.uuid4().hex
    expires_at = int(time.time()) + SESSION_SECONDS
    token_hash = hashlib.sha256(token.encode("ascii")).hexdigest()
    postgres = bool(db_url())
    placeholder = "%s" if postgres else "?"
    try:
        with (connect_pg() if postgres else connect_sqlite()) as conn:
            conn.execute(
                "INSERT INTO visitor_sessions (token_hash, visitor_id, expires_at) "
                f"VALUES ({placeholder}, {placeholder}, {placeholder})",
                (token_hash, visitor_id, expires_at),
            )
            conn.commit()
    except Exception as exc:
        raise ServiceUnavailable("memory service unavailable") from exc
    return {"sessionToken": token, "visitorId": visitor_id, "expiresAt": expires_at, "authVersion": 1}


def require_session_visitor() -> str:
    authorization = request.headers.get("Authorization", "")
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer" or not re.fullmatch(r"[A-Za-z0-9_-]{43}", parts[1]):
        raise Unauthorized("valid anonymous session required")
    token_hash = hashlib.sha256(parts[1].encode("ascii")).hexdigest()
    postgres = bool(db_url())
    placeholder = "%s" if postgres else "?"
    try:
        with (connect_pg() if postgres else connect_sqlite()) as conn:
            row = conn.execute(
                f"SELECT visitor_id, expires_at FROM visitor_sessions WHERE token_hash = {placeholder}",
                (token_hash,),
            ).fetchone()
    except Exception as exc:
        raise ServiceUnavailable("memory service unavailable") from exc
    if not row or int(row[1]) <= int(time.time()):
        raise Unauthorized("anonymous session expired or invalid")
    return str(row[0])
