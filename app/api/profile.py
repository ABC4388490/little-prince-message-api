from __future__ import annotations

import json
from typing import Any, Optional

from flask import Blueprint, jsonify, request

from app.auth import require_session_visitor
from app.db import connect_pg, connect_sqlite, db_url, utc_iso_now

bp = Blueprint("profile", __name__, url_prefix="/api")

@bp.after_request
def private_response(response):
    response.headers["Cache-Control"] = "no-store"
    return response


_EMOTIONS = frozenset({"neutral", "low", "high"})

_DEFAULT_PROFILE: dict[str, Any] = {
    "emotion": "neutral",
    "worries": [],
    "likes": [],
    "keywords": [],
    "roundCount": 0,
    "updatedAt": None,
}


def _normalize_string_list(value: Any, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        s = str(item or "").strip()
        if not s or s in out:
            continue
        out.append(s[:64])
        if len(out) >= limit:
            break
    return out


def _normalize_profile_fields(data: dict[str, Any]) -> dict[str, Any]:
    emotion = str(data.get("emotion") or "neutral").strip()
    if emotion not in _EMOTIONS:
        raise ValueError("emotion must be neutral, low, or high")
    try:
        round_count = int(data.get("roundCount", 0))
    except (TypeError, ValueError) as e:
        raise ValueError("roundCount must be an integer") from e
    if round_count < 0:
        raise ValueError("roundCount must be >= 0")
    return {
        "emotion": emotion,
        "worries": _normalize_string_list(data.get("worries"), 8),
        "likes": _normalize_string_list(data.get("likes"), 8),
        "keywords": _normalize_string_list(data.get("keywords"), 12),
        "roundCount": round_count,
    }


def _parse_json_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return _normalize_string_list(raw, 12)
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="replace")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return []
        return _normalize_string_list(parsed, 12)
    return []


def _iso_timestamp(value: Any) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    s = str(value).strip()
    return s or None


def _row_to_api(row: Any) -> dict[str, Any]:
    worries = _parse_json_list(row[1])[:8]
    likes = _parse_json_list(row[2])[:8]
    keywords = _parse_json_list(row[3])[:12]
    return {
        "emotion": str(row[0] or "neutral"),
        "worries": worries,
        "likes": likes,
        "keywords": keywords,
        "roundCount": int(row[4] or 0),
        "updatedAt": _iso_timestamp(row[5]),
    }


def _get_profile_pg(visitor_id: str) -> Optional[dict[str, Any]]:
    with connect_pg() as conn:
        row = conn.execute(
            """
            SELECT emotion, worries, likes, keywords, round_count, updated_at
            FROM user_profiles
            WHERE visitor_id = %s
            """,
            (visitor_id,),
        ).fetchone()
    if not row:
        return None
    return _row_to_api(row)


def _upsert_profile_pg(visitor_id: str, profile: dict[str, Any]) -> dict[str, Any]:
    now = utc_iso_now()
    worries_json = json.dumps(profile["worries"], ensure_ascii=False)
    likes_json = json.dumps(profile["likes"], ensure_ascii=False)
    keywords_json = json.dumps(profile["keywords"], ensure_ascii=False)
    with connect_pg() as conn:
        row = conn.execute(
            """
            INSERT INTO user_profiles (
                visitor_id, emotion, worries, likes, keywords, round_count, updated_at, created_at
            )
            VALUES (%s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s::timestamptz, %s::timestamptz)
            ON CONFLICT (visitor_id) DO UPDATE SET
                emotion = EXCLUDED.emotion,
                worries = EXCLUDED.worries,
                likes = EXCLUDED.likes,
                keywords = EXCLUDED.keywords,
                round_count = EXCLUDED.round_count,
                updated_at = EXCLUDED.updated_at
            RETURNING emotion, worries, likes, keywords, round_count, updated_at
            """,
            (
                visitor_id,
                profile["emotion"],
                worries_json,
                likes_json,
                keywords_json,
                profile["roundCount"],
                now,
                now,
            ),
        ).fetchone()
        conn.commit()
    return _row_to_api(row)


def _get_profile_sqlite(visitor_id: str) -> Optional[dict[str, Any]]:
    with connect_sqlite() as conn:
        row = conn.execute(
            """
            SELECT emotion, worries, likes, keywords, round_count, updated_at
            FROM user_profiles
            WHERE visitor_id = ?
            """,
            (visitor_id,),
        ).fetchone()
    if not row:
        return None
    return _row_to_api(row)


def _upsert_profile_sqlite(visitor_id: str, profile: dict[str, Any]) -> dict[str, Any]:
    now = utc_iso_now()
    worries_json = json.dumps(profile["worries"], ensure_ascii=False)
    likes_json = json.dumps(profile["likes"], ensure_ascii=False)
    keywords_json = json.dumps(profile["keywords"], ensure_ascii=False)
    with connect_sqlite() as conn:
        existing = conn.execute(
            "SELECT created_at FROM user_profiles WHERE visitor_id = ?",
            (visitor_id,),
        ).fetchone()
        created_at = str(existing["created_at"]) if existing else now
        conn.execute(
            """
            INSERT INTO user_profiles (
                visitor_id, emotion, worries, likes, keywords, round_count, updated_at, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(visitor_id) DO UPDATE SET
                emotion = excluded.emotion,
                worries = excluded.worries,
                likes = excluded.likes,
                keywords = excluded.keywords,
                round_count = excluded.round_count,
                updated_at = excluded.updated_at
            """,
            (
                visitor_id,
                profile["emotion"],
                worries_json,
                likes_json,
                keywords_json,
                profile["roundCount"],
                now,
                created_at,
            ),
        )
        row = conn.execute(
            """
            SELECT emotion, worries, likes, keywords, round_count, updated_at
            FROM user_profiles
            WHERE visitor_id = ?
            """,
            (visitor_id,),
        ).fetchone()
        conn.commit()
    return _row_to_api(row)


def _get_profile(visitor_id: str) -> Optional[dict[str, Any]]:
    if db_url():
        return _get_profile_pg(visitor_id)
    return _get_profile_sqlite(visitor_id)


def _upsert_profile(visitor_id: str, profile: dict[str, Any]) -> dict[str, Any]:
    if db_url():
        return _upsert_profile_pg(visitor_id, profile)
    return _upsert_profile_sqlite(visitor_id, profile)


@bp.get("/profile")
def get_profile() -> Any:
    try:
        visitor_id = require_session_visitor()
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    try:
        row = _get_profile(visitor_id)
        if row:
            return jsonify(row)
        return jsonify(dict(_DEFAULT_PROFILE))
    except Exception:
        return jsonify({**_DEFAULT_PROFILE, "degraded": True}), 503


@bp.post("/profile")
def post_profile() -> Any:
    data = request.get_json(silent=True) or {}
    try:
        visitor_id = require_session_visitor()
        profile = _normalize_profile_fields(data)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    try:
        saved = _upsert_profile(visitor_id, profile)
        return jsonify(saved)
    except Exception:
        return jsonify({"error": "profile save failed", "degraded": True}), 503
