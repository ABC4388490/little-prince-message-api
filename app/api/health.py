from __future__ import annotations

from typing import Any

from flask import Blueprint, jsonify

from app.db import db_url

bp = Blueprint("health", __name__)


@bp.get("/health")
def health() -> Any:
    return jsonify({"ok": True, "service": "little-prince-memory", "memoryAuthVersion": 1,
                    "memoryStore": "postgres" if db_url() else "sqlite"})
