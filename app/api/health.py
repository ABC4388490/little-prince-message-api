from __future__ import annotations

from typing import Any

from flask import Blueprint, jsonify

bp = Blueprint("health", __name__)


@bp.get("/health")
def health() -> Any:
    return jsonify({"ok": True})
