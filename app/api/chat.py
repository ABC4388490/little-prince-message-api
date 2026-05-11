from __future__ import annotations

from typing import Any

from flask import Blueprint, jsonify, request

from app.api.common import require_messages_json
from app.services.analyze_pipeline import run_analyze
from app.services.chat_pipeline import run_chat

bp = Blueprint("chat", __name__, url_prefix="/api")


@bp.post("/chat")
def chat_messages() -> Any:
    try:
        messages = require_messages_json(request)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    out = run_chat(messages)
    return jsonify(out), 200


@bp.post("/analyze")
def analyze_reply() -> Any:
    data = request.get_json(silent=True) or {}
    text = str(data.get("text") or data.get("lastReply") or data.get("reply") or "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400
    try:
        out = run_analyze(text)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify(out), 200
