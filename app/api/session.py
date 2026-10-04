from flask import Blueprint, jsonify, request

from app.auth import create_session

bp = Blueprint("session", __name__, url_prefix="/api")


@bp.after_request
def private_response(response):
    response.headers["Cache-Control"] = "no-store"
    return response


@bp.post("/session")
def new_session():
    # An old browser-generated ID is not proof of ownership of existing data.
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict) or "visitorId" in data or "visitorId" in request.args or request.headers.get("X-Visitor-Id"):
        return jsonify({"error": "legacy visitor IDs cannot claim a session"}), 400
    return jsonify(create_session()), 201
