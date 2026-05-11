from __future__ import annotations

from flask import Flask

from app.api.chat import bp as chat_bp
from app.api.conversations import bp as conversations_bp
from app.api.health import bp as health_bp
from app.api.messages import bp as messages_bp


def register_blueprints(app: Flask, *, skip_chat: bool = False) -> None:
    app.register_blueprint(health_bp)
    app.register_blueprint(messages_bp)
    app.register_blueprint(conversations_bp)
    if not skip_chat:
        app.register_blueprint(chat_bp)
