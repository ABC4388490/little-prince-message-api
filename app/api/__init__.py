from __future__ import annotations

from flask import Flask


def register_blueprints(app: Flask, *, skip_chat: bool = False) -> None:
    # Memory endpoints do not need to load the optional RAG/model stack.
    from app.api.conversations import bp as conversations_bp
    from app.api.health import bp as health_bp
    from app.api.messages import bp as messages_bp
    from app.api.profile import bp as profile_bp
    from app.api.session import bp as session_bp

    app.register_blueprint(health_bp)
    app.register_blueprint(messages_bp)
    app.register_blueprint(conversations_bp)
    app.register_blueprint(profile_bp)
    app.register_blueprint(session_bp)
    if not skip_chat:
        from app.api.chat import bp as chat_bp

        app.register_blueprint(chat_bp)
