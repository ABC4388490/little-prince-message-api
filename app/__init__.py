from __future__ import annotations

import os

from flask import Flask
from flask_cors import CORS


def _load_local_env_file() -> None:
    api_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    env_path = os.path.join(api_root, ".env")
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


def create_app(*, skip_chat_blueprint: bool = False, skip_db_init: bool = False) -> Flask:
    from app.api import register_blueprints
    from app.db import init_databases

    if not skip_db_init:
        init_databases()

    app = Flask(__name__)

    cors_origins = os.environ.get("CORS_ORIGINS", "*").strip()
    origins: str | list[str] = "*"
    if cors_origins and cors_origins != "*":
        parsed_origins: list[str] = []
        for raw_origin in cors_origins.split(","):
            origin = raw_origin.strip().strip('"').strip("'").rstrip("/")
            if origin:
                parsed_origins.append(origin)
        if parsed_origins:
            origins = parsed_origins

    CORS(
        app,
        resources={r"/api/*": {"origins": origins}},
        supports_credentials=False,
    )

    register_blueprints(app, skip_chat=skip_chat_blueprint)
    return app
