"""
ASGI entry: FastAPI owns POST /api/chat; Flask WSGI app mounted for all other routes.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from app import create_app
from app.api.common import validate_messages_list
from app.db import init_databases
from app.services.analyze_pipeline import run_analyze
from app.services.chat_pipeline import run_chat
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.middleware.wsgi import WSGIMiddleware


def _cors_allow_origins() -> list[str]:
    raw = os.environ.get("CORS_ORIGINS", "*").strip()
    if not raw or raw == "*":
        return ["*"]
    out: list[str] = []
    for part in raw.split(","):
        o = part.strip().strip('"').strip("'").rstrip("/")
        if o:
            out.append(o)
    return out or ["*"]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_databases()
    yield


app = FastAPI(lifespan=lifespan, title="little-prince-message-api")

# 显式 CORS_ORIGINS 时易漏本地端口；用 regex 放行 localhost / 127.0.0.1 任意端口，避免 OPTIONS 预检 400
_LOCAL_DEV_ORIGIN_REGEX = r"^https?://(127\.0\.0\.1|localhost)(:\d+)?$"

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_allow_origins(),
    allow_origin_regex=_LOCAL_DEV_ORIGIN_REGEX,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequestBody(BaseModel):
    messages: list[dict[str, Any]] = Field(default_factory=list)


@app.post("/api/chat")
async def post_api_chat(body: ChatRequestBody):
    try:
        msgs = validate_messages_list(body.messages)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return run_chat(msgs)


class AnalyzeBody(BaseModel):
    """Interpretation-only call: AI reply text to interpret."""

    text: str = ""
    reply: str = ""
    lastReply: str = ""


@app.post("/api/analyze")
async def post_api_analyze(body: AnalyzeBody):
    effective = str(body.text or body.lastReply or body.reply or "").strip()
    if not effective:
        raise HTTPException(status_code=400, detail="text or lastReply or reply is required")
    try:
        return run_analyze(effective)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


_flask_app = create_app(skip_chat_blueprint=True, skip_db_init=True)
app.mount("/", WSGIMiddleware(_flask_app))
