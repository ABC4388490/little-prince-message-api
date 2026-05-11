"""
Shared setup + one RAG turn for demos and validation scripts.

Run all entry points with cwd = ``message-api`` (``prepare_demo_environment()`` handles this).
"""

from __future__ import annotations

import os
import sys
from typing import Any

DEFAULT_VALIDATION_QUERIES: list[str] = [
    "小王子和玫瑰之间发生了什么？",
    "狐狸说的驯服是什么意思？",
    "真正重要的东西为什么用眼睛看不见？",
    "点灯人为什么让小王子尊敬？",
    "沙漠里藏着什么希望？",
    "地理学家为什么不亲自探险？",
    "商人数星星的故事说明什么？",
]


def api_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def load_local_env() -> None:
    env_path = os.path.join(api_root(), ".env")
    if not os.path.isfile(env_path):
        return
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            with open(env_path, "r", encoding=enc) as f:
                raw = f.read()
            break
        except OSError:
            continue
    else:
        return
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def prepare_demo_environment() -> str:
    """``chdir`` to message-api, ensure import path, load ``.env``."""
    root = api_root()
    os.chdir(root)
    if root not in sys.path:
        sys.path.insert(0, root)
    load_local_env()
    return root


def citations_schema_ok(citations: list[Any]) -> bool:
    if not citations:
        return True
    return all(
        isinstance(c, dict) and bool(c.get("chunk_id")) and c.get("source_name") is not None
        for c in citations
    )


def run_turn(user_text: str, *, use_rag: bool = True) -> dict[str, Any]:
    """Single user message → merged messages path used by ``/api/chat``."""
    from app.services.llm_service import SYSTEM_PROMPT, complete_chat
    from app.services.rag_service import merge_rag

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    msgs, citations = merge_rag(messages, use_rag=use_rag)
    reply = complete_chat(msgs, skip_rag=True)
    return {
        "query": user_text,
        "citations": citations,
        "answer": reply,
        "citation_schema_ok": citations_schema_ok(citations),
    }
