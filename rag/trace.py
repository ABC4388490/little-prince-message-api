"""Optional RAG diagnostics: set RAG_DEBUG=1 for short INFO lines (default: silent)."""

from __future__ import annotations

import logging
import os

_TRACE_ENV = ("1", "true", "yes", "on")


def is_rag_trace() -> bool:
    return os.environ.get("RAG_DEBUG", "").strip().lower() in _TRACE_ENV


def rag_debug(logger: logging.Logger, fmt: str, *args: object) -> None:
    """One-line diagnostic; only when RAG_DEBUG is enabled (no DEBUG-level spam by default)."""
    if not is_rag_trace():
        return
    try:
        msg = fmt % args if args else fmt
    except TypeError:
        msg = fmt + (" " + str(args) if args else "")
    logger.info("[rag] %s", msg)


def trunc_preview(s: str, n: int = 160) -> str:
    """Truncate for user-visible strings (not used in logs by default)."""
    t = " ".join(str(s or "").replace("\n", " ").strip().split())
    return t[:n] + ("…" if len(t) > n else "")
