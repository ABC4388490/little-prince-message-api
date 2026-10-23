"""Normalize `data.json` payload into chunk records (no ML)."""

from __future__ import annotations

from typing import Any

from rag.chunk_meta import default_source_name_from_payload, source_name_from_dict

_EXTRA_KEYS = (
    "content",
    "chapter",
    "topic",
    "role",
    "chunk_type",
    "source_url",
)


def _derive_text(row: dict[str, Any]) -> str:
    explicit = str(row.get("text", "")).strip()
    if explicit:
        return explicit
    content = str(row.get("content", "")).strip()
    if content:
        return content
    return ""


def normalize_records(payload: dict) -> list[dict[str, Any]]:
    """Normalize legacy string chunks or rich dict chunks; preserve extended metadata."""
    chunks = payload.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("data.json must contain a non-empty 'chunks' array")
    default_src = default_source_name_from_payload(payload)
    out: list[dict[str, Any]] = []
    for c in chunks:
        if isinstance(c, str):
            t = c.strip()
            if not t:
                continue
            out.append({"chunk_id": f"kb_{len(out):03d}", "source_name": default_src, "text": t})
        elif isinstance(c, dict):
            t = _derive_text(c)
            if not t:
                continue
            cid = str(c.get("chunk_id") or "").strip()
            if not cid:
                cid = f"kb_{len(out):03d}"
            src = source_name_from_dict(c, default=default_src)
            row: dict[str, Any] = {
                "chunk_id": cid,
                "source_name": src,
                "text": t,
            }
            for key in _EXTRA_KEYS:
                if key in c and c[key] is not None:
                    row[key] = c[key]
            out.append(row)
    if not out:
        raise ValueError("No non-empty chunks after normalization")
    return out
