"""Normalize `data.json` payload into chunk records (no ML)."""

from __future__ import annotations

from rag.chunk_meta import default_source_name_from_payload, source_name_from_dict


def normalize_records(payload: dict) -> list[dict[str, str]]:
    """Normalize legacy string chunks or dict chunks into chunk_id, source_name, text."""
    chunks = payload.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("data.json must contain a non-empty 'chunks' array")
    default_src = default_source_name_from_payload(payload)
    out: list[dict[str, str]] = []
    for c in chunks:
        if isinstance(c, str):
            t = c.strip()
            if not t:
                continue
            out.append({"chunk_id": f"kb_{len(out):03d}", "source_name": default_src, "text": t})
        elif isinstance(c, dict):
            t = str(c.get("text", "")).strip()
            if not t:
                continue
            cid = str(c.get("chunk_id") or "").strip()
            if not cid:
                cid = f"kb_{len(out):03d}"
            src = source_name_from_dict(c, default=default_src)
            out.append({"chunk_id": cid, "source_name": src, "text": t})
    if not out:
        raise ValueError("No non-empty chunks after normalization")
    return out
