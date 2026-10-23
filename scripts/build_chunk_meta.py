"""
Build rag/chunk_meta.json and rag/data.json from rag/sources/*.json.

Run from message-api:
    .venv\\Scripts\\python.exe scripts/build_chunk_meta.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "rag" / "sources"
CHUNK_META_PATH = ROOT / "rag" / "chunk_meta.json"
DATA_PATH = ROOT / "rag" / "data.json"


def _quotes_block(quotes: list[str]) -> str:
    parts = [q.strip() for q in quotes if str(q).strip()]
    if not parts:
        return "（本节要点以叙述为主。）"
    return "；".join(parts)


def build_content(quotes: list[str], gist: str) -> str:
    if not quotes:
        return f"【情节梗概，非逐字引文】{gist.strip()}"
    return f"【原文】{_quotes_block(quotes)}\n【释义】{gist.strip()}"


def build_text(
    *,
    chapter: str,
    topic: list[str],
    role: list[str],
    content: str,
    chunk_type: str,
    title: str | None = None,
) -> str:
    """Embedding text: roles/topics first for better entity recall."""
    topic_s = " ".join(topic) if topic else ""
    role_s = " ".join(role) if role else ""
    kw = " ".join(x for x in (role_s, topic_s, title or "") if x).strip()
    head = f"{kw} | {chapter}" if kw else chapter
    return f"{head} | {chunk_type} | {content}"


def load_json(name: str) -> dict[str, Any]:
    path = SOURCES / name
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_passage_chunks(chapters_payload: dict) -> list[dict[str, Any]]:
    default_src = str(chapters_payload.get("default_source_name") or "little_prince_kb")
    records: list[dict[str, Any]] = []
    for ch in chapters_payload.get("chapters", []):
        num = int(ch["num"])
        title = str(ch["title"])
        chapter_label = title
        for idx, p in enumerate(ch.get("passages", []), start=1):
            quotes = p.get("quotes") or []
            gist = str(p.get("gist", "")).strip()
            topic = list(p.get("topic") or [])
            role = list(p.get("role") or [])
            content = build_content(quotes, gist)
            chunk_id = f"passage_{num:02d}_{idx:02d}"
            text = build_text(
                chapter=chapter_label,
                topic=topic,
                role=role,
                content=content,
                chunk_type="passage",
            )
            records.append(
                {
                    "chunk_id": chunk_id,
                    "chunk_type": "passage",
                    "chapter": chapter_label,
                    "content": content,
                    "topic": topic,
                    "role": role,
                    "source_name": default_src,
                    "text": text,
                }
            )
    return records


def build_analysis_chunks(analysis_payload: dict) -> list[dict[str, Any]]:
    default_src = str(analysis_payload.get("default_source_name") or "little_prince_kb")
    records: list[dict[str, Any]] = []

    for item in analysis_payload.get("characters", []):
        slug = str(item.get("slug", "char"))
        title = str(item.get("title", ""))
        body = str(item.get("content", "")).strip()
        topic = list(item.get("topic") or [])
        role = list(item.get("role") or [])
        chapter = str(item.get("related_chapters") or "跨章")
        content = f"【阅读解读·角色·{title}】\n{body}"
        chunk_id = f"char_{slug}_01"
        text = build_text(
            chapter=chapter,
            topic=topic,
            role=role,
            content=content,
            chunk_type="character",
            title=title,
        )
        records.append(
            {
                "chunk_id": chunk_id,
                "chunk_type": "character",
                "chapter": chapter,
                "content": content,
                "topic": topic,
                "role": role,
                "source_name": default_src,
                "text": text,
            }
        )

    for item in analysis_payload.get("themes", []):
        slug = str(item.get("slug", "theme"))
        title = str(item.get("title", ""))
        body = str(item.get("content", "")).strip()
        topic = list(item.get("topic") or [])
        role = list(item.get("role") or [])
        chapter = str(item.get("related_chapters") or "跨章")
        content = f"【阅读解读·主题·{title}】\n{body}"
        chunk_id = f"theme_{slug}_01"
        text = build_text(
            chapter=chapter,
            topic=topic,
            role=role,
            content=content,
            chunk_type="theme",
            title=title,
        )
        records.append(
            {
                "chunk_id": chunk_id,
                "chunk_type": "theme",
                "chapter": chapter,
                "content": content,
                "topic": topic,
                "role": role,
                "source_name": default_src,
                "text": text,
            }
        )

    return records


def main() -> int:
    chapters_payload = load_json("chapters.json")
    analysis_payload = load_json("analysis.json")
    records = build_passage_chunks(chapters_payload) + build_analysis_chunks(analysis_payload)
    for record in records:
        record["source_url"] = "https://gutenberg.net.au/ebooks03/0300771h.html"

    if len(records) < 100:
        print(f"ERROR: only {len(records)} chunks (need >= 100)", file=sys.stderr)
        return 1

    CHUNK_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNK_META_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    data_payload = {
        "default_source_name": "little_prince_reading_notes",
        "chunks": records,
    }
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data_payload, f, ensure_ascii=False, indent=2)

    by_type: dict[str, int] = {}
    for r in records:
        t = r.get("chunk_type", "?")
        by_type[t] = by_type.get(t, 0) + 1

    print(f"Wrote {len(records)} chunks -> {CHUNK_META_PATH}")
    print(f"  types: {by_type}")
    print(f"  data.json -> {DATA_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
