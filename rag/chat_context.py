"""Small verified chapter summaries for conversational follow-ups only."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag.config import rerank_output_top_k
from rag.retriever.session import rank_lexical_rows

CHAT_PASSAGES_PATH = Path(__file__).with_name("chat_passages.json")


def supplement_chat_context(query: str, selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    with CHAT_PASSAGES_PATH.open(encoding="utf-8") as source:
        supplemental = json.load(source)
    # Supplement only relevant material; preserve the original ranking when no
    # verified summary matches. The vector index and strict analysis stay intact.
    matches = rank_lexical_rows(query, supplemental, rerank_output_top_k())
    if not matches:
        return selected
    by_id = {row.get("chunk_id"): row for row in [*selected, *matches]}
    return rank_lexical_rows(query, list(by_id.values()), rerank_output_top_k())
