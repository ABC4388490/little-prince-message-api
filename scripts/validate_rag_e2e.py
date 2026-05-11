#!/usr/bin/env python3
"""
Batch RAG + LLM validation (run from ``message-api``).

  python scripts/validate_rag_e2e.py
  python scripts/validate_rag_e2e.py --quick

Uses ``demo_core.run_turn`` (same path as ``/api/chat``). Set ``RAG_DEBUG=1`` for short ``[rag]`` logs.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys


def _api_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG + LLM batch validation")
    parser.add_argument("--quick", action="store_true", help="only first 2 queries")
    args = parser.parse_args()

    root = _api_root()
    os.chdir(root)
    if root not in sys.path:
        sys.path.insert(0, root)

    log_level = logging.INFO if os.environ.get("RAG_DEBUG", "").strip().lower() in ("1", "true", "yes", "on") else logging.WARNING
    logging.basicConfig(level=log_level, format="%(levelname)s %(name)s %(message)s")
    for noisy in ("sentence_transformers", "httpx", "httpcore", "faiss.loader", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    from demo_core import DEFAULT_VALIDATION_QUERIES, prepare_demo_environment, run_turn

    prepare_demo_environment()

    queries = DEFAULT_VALIDATION_QUERIES[:2] if args.quick else DEFAULT_VALIDATION_QUERIES
    results = []
    for q in queries:
        out = run_turn(q, use_rag=True)
        if out.get("citations") and not out.get("citation_schema_ok"):
            logging.error("invalid citation schema for query %r", q)
        results.append(
            {
                "query": out["query"],
                "citation_count": len(out.get("citations") or []),
                "citations": [
                    {
                        "label": c.get("label"),
                        "chunk_id": c.get("chunk_id"),
                        "source_name": c.get("source_name"),
                        "text_preview": (c.get("text") or "")[:120],
                    }
                    for c in (out.get("citations") or [])
                ],
                "answer": out.get("answer"),
                "citation_schema_ok": out.get("citation_schema_ok", True),
            }
        )

    sys.stdout.reconfigure(encoding="utf-8") if hasattr(sys.stdout, "reconfigure") else None
    print(json.dumps({"ok": True, "runs": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
