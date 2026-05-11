#!/usr/bin/env python3
"""
B612 local demo — one question through RAG + LLM.

Usage (from ``message-api`` directory)::

    python run_demo.py
    python run_demo.py "狐狸说的驯服是什么意思？"
    python run_demo.py --json "小王子和玫瑰"
    python run_demo.py --no-rag "今天有点累"

Options:
    --json      machine-readable output (UTF-8)
    --no-rag    skip retrieval (only style system prompt)
    --quiet     no decorative banner (good with --json)

Diagnostics: set ``RAG_DEBUG=1`` for a few short ``[rag]`` log lines.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys


def _silence_noisy_loggers() -> None:
    if os.environ.get("RAG_DEBUG", "").strip().lower() in ("1", "true", "yes", "on"):
        return
    for name in ("sentence_transformers", "httpx", "httpcore", "faiss.loader", "huggingface_hub", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)


def main() -> int:
    parser = argparse.ArgumentParser(description="B612 RAG + LLM demo")
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="user question (default: built-in sample about 狐狸/驯服)",
    )
    parser.add_argument("--json", action="store_true", help="print JSON only")
    parser.add_argument("--no-rag", action="store_true", help="disable retrieval")
    parser.add_argument("--quiet", action="store_true", help="minimal output")
    args = parser.parse_args()

    from demo_core import prepare_demo_environment, run_turn

    prepare_demo_environment()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    _silence_noisy_loggers()

    default_q = "狐狸说的驯服是什么意思？"
    question = (args.question or default_q).strip()
    if not question:
        print("Error: empty question.", file=sys.stderr)
        return 2

    out = run_turn(question, use_rag=not args.no_rag)

    if args.json:
        sys.stdout.reconfigure(encoding="utf-8") if hasattr(sys.stdout, "reconfigure") else None
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0 if out.get("citation_schema_ok", True) else 1

    if not args.quiet:
        print("=" * 48)
        print(" B612 demo — RAG + DeepSeek")
        print("=" * 48)
        print()

    print("问：", out["query"])
    print()
    cites = out.get("citations") or []
    if cites:
        print("引用来源（本次注入模型的片段）：")
        for c in cites:
            lab = c.get("label", "")
            cid = c.get("chunk_id", "")
            src = c.get("source_name", "")
            prev = (c.get("text") or "")[:100]
            print(f"  [{lab}] {cid} @ {src}")
            if prev:
                print(f"      …{prev}…")
        print()
    elif not args.no_rag:
        print("（未检索到片段：请确认已运行 python -m rag.embed）")
        print()

    print("答：")
    print(out.get("answer") or "")
    print()

    if not args.quiet:
        print("—")
        print("提示: python run_demo.py --json \"…\"  # 结构化输出")
        print("      RAG_DEBUG=1 python run_demo.py  # 简短阶段日志")

    return 0 if out.get("citation_schema_ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
