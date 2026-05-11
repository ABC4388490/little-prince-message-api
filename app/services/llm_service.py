from __future__ import annotations

import json
import logging
import os
import re
from copy import deepcopy
from typing import Any, Sequence

from rag.generator.deepseek_chat import DeepSeekChatGenerator
from rag.trace import rag_debug

logger = logging.getLogger(__name__)

STRUCTURED_JSON_SUFFIX = (
    "\n\n【输出格式】你必须只回复一段合法 UTF-8 JSON 对象，不要 markdown 代码围栏，不要前后解释文字。"
    '键仅包含 "conclusion"（1–2 句简短结论）与 "analysis"（2–6 句展开分析，可含比喻）。'
    '示例：{"conclusion":"……","analysis":"……"}'
)

SYSTEM_PROMPT = "\n".join(
    [
        "你是小王子的星球管家，住在 B612。",
        "你的语气温柔、童真、克制，像夜里轻声说话。",
        "请用富有画面感的比喻写 2-4 句短句，给用户一段带哲理的回应。",
        "结构建议：先共情，再给一小句启发，最后留一句柔和的提问。",
        "避免说教、避免口号、避免网络热词、避免 emoji。",
    ]
)

FALLBACK_REPLY = (
    "我听见了你的心事。就像黄昏里的风，不必一下子把答案带来，"
    "先把今天最在意的一件小事轻轻放在掌心里看看——"
    "它也许就是通往星光的第一步。"
)


def safe_message_text(text: str, limit: int = 1800) -> str:
    clean = " ".join(str(text or "").strip().split())
    return clean[:limit]


def _messages_to_base(messages: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    base_messages: list[dict[str, str]] = []
    for m in messages:
        role = str((m or {}).get("role", "")).strip()
        lim = 3200 if role == "system" else 900
        content = safe_message_text(str((m or {}).get("content", "")), limit=lim)
        if role not in ("system", "user", "assistant"):
            continue
        if not content:
            continue
        base_messages.append({"role": role, "content": content})
    if not base_messages:
        base_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    return base_messages


def _parse_conclusion_analysis(raw: str) -> tuple[str, str]:
    """Parse JSON conclusion/analysis; fallback on failure."""
    text = str(raw or "").strip()
    if not text:
        return "（暂无回复）", ""
    # strip optional ```json fences
    fence = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", text, re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            c = str(data.get("conclusion", "")).strip()
            a = str(data.get("analysis", "")).strip()
            if c or a:
                return c or "（见分析）", a or c
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    # extract first {...} substring
    m = re.search(r"\{[\s\S]*\"conclusion\"[\s\S]*\"analysis\"[\s\S]*\}", text)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, dict):
                c = str(data.get("conclusion", "")).strip()
                a = str(data.get("analysis", "")).strip()
                if c or a:
                    return c or "（见分析）", a or c
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    # fallback: first sentence conclusion, rest analysis
    parts = re.split(r"(?<=[。！？!?])\s*", text, maxsplit=1)
    if len(parts) >= 2 and parts[0]:
        return parts[0].strip(), parts[1].strip()
    return text[:120].strip() + ("…" if len(text) > 120 else ""), text


def complete_chat_structured(messages_after_rag: Sequence[dict[str, str]]) -> dict[str, Any]:
    """
    After merge_rag: ask model for JSON {conclusion, analysis}.
    Returns keys: conclusion, analysis, assistant_plain (merged for display/API compat).
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        fb = FALLBACK_REPLY
        return {"conclusion": "（未配置 DEEPSEEK_API_KEY）", "analysis": fb, "assistant_plain": fb}

    base = deepcopy(_messages_to_base(messages_after_rag))
    for i, m in enumerate(base):
        if m.get("role") == "system":
            base[i] = {"role": "system", "content": m["content"] + STRUCTURED_JSON_SUFFIX}
            break
    else:
        base.insert(0, {"role": "system", "content": SYSTEM_PROMPT + STRUCTURED_JSON_SUFFIX})

    sys_len = len(next((m["content"] for m in base if m["role"] == "system"), ""))
    rag_debug(logger, "llm structured call: messages=%s system_chars=%s", len(base), sys_len)

    raw = DeepSeekChatGenerator().generate(base, output_char_limit=1400)
    if not raw.strip():
        fb = FALLBACK_REPLY
        return {"conclusion": fb[:120], "analysis": fb, "assistant_plain": fb}

    conclusion, analysis = _parse_conclusion_analysis(raw)
    plain = conclusion
    if analysis:
        plain = f"{conclusion}\n\n{analysis}"
    return {"conclusion": conclusion, "analysis": analysis, "assistant_plain": plain}


def complete_chat(messages: Sequence[dict[str, str]], *, skip_rag: bool = False) -> str:
    """Call DeepSeek chat/completions. When skip_rag is False, merge RAG into messages first."""
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        return FALLBACK_REPLY

    msgs: list[dict[str, str]] = list(messages)
    if not skip_rag:
        from app.services.rag_service import merge_rag

        msgs, _ = merge_rag(msgs, use_rag=True)

    base_messages = _messages_to_base(msgs)

    sys_len = len(next((m["content"] for m in base_messages if m["role"] == "system"), ""))
    rag_debug(logger, "llm call: messages=%s system_chars=%s", len(base_messages), sys_len)

    reply = DeepSeekChatGenerator().generate(base_messages)
    return reply or FALLBACK_REPLY
