"""
Lightweight strategy router for /api/chat (no LangChain).

decide_strategy(user_input) -> emotion | philosophy | general,
each with use_rag flag and a short tone block to merge into system prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.services.llm_service import SYSTEM_PROMPT, safe_message_text

STRATEGY_EMOTION = "emotion"
STRATEGY_PHILOSOPHY = "philosophy"
STRATEGY_GENERAL = "general"

_EMOTION_KWS: tuple[str, ...] = (
    "孤独",
    "孤单",
    "寂寞",
    "难过",
    "伤心",
    "想哭",
    "没人懂",
    "好累",
    "抑郁",
    "崩溃",
    "好难受",
    "心里空",
    "被忽略",
)

_PHILOSOPHY_KWS: tuple[str, ...] = (
    "意义",
    "为什么",
    "人生",
    "存在",
    "价值",
    "到底",
    "本质",
    "活着",
    "宇宙",
    "命运",
)

_TONE_EMOTION = (
    "【语气-温柔】请以陪伴为先：语气更轻、更暖，先接住对方的感受，少讲道理、少给建议；"
    "可用一两句意象（星星、风、夜）呼应，结尾用很轻的问句留白，不要列清单。"
)

_TONE_PHILOSOPHY = (
    "【语气-哲理】请在温柔中略带思辨：可用隐喻与留白，不必下结论；"
    "避免说教与口号，避免长段排比；可轻轻反问，引导对方自己摸到一个方向。"
)

_TONE_GENERAL = (
    "【语气-普通】像朋友自然聊天：回答简洁、口语一点，不必强行诗意；"
    "若对方只是打招呼或闲聊，可同样轻松回应。"
)


@dataclass(frozen=True)
class AgentDecision:
    strategy: str
    use_rag: bool
    tone_system_append: str


def decide_strategy(user_input: str) -> AgentDecision:
    t = (user_input or "").strip()
    if not t:
        return AgentDecision(STRATEGY_GENERAL, False, _TONE_GENERAL)

    if any(k in t for k in _EMOTION_KWS):
        return AgentDecision(STRATEGY_EMOTION, True, _TONE_EMOTION)

    if any(k in t for k in _PHILOSOPHY_KWS):
        return AgentDecision(STRATEGY_PHILOSOPHY, True, _TONE_PHILOSOPHY)

    return AgentDecision(STRATEGY_GENERAL, False, _TONE_GENERAL)


def merge_agent_tone(messages: Sequence[dict[str, str]], decision: AgentDecision) -> list[dict[str, str]]:
    """Merge tone into first system (or insert SYSTEM_PROMPT + tone). Call before RAG."""
    msgs: list[dict[str, str]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", "")).strip()
        content = str(m.get("content", ""))
        if role not in ("system", "user", "assistant"):
            continue
        msgs.append({"role": role, "content": content})

    tone = safe_message_text(decision.tone_system_append, limit=400)
    if not tone:
        return msgs

    first_sys = next((i for i, m in enumerate(msgs) if m.get("role") == "system"), None)
    if first_sys is not None:
        merged = msgs[first_sys]["content"] + "\n\n" + tone
        msgs[first_sys] = {"role": "system", "content": safe_message_text(merged, limit=1800)}
    else:
        msgs.insert(
            0,
            {"role": "system", "content": safe_message_text(SYSTEM_PROMPT + "\n\n" + tone, limit=1800)},
        )
    return msgs
