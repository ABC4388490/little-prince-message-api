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

_TONE_EMOTION = "回应对方明确说出的处境，不凭空猜心事；不急着建议，不强加比喻和结尾提问。"
_TONE_PHILOSOPHY = "先直接回答问题，需要时解释，不用隐喻或反问代替回答。"
_TONE_GENERAL = "像朋友自然接话，问候就回应问候，不把闲聊变成心理辅导。"


def is_book_query(text: str, messages: Sequence[dict[str, str]] = ()) -> bool:
    import re
    book = r"原著|这本书|书[里中]|第.{0,5}章|驯[养服]|点灯人|地理学家|(?:小王子|狐狸|玫瑰|国王|酒鬼|商人|蛇).{0,16}(?:说|为什么|为何|意思|象征|离开)|(?:为什么|为何|怎么).{0,16}(?:狐狸|玫瑰|小王子)"
    if re.search(book, text):
        return True
    if re.match(r"^(为什么[呢呀啊]?[？?]?$|然后呢|后来呢|那[它他她]|这句话|继续讲|再解释)", text):
        previous = [m for m in messages if m.get("role") == "user"][:-1]
        # Walk through short follow-ups; stop at a real topic change.
        for m in reversed(previous):
            content = m.get("content", "").strip()
            if re.search(book, content):
                return True
            if not re.match(r"^(为什么[呢呀啊]?[？?]?$|然后呢|后来呢|那[它他她]|这句话|继续讲|再解释)", content):
                break
    return False


@dataclass(frozen=True)
class AgentDecision:
    strategy: str
    use_rag: bool
    tone_system_append: str


def decide_strategy(user_input: str, messages: Sequence[dict[str, str]] = ()) -> AgentDecision:
    t = (user_input or "").strip()
    if not t:
        return AgentDecision(STRATEGY_GENERAL, False, _TONE_GENERAL)

    if any(k in t for k in _EMOTION_KWS):
        return AgentDecision(STRATEGY_EMOTION, is_book_query(t, messages), _TONE_EMOTION)

    if any(k in t for k in _PHILOSOPHY_KWS):
        return AgentDecision(STRATEGY_PHILOSOPHY, is_book_query(t, messages), _TONE_PHILOSOPHY)

    return AgentDecision(STRATEGY_GENERAL, is_book_query(t, messages), _TONE_GENERAL)


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

    # The server owns the character; client context is supplemental.
    msgs.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
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
