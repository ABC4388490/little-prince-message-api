from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Message:
    id: int
    content: str
    posX: float
    posY: float
    reply: str
    createdAt: str
    replyCreatedAt: str
