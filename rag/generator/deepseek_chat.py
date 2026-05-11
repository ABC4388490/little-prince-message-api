"""
DeepSeek-compatible chat/completions client.

Independent of RAG; injectable for tests (mock `generate`).
"""

from __future__ import annotations

import json
import os
from typing import Sequence
from urllib import error as urlerror
from urllib import request as urlrequest

from rag.prompts.text import safe_message_text


class DeepSeekChatGenerator:
    def __init__(
        self,
        *,
        api_url: str | None = None,
        model: str | None = None,
        timeout: float = 20.0,
    ) -> None:
        self.api_url = (api_url or os.environ.get("DEEPSEEK_API_URL", "")).strip() or (
            "https://api.deepseek.com/v1/chat/completions"
        )
        self.model = (model or os.environ.get("DEEPSEEK_MODEL", "")).strip() or "deepseek-chat"
        self.timeout = timeout

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float = 0.8,
        output_char_limit: int = 220,
    ) -> str:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            return ""

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
            return ""

        payload = {
            "model": self.model,
            "temperature": temperature,
            "messages": base_messages,
        }
        req = urlrequest.Request(
            url=self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )

        try:
            with urlrequest.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                data = json.loads(body)
                content = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                return safe_message_text(content, limit=max(80, int(output_char_limit)))
        except (urlerror.URLError, TimeoutError, json.JSONDecodeError, KeyError, IndexError, ValueError):
            return ""
