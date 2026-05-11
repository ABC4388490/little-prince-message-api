"""Small text helpers (keeps prompts independent of Flask app)."""


def safe_message_text(text: str, limit: int = 1800) -> str:
    clean = " ".join(str(text or "").strip().split())
    return clean[:limit]
