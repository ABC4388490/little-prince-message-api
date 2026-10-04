from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Optional

try:
    import psycopg
except Exception:  # pragma: no cover
    psycopg = None  # type: ignore

_API_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def api_root() -> str:
    return _API_ROOT


def utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def db_url() -> str:
    return os.environ.get("DATABASE_URL", "").strip()


def db_path() -> str:
    return os.path.join(_API_ROOT, "messages.db")


class ClosingSQLiteConnection(sqlite3.Connection):
    def __exit__(self, exc_type, exc, tb):
        try:
            return super().__exit__(exc_type, exc, tb)
        finally:
            self.close()


def connect_sqlite() -> sqlite3.Connection:
    conn = sqlite3.connect(db_path(), factory=ClosingSQLiteConnection)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
    names = {str(c["name"]) for c in cols}
    if column not in names:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def init_sqlite() -> None:
    with connect_sqlite() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                posX REAL NOT NULL,
                posY REAL NOT NULL,
                createdAt TEXT NOT NULL
            )
            """
        )
        _ensure_column(conn, "messages", "reply", "TEXT")
        _ensure_column(conn, "messages", "replyCreatedAt", "TEXT")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                visitor_id TEXT NOT NULL UNIQUE,
                emotion TEXT NOT NULL DEFAULT 'neutral',
                worries TEXT NOT NULL DEFAULT '[]',
                likes TEXT NOT NULL DEFAULT '[]',
                keywords TEXT NOT NULL DEFAULT '[]',
                round_count INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS visitor_sessions (
                token_hash TEXT PRIMARY KEY,
                visitor_id TEXT NOT NULL UNIQUE,
                expires_at BIGINT NOT NULL
            )
            """
        )
        conn.commit()


def connect_pg():
    if not db_url():
        raise RuntimeError("DATABASE_URL is not set")
    if psycopg is None:
        raise RuntimeError("psycopg is not installed")
    return psycopg.connect(db_url())


def init_postgres() -> None:
    if not db_url():
        return
    with connect_pg() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id BIGSERIAL PRIMARY KEY,
                visitor_id TEXT NOT NULL UNIQUE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages_v2 (
                id BIGSERIAL PRIMARY KEY,
                conversation_id BIGINT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                pos_x REAL,
                pos_y REAL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )
        conn.execute("ALTER TABLE messages_v2 ADD COLUMN IF NOT EXISTS pos_x REAL;")
        conn.execute("ALTER TABLE messages_v2 ADD COLUMN IF NOT EXISTS pos_y REAL;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_profiles (
                id BIGSERIAL PRIMARY KEY,
                visitor_id TEXT NOT NULL UNIQUE,
                emotion TEXT NOT NULL DEFAULT 'neutral'
                    CHECK (emotion IN ('neutral', 'low', 'high')),
                worries JSONB NOT NULL DEFAULT '[]'::jsonb,
                likes JSONB NOT NULL DEFAULT '[]'::jsonb,
                keywords JSONB NOT NULL DEFAULT '[]'::jsonb,
                round_count INTEGER NOT NULL DEFAULT 0
                    CHECK (round_count >= 0),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_user_profiles_visitor_id ON user_profiles (visitor_id);"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS visitor_sessions (
                token_hash TEXT PRIMARY KEY,
                visitor_id TEXT NOT NULL UNIQUE,
                expires_at BIGINT NOT NULL
            )
            """
        )
        conn.commit()


def init_databases() -> None:
    init_sqlite()
    init_postgres()


def parse_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        f = float(value)
        if f != f:
            return None
        if f in (float("inf"), float("-inf")):
            return None
        return f
    except Exception:
        return None
