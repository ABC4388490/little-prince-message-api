from __future__ import annotations

import os

from app import create_app
from app.db import connect_sqlite, init_sqlite


def test_profile_sqlite_roundtrip():
    os.environ["DATABASE_URL"] = ""
    init_sqlite()
    conn = connect_sqlite()
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    conn.close()
    assert "user_profiles" in tables

    app = create_app(skip_chat_blueprint=True, skip_db_init=True)
    client = app.test_client()
    session = client.post("/api/session").get_json()
    headers = {"Authorization": "Bearer " + session["sessionToken"]}

    r = client.get("/api/profile", headers=headers)
    assert r.status_code == 200
    assert r.get_json()["emotion"] == "neutral"
    assert r.get_json()["roundCount"] == 0

    r2 = client.post(
        "/api/profile",
        headers=headers,
        json={
            "emotion": "low",
            "worries": ["焦虑"],
            "likes": ["玫瑰"],
            "keywords": ["工作"],
            "roundCount": 3,
        },
    )
    assert r2.status_code == 200
    body = r2.get_json()
    assert body["emotion"] == "low"
    assert body["roundCount"] == 3
    assert "焦虑" in body["worries"]

    r3 = client.get("/api/profile", headers=headers)
    assert r3.status_code == 200
    assert r3.get_json()["roundCount"] == 3
