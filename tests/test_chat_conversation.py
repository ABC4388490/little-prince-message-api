"""Transport acceptance with mocked model output; not a model quality evaluation."""
import os
from unittest.mock import patch

import pytest
from flask import Flask

from app.api.chat import bp
from app.services.agent_service import decide_strategy
from app.services.chat_pipeline import run_chat
from app.services.llm_service import ChatUnavailable, complete_chat
from app.services.rag_service import merge_rag


def turns(*contents):
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": text}
            for i, text in enumerate(contents)]


@pytest.mark.parametrize("text", ["你好呀，最近过得还好吗？", "上班好累", "为什么每天都要开会？", "人生好难", "今天吃了火锅，好开心"])
def test_daily_chat_never_forces_book_retrieval(text):
    assert not decide_strategy(text).use_rag


def test_book_followups_keep_topic_until_user_changes_it():
    history = turns("狐狸说的驯养是什么意思？", "建立关系。", "为什么呢？", "需要花时间。", "然后呢？")
    assert decide_strategy(history[-1]["content"], history).use_rag
    history += [{"role": "assistant", "content": "慢慢熟悉。"}, {"role": "user", "content": "今天上班好累"}]
    assert not decide_strategy(history[-1]["content"], history).use_rag
    history += [{"role": "assistant", "content": "事情很多吗？"}, {"role": "user", "content": "为什么呢？"}]
    assert not decide_strategy(history[-1]["content"], history).use_rag


def test_short_followup_retrieves_original_topic():
    history = turns("狐狸说的驯养是什么意思？", "建立关系。", "为什么呢？")
    with patch("app.services.rag_service.default_rag_orchestrator") as factory:
        orch = factory.return_value
        orch.retrieve_and_rank.return_value = []
        orch.build_prompt.return_value = ("", [])
        merge_rag(history)
        query = orch.retrieve_and_rank.call_args.args[0]
        assert "狐狸" in query and "为什么" in query


def test_plain_reply_preserves_context_and_has_no_analysis_format():
    history = turns("上班好累", "事情太多吗？", "不是，就是每天都一样", "每天的重复让你累。", "对，别给建议，陪我聊聊")
    with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-only"}), patch("app.services.llm_service.DeepSeekChatGenerator.generate", return_value="好，我们就聊聊。") as generate:
        result = run_chat(history)
    sent = generate.call_args.args[0]
    assert [m for m in sent if m["role"] != "system"] == history
    assert "JSON" not in sent[0]["content"]
    assert result["assistant"]["content"] == "好，我们就聊聊。"
    assert result["analysis"] == ""
    assert result["citations"] == []
    assert "小王子" in sent[0]["content"]


def test_missing_key_is_an_error_not_a_reply():
    with patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}):
        with pytest.raises(ChatUnavailable):
            complete_chat(turns("你好"), skip_rag=True, strict=True)


@pytest.mark.parametrize("reply", ["", "   "])
def test_empty_model_output_is_an_error(reply):
    with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-only"}), patch("app.services.llm_service.DeepSeekChatGenerator.generate", return_value=reply):
        with pytest.raises(ChatUnavailable):
            run_chat(turns("你好"))


def test_flask_failure_has_503_and_no_fake_assistant():
    app = Flask(__name__)
    app.register_blueprint(bp)
    with patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}):
        response = app.test_client().post("/api/chat", json={"messages": turns("你好")})
    assert response.status_code == 503
    assert response.json["code"] == "chat_unavailable"
    assert "assistant" not in response.json


def test_flask_success_and_bad_request():
    app = Flask(__name__)
    app.register_blueprint(bp)
    client = app.test_client()
    assert client.post("/api/chat", json={"messages": []}).status_code == 400
    with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-only"}), patch("app.services.llm_service.DeepSeekChatGenerator.generate", return_value="还不错，你呢？"):
        response = client.post("/api/chat", json={"messages": turns("最近好吗")})
    assert response.status_code == 200
    assert response.json["assistant"]["content"] == "还不错，你呢？"


def test_actual_network_failure_is_a_service_error():
    from urllib.error import URLError
    with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-only"}), patch("rag.generator.deepseek_chat.urlrequest.urlopen", side_effect=URLError("offline")):
        with pytest.raises(ChatUnavailable):
            run_chat(turns("你好"))


def test_fastapi_failure_returns_503():
    import asyncio
    from fastapi import HTTPException
    from asgi import ChatRequestBody, post_api_chat
    with patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}):
        with pytest.raises(HTTPException) as error:
            asyncio.run(post_api_chat(ChatRequestBody(messages=turns("你好"))))
    assert error.value.status_code == 503


def test_fastapi_success_returns_plain_reply():
    import asyncio
    from asgi import ChatRequestBody, post_api_chat
    with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-only"}), patch("app.services.llm_service.DeepSeekChatGenerator.generate", return_value="还不错，你呢？"):
        result = asyncio.run(post_api_chat(ChatRequestBody(messages=turns("最近好吗"))))
    assert result["assistant"]["content"] == "还不错，你呢？"
    assert result["analysis"] == ""
