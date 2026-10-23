"""A waiting upstream model must not block other visitors or the event loop."""
import asyncio
import threading
from unittest.mock import patch

from asgi import ChatRequestBody, post_api_chat


def test_two_visitors_can_wait_for_models_concurrently():
    async def exercise():
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        release = threading.Event()
        lock = threading.Lock()
        count = 0

        def waiting_model(messages):
            nonlocal count
            with lock:
                count += 1
                if count == 2:
                    loop.call_soon_threadsafe(entered.set)
            assert release.wait(2), 'The event loop was blocked by a model call'
            return {'assistant': {'content': messages[-1]['content']}}

        body = lambda text: ChatRequestBody(messages=[{'role': 'user', 'content': text}])
        with patch('asgi.run_chat', side_effect=waiting_model):
            first = asyncio.create_task(post_api_chat(body('第一位访客')))
            second = asyncio.create_task(post_api_chat(body('第二位访客')))
            try:
                await asyncio.wait_for(entered.wait(), 1)
                assert not first.done() and not second.done()
            finally:
                release.set()
            replies = await asyncio.gather(first, second)
        assert [r['assistant']['content'] for r in replies] == ['第一位访客', '第二位访客']

    asyncio.run(exercise())
