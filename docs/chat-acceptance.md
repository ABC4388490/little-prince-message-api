# 聊天修正与验收

日常聊天优先直接接话，保留最近双方对话；只有原著问题及连续追问才检索。/api/chat 返回一条自然语言回复，analysis 兼容字段为空；/api/analyze 继续提供结构化解读。缺少密钥、调用失败或空结果返回 503，前端应显示重试。

部署时在实际后端服务设置 DEEPSEEK_API_KEY（只通过环境变量注入）、DEEPSEEK_MODEL=deepseek-flash、DEEPSEEK_API_URL=https://api.deepseek.com/chat/completions。密钥不得提交。

自动化：`python -m pytest tests/test_chat_conversation.py tests/test_rag_prompt.py -q`。19 项检查包含上下文、检索路由、Flask/FastAPI、缺少密钥、断网和空模型输出。

真实对话：`python scripts/validate_chat_conversation.py`。使用 tests/fixtures/chat_acceptance_cases.json 的 8 组连续 3 轮，共24轮，输出 work/chat-acceptance/live-results.json。该脚本采集输出，不自动宣称自然度通过。

每组按接话准确、上下文连贯、日常表达、尊重偏好、角色与真实边界五项各0–2分评审；每组至少8分且任何单项不得为0。固定兜底冒充模型、记错明确事实、拒绝后继续强推建议，任一项即不通过。

本地使用真实模型进行了三轮24轮对话采样；最终一轮已经完成，仍需正式环境接口和浏览器验收。临时本地环境没有FAISS，原著两轮发生检索降级，不能视为原著引用链验收通过。
