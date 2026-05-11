# message-api（投稿后端）

前端 `index.html` 会请求：

- 旧版“星星留言墙”（可选保留）：
  - `GET http://localhost:5000/api/messages`
  - `POST http://localhost:5000/api/messages`
- 新版“B612 连续对话（每个访问者独立记忆）”：
  - `GET  http://localhost:5000/api/conversations/me?visitorId=...`
  - `GET  http://localhost:5000/api/conversations/<conversationId>/messages`
  - `POST http://localhost:5000/api/conversations/<conversationId>/messages`
- OpenAI 风格对话（带 Agent + 可选 RAG；**生产默认由 FastAPI 提供**，返回含 `conclusion` / `analysis` / `citations`）：
  - `POST http://localhost:5000/api/chat`
- 仅解读一段「小王子回信」文本（专用 RAG + 温柔解读，**不经 Agent 策略**；返回同上结构，`strategy` 为 `interpret`）：
  - `POST http://localhost:5000/api/analyze`，JSON：`{"text":"……"}`（兼容 `lastReply` / `reply`）

本目录提供一个轻量可运行的后端（**ASGI：FastAPI + 挂载 Flask**），工程化目录如下：

- `asgi.py`：**推荐**本地与部署入口（`uvicorn asgi:app`）；`POST /api/chat` 与 **`POST /api/analyze`** 走 FastAPI，其余路由走原 Flask
- `wsgi.py`：纯 Flask 入口（`gunicorn wsgi:app`），仍可用于不装 FastAPI 的旧部署
- `run_demo.py`：本地一条对话演示（RAG + LLM）；`demo_core.py` 为脚本共享逻辑
- `app/`：应用包
  - `api/`：HTTP 路由（Blueprint）
  - `services/`：`llm_service`、`rag_service`、`agent_service`、`analyze_pipeline`（`/api/analyze`）
  - `models/`：数据类（如 `Message`）
  - `db.py`：SQLite / Postgres 初始化与连接
- `rag/`：RAG 子系统（模块化，可单测）
  - `rag/config.py`：路径与环境变量
  - `rag/corpus.py`：`data.json` → 规范化记录
  - `rag/embedding/`：句向量 bi-encoder（`SentenceBiEncoder`）
  - `rag/vectorstore/`：FAISS IP 索引读写与检索（`FaissIPVectorStore`）
  - `rag/retriever/`：查询编码 + 向量库命中（`FaissRetriever` + `session` 懒加载）
  - `rag/reranker/`：CrossEncoder 重排（`CrossEncoderReranker`）
  - `rag/prompts/`：grounding 文本与 `citations` 列表（`RAGPromptBuilder`）
  - `rag/generator/`：DeepSeek Chat Completions（`DeepSeekChatGenerator`）
  - `rag/pipeline/`：编排（`RetrievalPipeline`、`RAGOrchestrator`、`run_retrieval_pipeline`）
  - `python -m rag.embed`：建索引入口（组合 embedding + vectorstore）

在 **`message-api`** 下运行单元测试：`python -m unittest discover -s tests -v`（需已安装 `requirements.txt`）。

**演示入口（推荐）**：在 `message-api` 下运行 **`python run_demo.py`**（默认一条样例问题）；自定义问题：`python run_demo.py "你的问题"`；只要 JSON：`python run_demo.py --json "狐狸"`。加 **`RAG_DEBUG=1`** 时才会打出少量 `[rag]` 阶段日志（默认不刷屏）。

批量自检：`python scripts/validate_rag_e2e.py`；快速 2 条：`python scripts/validate_rag_e2e.py --quick`。首次或 CI 可加 **`RAG_DISABLE_RERANK=1`** 跳过 CrossEncoder 下载。

- **未配置 `DATABASE_URL`**：只启用 SQLite 的 `/api/messages`（旧版星星模式）。
- **配置了 `DATABASE_URL`（Postgres）**：启用“连续对话记忆”接口（并把对话写入 Postgres）。

## 环境变量（部署/跨域）

- **`DATABASE_URL`**：Postgres 连接串（Railway/Render 通常会提供）
- **`CORS_ORIGINS`**：允许访问 API 的前端域名（逗号分隔）。例如：
  - `CORS_ORIGINS=https://<user>.github.io,https://<your-vercel-domain>`
- （可选）**`DEEPSEEK_API_KEY`**：配置后端生成更“因你而异”的回信；不配置则使用固定兜底回信。

## RAG（知识库检索 + 重排 + 引用）

在调用大模型前，会用用户最后一句话：**FAISS 向量检索 Top-K（默认 10）→ CrossEncoder 重排 → 取 Top-3**（可由环境变量调整），将带 `[S1]`… 编号的摘录与 **grounding 规则** 拼进首条 `system`。**`POST /api/chat`** 响应中会返回 **`citations`**（本次注入模型的摘录来源，含 `label` / `chunk_id` / **`source_name`**（出处标签）/ `text` 预览）。整条链路中每条 chunk 均携带 **`chunk_id` + `source_name`**。

### 知识库格式（`rag/data.json`）

- **兼容旧格式**：`"chunks": ["段落1", "段落2", …]` — 建索引时为每条自动生成 `chunk_id`（`kb_000`…）与默认 **`source_name`**（`little_prince_kb`，可由根级 **`default_source_name`** 或兼容字段 **`default_source`** 覆盖）。
- **推荐格式**：`"chunks": [{ "chunk_id": "可选", "source_name": "可选", "text": "正文" }, …]` — 仍可使用旧字段 **`source`** 代替 **`source_name`**（写入 `chunk_meta.json` 时统一为 `source_name`）。

### 建索引与生成文件

1. 安装依赖后，在 **`message-api`** 目录执行：

```powershell
cd .\message-api
.\.venv\Scripts\Activate.ps1
pip install -r .\requirements.txt
python -m rag.embed
```

2. 生成 **`rag/index.faiss`**（FAISS 序列化二进制）与 **`rag/chunk_meta.json`**（与向量行一一对应的 `{chunk_id, source_name, text}`；根目录 `.gitignore` 已忽略，部署需在目标环境执行 `python -m rag.embed` 或拷贝生成物）。若仅有旧版 **`chunk_order.json`**（无 `chunk_meta.json`），检索层会从文本列表合成伪元数据，避免未重建索引即崩溃。**旧版 `chunk_meta.json` 若只有 `source` 字段**，检索时仍会映射为 `source_name`，无需强制立刻重建（重建后可统一键名）。

3. 编辑知识库后务必重新运行 `python -m rag.embed`。

### RAG 相关环境变量（可选）

| 变量 | 默认 | 说明 |
|------|------|------|
| `RAG_FAISS_TOP_K` | `10` | 向量检索候选条数 |
| `RAG_FINAL_TOP_K` | `3` | 重排后注入模型的条数（≥1） |
| `RAG_RERANKER_MODELS` | （未设置则见下） | 逗号分隔的多个模型 id，**从左到右**尝试加载，直至成功（便于 bge 失败时自动换 ms-marco） |
| `RAG_RERANKER_MODEL` | （未设置则默认链） | 仅使用**单一** CrossEncoder 模型，不设则默认依次尝试 `BAAI/bge-reranker-base` → `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `RAG_DISABLE_RERANK` | 未设置 | 设为 `1` / `true` / `yes` 时跳过重排，直接取 FAISS 前 `RAG_FINAL_TOP_K` 条 |

首次使用 **`bge-reranker-base`** 会从 Hugging Face 下载权重（体积与内存占用明显高于 bi-encoder）；容器内存偏小或冷启动敏感时可改用 MiniLM 交叉编码器或设置 **`RAG_DISABLE_RERANK=1`**。若重排模型加载或推理失败，会自动降级为「不重排、截取 FAISS 前 K 条」。

## Agent（`/api/chat` 策略路由）

仅 **`POST /api/chat`** 走纯 Python 策略层（[`app/services/agent_service.py`](app/services/agent_service.py)），其它 LLM 接口不变。

1. **`decide_strategy(user_input)`** 将最后一条用户话分为三类（关键词匹配，先情绪后哲学）：
   - `emotion`：孤独、难过等 → **开启 RAG**，语气偏温柔共情
   - `philosophy`：意义、为什么等 → **开启 RAG**，语气偏哲理留白
   - `general`：其余 → **关闭 RAG**，语气偏日常闲聊

2. 处理顺序：**语气写入首条 `system`** → **按需拼接 RAG** → 调用 DeepSeek。

3. 响应 JSON 在 `assistant` 同级包含 **`strategy`**（`emotion` / `philosophy` / `general`）、**`citations`**（RAG 开启且检索成功时为非空数组，否则为 `[]`），以及 **`conclusion`**（短结论）与 **`analysis`**（展开分析）；`assistant.content` 仍为合并后的可读文本，便于旧前端兼容。

## 启动

在项目根目录（`小王子`）打开 PowerShell，执行：

```powershell
cd .\message-api
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r .\requirements.txt
uvicorn asgi:app --reload --host 127.0.0.1 --port 5000
```

看到 `Uvicorn running on http://127.0.0.1:5000` 即成功。`POST /api/chat`、`POST /api/analyze` 与其它 `/api/*`、`/health` 均可使用。

可选：仅 Flask 开发时在同一目录执行 `python .\wsgi.py`。生产（Railway/Render 等）默认 **`Procfile` 使用 `uvicorn asgi:app`**；若坚持用 Gunicorn 纯 WSGI，可改回 `gunicorn wsgi:app`（此时无 FastAPI 层的 `/api/chat`，由 Flask blueprint 提供 chat）。

## 一次性接上 Postgres（跨设备记忆）

1. 复制配置模板：

```powershell
cd .\message-api
copy .\.env.example .\.env
```

2. 打开 `message-api/.env`，把 `DATABASE_URL` 改成你的 Postgres 连接串。

3. 重启后端（或双击项目根目录的 `start_little_prince.bat`）。

> 现在 `app` 包在加载时会读取 `message-api/.env`，不用手动 `set` 环境变量。

## 验证

浏览器打开 `http://localhost:5000/health`，应返回 `{"ok":true}`。

然后再打开前端页面（The Little Prince 页面），到 B612 页面发送一句话：

- 若后端已配置 `DATABASE_URL`，会走连续对话接口并写入数据库；下次打开仍能看到历史。
- 若未配置 `DATABASE_URL`，会提示“后端暂不可达（邮局）”，并回退为离线回信（不跨设备记忆）。

