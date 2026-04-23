# message-api（投稿后端）

前端 `index.html` 会请求：

- 旧版“星星留言墙”（可选保留）：
  - `GET http://localhost:5000/api/messages`
  - `POST http://localhost:5000/api/messages`
- 新版“B612 连续对话（每个访问者独立记忆）”：
  - `GET  http://localhost:5000/api/conversations/me?visitorId=...`
  - `GET  http://localhost:5000/api/conversations/<conversationId>/messages`
  - `POST http://localhost:5000/api/conversations/<conversationId>/messages`

本目录提供一个轻量可运行的后端（Flask）。

- **未配置 `DATABASE_URL`**：只启用 SQLite 的 `/api/messages`（旧版星星模式）。
- **配置了 `DATABASE_URL`（Postgres）**：启用“连续对话记忆”接口（并把对话写入 Postgres）。

## 环境变量（部署/跨域）

- **`DATABASE_URL`**：Postgres 连接串（Railway/Render 通常会提供）
- **`CORS_ORIGINS`**：允许访问 API 的前端域名（逗号分隔）。例如：
  - `CORS_ORIGINS=https://<user>.github.io,https://<your-vercel-domain>`
- （可选）**`DEEPSEEK_API_KEY`**：配置后端生成更“因你而异”的回信；不配置则使用固定兜底回信。

## 启动

在项目根目录（`小王子`）打开 PowerShell，执行：

```powershell
cd .\message-api
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r .\requirements.txt
python .\app.py
```

看到类似 `Running on http://127.0.0.1:5000` 就说明启动成功。

## 一次性接上 Postgres（跨设备记忆）

1. 复制配置模板：

```powershell
cd .\message-api
copy .\.env.example .\.env
```

2. 打开 `message-api/.env`，把 `DATABASE_URL` 改成你的 Postgres 连接串。

3. 重启后端（或双击项目根目录的 `start_little_prince.bat`）。

> 现在 `app.py` 会自动读取 `message-api/.env`，不用手动 `set` 环境变量。

## 验证

浏览器打开 `http://localhost:5000/health`，应返回 `{"ok":true}`。

然后再打开前端页面（The Little Prince 页面），到 B612 页面发送一句话：

- 若后端已配置 `DATABASE_URL`，会走连续对话接口并写入数据库；下次打开仍能看到历史。
- 若未配置 `DATABASE_URL`，会提示“后端暂不可达（邮局）”，并回退为离线回信（不跨设备记忆）。

