# 后端部署到 Render（方案 A）

## 1) 推送代码到 GitHub

把当前项目推到一个 GitHub 仓库（Render 从仓库拉代码部署）。

## 2) 在 Render 创建 Web Service

1. 打开 [Render](https://render.com/) 并登录  
2. `New +` -> `Web Service`  
3. 连接你的 GitHub 仓库  
4. 关键配置：
   - **Root Directory**: `message-api`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
5. 点击 `Create Web Service`

部署完成后你会得到一个地址，例如：
`https://little-prince-api.onrender.com`

健康检查：
`https://little-prince-api.onrender.com/health`

## 3) 前端接入线上 API

你的前端支持 URL 参数覆盖 API 地址（注意这里传的是 **`/api` 根路径**），直接这样访问即可：

`https://little-prince.xyz/?api=https://little-prince-api.onrender.com/api`

把上面的域名换成你 Render 实际域名。

## 4) 注意事项

- **连续对话记忆需要 Postgres**：给 Render Web Service 配置 `DATABASE_URL`（指向 Render Postgres）。
- **CORS**：可在后端环境变量里设置 `CORS_ORIGINS`，例如：
  - `CORS_ORIGINS=https://<yourname>.github.io,https://<your-site-domain>`
- 若不配置 `DATABASE_URL`，后端只会启用 SQLite 的旧接口 `/api/messages`（不具备“下次还记得”）。
