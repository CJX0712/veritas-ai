# Veritas AI 部署指南

> Author: 晨星

## 1. 本地开发（推荐，5 分钟）

前置：Python 3.13+，[uv](https://docs.astral.sh/uv/)（可选但推荐）。

```bash
git clone https://github.com/CJX0712/veritas-ai.git
cd veritas-ai
uv sync --frozen                      # 锁版安装（uv.lock 全树锁定，零编译）
uv run python tools/verify.py         # 门禁全绿后启动
uv run uvicorn veritas.app:app --port 8000
```

无 uv：`python -m venv .venv && .venv/Scripts/pip install -r requirements.txt`
（Linux/macOS 为 `.venv/bin/pip`），然后 `python -m pytest tests` 验证。

启动后：`curl http://127.0.0.1:8000/health` → `{"status":"ok",...}`。
控制台：浏览器打开 `web/console.html`（API 地址在文件顶部 `API` 常量）。

## 2. 在线轨（真模型）

前提：本机 Ollama ≥ 0.34 且已 `ollama pull bge-m3 qwen2.5:1.5b-instruct`。

```bash
# Windows PowerShell:  $env:VERITAS_TRACK="online"
export VERITAS_TRACK=online
uv run uvicorn veritas.app:app --port 8000
```

在线轨 `min_similarity=0.55`（bge-m3 语义空间），由 app 工厂自动设定。

## 3. Docker

```bash
docker compose up --build     # 离线轨，端口 8000
curl http://127.0.0.1:8000/health
```

在线轨（容器访问宿主机 Ollama）：在 `docker-compose.yml` 中放开
`VERITAS_TRACK=online` 与 `extra_hosts: host.docker.internal:host-gateway` 两段注释，
并设置环境变量使 Ollama 客户端指向 `http://host.docker.internal:11434`（v0.2 支持环境变量覆盖 base_url，当前版本可用镜像内 `OLLAMA_BASE_URL` 构建参数）。

## 4. CI（GitHub Actions）

`.github/workflows/ci.yml`：ubuntu + windows 双矩阵。
每步内容：`uv sync --frozen` → `tools/verify.py`（lint+types+tests+P0 扫描）
→ 拉起服务做 `/health` 与 `/api/v1/eval/run` 冒烟。
**CI runner 上没有任何模型与 GPU** —— 离线轨设计使 CI 完全不依赖外部资源。

## 5. 健康检查与运维

| 端点 | 用途 |
|---|---|
| `GET /health` | 存活 + 当前轨/模型/索引规模 |
| `GET /api/v1/evolution` | 错误家族分布与修复记录（排障入口） |
| `GET /api/v1/traces/{id}` | 单请求全链路 span 树 |

数据落盘：`data/evolution.jsonl`（自进化记录）、`data/memory.db`（SQLite 记忆，仅在线轨配置 SQLiteMemory 时）。二者均为可重建数据，不进仓库。

## 6. 升级与回滚

- 依赖变更必须走 `uv lock` 生成新 `uv.lock` 并提交；禁止手工编辑锁文件；
- 路由/阈值等运行参数集中在 `router.py` / `VeritasApp.__init__`，改后必须跑 `tools/verify.py` + golden 评测对比；
- 自进化采纳前应有 golden 无退化证据（当前版本记录 episode，v0.2 提供自动回归门禁）。

---
作者：晨星
