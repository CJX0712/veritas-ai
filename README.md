# Veritas AI

> **不变量优先、可自进化的 AI Agent 运行时** —— 把「正确性」从测试环节提升到架构层。
>
> Author: **晨星** | License: Apache-2.0 | Python 3.13 | CPU-only 友好

[![CI](https://github.com/CJX0712/veritas-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/CJX0712/veritas-ai/actions/workflows/ci.yml)

## 这是什么

LangGraph / AutoGen / CrewAI 回答"怎么把 Agent 拼出来"，Veritas AI 回答的是：
**怎么证明它这一跳没走错、错了怎么自己修好、以及这一切在多大算力预算内能跑完。**

四个内核（相互咬合，缺一即退化）：

| 内核 | 一句话 | 关键模块 |
|---|---|---|
| **C1 Invariant-First** | 每个模块自带机器可判定的正确性契约，运行期实时校验（二值判定，非打分） | `guardian` |
| **C2 Self-Evolving Repair** | 失败按 5 个错误家族归类 → 定向修复 → 修复后再校验，通过才闭环 | `guardian` + `pipeline` |
| **C3 Compute-Aware Routing** | 按复杂度 × 预算在 规则引擎 → 1.5b → 检索增强 间路由，超预算自动降级 | `router` |
| **C4 双轨验证** | 在线（Ollama 真模型）/ 离线（哈希嵌入 + Mock LLM），CI 在零模型零网络下全绿 | 全部（依赖注入） |

> 三者的因果链：**有了 C1，小模型路径的错误变成可检测事件；有了 C2，这些事件被自动修复；
> 于是 C3 才敢把流量激进地下放到便宜的档位。** 这是组合差异化的核心，不是并列卖点。

## 一键复现（干净环境）

```bash
git clone https://github.com/CJX0712/veritas-ai.git
cd veritas-ai
uv sync --frozen                      # 锁版安装，全 wheel 零编译
uv run python tools/verify.py         # 门禁：lint + 类型 + 33 测试 + P0 扫描
uv run uvicorn veritas.app:app --port 8000   # 启动（默认离线轨，无需模型/网络）
```

无 uv 时：`pip install -r requirements.txt && python -m pytest tests`。

打开 `web/console.html`（单文件、零外部依赖、双击即用）即可操作。
**首次 clone 无需任何模型与 API Key** —— 默认离线轨全流程可跑通。

## 在线轨（可选，本机实测数据）

```bash
export VERITAS_TRACK=online           # Windows: set VERITAS_TRACK=online
ollama pull bge-m3 qwen2.5:1.5b-instruct
uv run uvicorn veritas.app:app --port 8000
```

本机（Ryzen 7 H 255 / 16 核 / 无 CUDA）实测：bge-m3 嵌入确定性成立（1024 维），
qwen2.5:1.5b-instruct function-calling 可用、吞吐约 2.6 tok/s；
Ollama `format=json` 会超时，故结构化输出采用提示词约束 + schema 后校验 + 有界重试。

## 12 秒理解架构

```
请求 → M09 路由(复杂度/预算) → M05 混合检索(BM25+向量 RRF)
     → M08 生成(逐句引用绑定, schema 校验+有界重试)
     → M10 守卫(不变量运行期校验)
     ├─ 通过 → M06 记忆 → 响应 + trace
     └─ 失败 → 错误家族归类(schema/retrieval/timeout/hallucination/tool)
              → 定向修复(降级/扩召回/删幻觉断言/重试) → 再校验 → 闭环
```

完整模块表、不变量清单、API、数据流见 **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**；
部署与运维见 **[docs/DEPLOY.md](docs/DEPLOY.md)**；使用与二次开发见 **[docs/USAGE.md](docs/USAGE.md)**。

## 仓库结构

```
src/veritas/    13 个单一职责模块（每文件 ≤300 行，依赖严格单向向下）
tests/          不变量测试（噪声注入必被捕获）+ API E2E
golden/         评测黄金集（独立索引评测，生产干扰不影响指标）
tools/          verify.py 一键门禁 + scan_emoji.py P0 扫描
web/            console.html 单文件控制台
docs/           架构 / 部署 / 使用 / ADR 决策记录
.github/        CI（ubuntu + windows 双平台，干净 runner 复现）
```

## 署名与许可

全部代码与文档作者：**晨星**。Apache-2.0。
