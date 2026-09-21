# Veritas AI 使用指南

> Author: 晨星

## 1. 启动

```bash
uv run uvicorn veritas.app:app --port 8000     # 离线轨（默认）
# VERITAS_TRACK=online 切换 Ollama 真模型
```

控制台：双击 `web/console.html`，或浏览器直接打开。五个页面：问答 / 链路 / 守卫 / 进化 / 评测。

## 2. 核心流程（控制台或 curl）

### 导入知识库

```bash
curl -X POST http://127.0.0.1:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"doc_id":"d1","title":"差旅管理办法","text":"市内交通费按实际发生额的 80% 报销……"}'
```

文本自动分块（段落聚合，≤400 字符），同时进向量索引与 BM25 词法索引。

### 提问

```bash
curl -X POST http://127.0.0.1:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query":"市内交通费报销上限是多少？"}'
```

响应关键字段：

| 字段 | 含义 |
|---|---|
| `answer.assertions[]` | 逐句断言，每句带 `citation_index` 指向证据 |
| `answer.citations[]` | 证据列表（chunk、章节、相似度、片段） |
| `answer.refused` | 检索置信度不足或全部断言无证据时为 true —— **系统宁可拒答也不编造** |
| `answer.invariants[]` | 本请求全部不变量判定结果（C1 可视化） |
| `answer.route` | 路由档位与决策依据（C3 可视化） |
| `repaired` / `family` / `strategy` | 是否触发自进化修复、错误家族、修复策略（C2） |

幂等：相同 `query`（或显式 `idempotency_key`）重复请求直接返回缓存结果，不重复推理。

### 查看链路与守卫

```bash
curl http://127.0.0.1:8000/api/v1/traces            # 最近 runs
curl http://127.0.0.1:8000/api/v1/traces/{trace_id} # 单请求 span 树
curl -X POST http://127.0.0.1:8000/api/v1/invariants/verify  # 手动全量校验
curl http://127.0.0.1:8000/api/v1/evolution          # 错误家族分布 + 修复记录
```

### 评测

```bash
curl -X POST http://127.0.0.1:8000/api/v1/eval/run   # NDCG@10 / Recall@5 / MRR
```

评测始终使用**独立索引**，可随时在生产服务运行，不影响线上数据。

## 3. 二次开发（替换实现）

所有外部依赖都面向 Protocol 注入 —— 换实现不改链路：

```python
from veritas.pipeline import VeritasApp
from veritas.embedder import OllamaEmbedder
from veritas.llm import OllamaLLM

app = VeritasApp(
    embedder=OllamaEmbedder(model="nomic-embed-text"),  # 换嵌入模型
    llm=OllamaLLM(model="qwen2.5:7b-instruct-q4_K_M"),  # 换生成模型（注意 CPU 吞吐）
    enable_large=True,                                   # 允许 large 手动档
    min_similarity=0.55,                                 # 语义嵌入阈值
)
app.ingest("d1", "文档", "正文……")
ans, meta = app.ask("你的问题")
```

要点：
- 自定义 Embedder / LLM / Memory 只需实现 `contracts.py` 中对应 Protocol；
- 新增不变量：`app.guardian.register(InvariantSpec(...), check_fn)`，校验函数返回 `(bool, detail)`；
- golden 集格式见 `golden/golden.json`（corpus + cases.relevant 为 chunk_id 列表）。

## 4. 常见问题

| 现象 | 原因与处理 |
|---|---|
| 查询总是"未找到可靠依据" | 置信度门槛拒答。离线轨阈值 0.12 已按实测标定；语料太少时补充相关文档，或降低 `min_similarity` |
| 在线轨响应 10 秒以上 | CPU 推理正常水平（1.5b 约 2.6 tok/s）；控制台有计时与中断提示；可 `force_tier="rules"` 直出检索片段 |
| `mock-*` 出现在模型字段 | 你在离线轨。这是设计行为：离线 Mock 保证零依赖可复现 |
| 7b 档如何启用 | `VeritasApp(enable_large=True)` 后 `force_tier="large"`；不建议在 CPU 上做默认档 |

---
作者：晨星
