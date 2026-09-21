# Veritas AI 系统架构

> Author: 晨星 | 版本 v0.1.0 | 配套 ADR 见 docs/decisions/

## 1. 分层与模块

```
L4 Interface     M12 api (FastAPI)              M13 console (web/console.html)
L3 Orchestration M09 router | M10 guardian | M11 orchestrator (pipeline 装配编排)
L2 Capability    M05 retriever | M06 memory | M07 llm | M08 reasoner
L1 Primitive     M02 embedder | M03 vectorstore | M04 lexical
L0 Foundation    M00 contracts (pydantic 模型 + Protocol) | M01 telemetry
```

| 模块 | 文件 | 唯一职责 | 关键不变量（机器可判定，二值） |
|---|---|---|---|
| M00 contracts | contracts.py | 类型/Protocol，零业务 | — |
| M01 telemetry | telemetry.py | span 树 + 结构化日志 | 每请求一棵完整 trace 树 |
| M02 embedder | embedder.py | 文本→向量 | INV-EMB-001 维度恒定；INV-EMB-002 确定性；INV-EMB-003 拒绝零向量 |
| M03 vectorstore | vectorstore.py | 内存余弦检索 | INV-VEC-001 upsert 后必召回自身；INV-VEC-002 count 精确 |
| M04 lexical | lexical.py | BM25 词法索引 | INV-LEX-001 空查询返空；INV-LEX-002 命中词文档得分更高 |
| M05 retriever | retriever.py | RRF 融合 + 重排 | INV-RET-001 重排只改顺序不改集合；INV-RET-002 结果只来自各路输入 |
| M06 memory | memory.py | 情景记忆 | INV-MEM-001 写入必可召回；INV-MEM-002 forget 后单调减 |
| M07 llm | llm.py | Ollama HTTP / 确定性 Mock | Mock 与真实实现同接口；注入失败模式供测试 |
| M08 reasoner | reasoner.py | 生成 + 引用绑定 | INV-GEN-001 引用必在本次命中集合内；INV-GEN-002 同输入同输出 |
| M09 router | router.py | 复杂度/预算路由 | INV-RT-001 任何输入产出合法档位；INV-RT-002 超预算必带降级标记 |
| M10 guardian | guardian.py | 不变量校验 + 自进化 | INV-GRD-001 噪声必被捕获；INV-GRD-002 校验器异常≠通过 |
| M11 orchestrator | orchestrator.py | 幂等执行 | INV-ORC-001 同幂等键只真执行一次，结果一致 |
| M12 evaluator | evaluator.py | NDCG/Recall/MRR | INV-EV 独立索引评测，生产灌干扰后指标逐位不变 |
| M13 api | app.py | HTTP 装配层（零业务） | /health 恒 200；错误码契约 |

**依赖纪律**：L(n) 只能 import L(n-1) 的 Protocol，具体实现由 `pipeline.VeritasApp` 构造注入 —— 这是 C4 双轨与可测试性的结构保证。

## 2. 端到端数据流

```
POST /api/v1/query
  └─ M11 orchestrator (幂等键 = q:md5(query))
      └─ M01 tracer 根 span
          ├─ M09 router.route(query)
          │    复杂度 = f(长度, 条件词, 问号数) → tier ∈ {rules, small, enhanced, large}
          │    elapsed > budget → 降级 rules 并标记 downgraded_from   [INV-RT-002]
          ├─ M05 retrieve(query, topk=5)
          │    向量路: M02.embed(query) → M03.search(k*2)
          │    词法路: M04.search(k*2)
          │    RRF 融合(k=60) → OverlapReranker 重排           [INV-RET-001/002]
          │    top1_sim < min_similarity → retrieval-miss → 拒答
          ├─ M08 reasoner.generate(query, hits, tier)
          │    prompt = CITE|chunk_id|text 上下文行 + JSON 输出指令
          │    LLM 输出 → JSON 解析 → AssertionPayload schema 校验
          │    引用绑定: citation ∉ 命中集合 → 丢弃（生成层兜幻觉）
          │    解析失败 → 有界重试 ×2 → 拒答
          ├─ M10 guardian.check(ctx)                            [C1]
          │    6 条核心不变量逐条执行（含校验器异常自保护）
          ├─ diagnose → ErrorFamily（首个失败项的家族）
          ├─ repair(family)                                      [C2]
          │    HALLUCINATION → 删无证据断言 → 重校验
          │    RETRIEVAL → topk 5→10 重查重生成
          │    SCHEMA/TIMEOUT → 重试 / 降级（在各自层内已处理）
          │    修复结果落 data/evolution.jsonl
          └─ M06 memory.write → 返回 Answer + trace 树
```

## 3. 双轨（C4）实现

| 依赖 | 在线轨 | 离线轨（默认） |
|---|---|---|
| 嵌入 | `OllamaEmbedder(bge-m3)` 1024 维，HTTP | `HashEmbedder(512)` 字符 bigram FNV 散列，进程内 |
| LLM | `OllamaLLM(qwen2.5:1.5b)` temperature=0 seed=42 | `MockLLM` 从 CITE 标记确定性生成；可注入 hallucinate/schema_violate 模式 |
| 记忆 | `SQLiteMemory` | `InMemoryMemory`（同接口） |
| 切换 | `VERITAS_TRACK=online` | 默认 |

**确定性**：离线轨全部组件无随机性、无时间依赖输出 —— 相同输入逐字节相同输出，这是"可复现=可信任"的基础。

**相似度阈值标定**（hash 嵌入实测）：相关查询 top1 ≥ 0.158，无关 ≤ 0.071 → 阈值取 0.12；在线轨 bge-m3 语义空间取 0.55。

## 4. 错误家族（产品契约，命名稳定）

| 家族 | 触发 | 修复策略 |
|---|---|---|
| `schema` | JSON 解析失败 / 输出违约 | 收短输出 + 有界重试 ×2 |
| `retrieval` | top1 相似度 < 阈值 / 拒答 | 扩大 top-k 重查 |
| `hallucination` | 断言无证据支撑（bigram 重叠 < 0.12） | 删除不支持断言，删空则拒答 |
| `timeout` | 档位耗时超预算 | 降档重试 |
| `tool` | 工具调用违约 | 回退无工具路径（v0.2） |

## 5. 评测协议（防"指标全错但不报错"）

- 每次评测**新建内存索引**重灌 golden 语料，绝不读写生产索引；
- 硬断言：生产索引灌入 50 篇干扰文档后，评测指标逐位不变（测试 `test_eval_independent_index_noise_free`）；
- 指标：NDCG@10、Recall@5、MRR（golden/enterprise-kb-smoke-v1，8 条标注用例起步）。

## 6. 关键工程约束（来自本机实测，见 ADR）

1. 全依赖有 wheel，零编译（本机无 cmake/MSVC）；
2. 不依赖 HuggingFace（网络不可达），模型只走 Ollama / ModelScope；
3. Ollama `format=json` 超时 → 提示词约束 + 后校验（ADR-002）；
4. 7b 模型 CPU 吞吐不可用 → 路由默认三档（ADR-003）；
5. rank_bm25 的 BM25Okapi 在 N≤2 语料下得分恒 0 → 用 BM25Plus（ADR-004）。

---
作者：晨星
