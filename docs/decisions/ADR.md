# ADR-001: 嵌入与 LLM 走 Ollama HTTP，不自建推理

## Status: Accepted (2026-09-22)

## Background
目标机器为 CPU-only（Radeon 780M 无 CUDA），无 cmake/MSVC。Python 侧推理库
（llama-cpp-python、sentence-transformers+torch、onnxruntime）均需编译或无 cp313 轮子。

## Decision
嵌入与生成统一走 Ollama HTTP API（`/api/embed`、`/api/chat`）；
Ollama 未安装或离线时注入零依赖实现（HashEmbedder / MockLLM）。

## Consequences
- 正面：零编译、零模型文件进仓库、模型管理交给 Ollama；
- 负面：多一跳 HTTP（本机实测 <5ms 开销）；吞吐受 Ollama 线程策略影响。

## Related: ADR-002, ADR-003

---

# ADR-002: 结构化输出不用 Ollama format=json

## Status: Accepted (2026-09-22)

## Background
本机实测 qwen2.5:1.5b-instruct + `format=json` 请求 120s 超时（概率性卡死），
而普通提示词约束的 JSON 输出稳定在数秒内返回。

## Decision
JSON 结构化输出采用「提示词强约束 + jsonschema(Pydantic) 后校验 + 有界重试 ×2 + 失败拒答」，
不使用 `format=json` 参数。

## Consequences
- 正面：避免已知超时路径；schema 校验成为运行期不变量（与 C1 一致）；
- 负面：极少数场景仍会产出非 JSON（由重试与拒答兜底）。

---

# ADR-003: 路由档位锁定为 规则/1.5b/检索增强，7b 仅手动

## Status: Accepted (2026-09-22)

## Background
实测 CPU 吞吐：1.5b ≈ 2.6 tok/s；7b q4_K_M（4.7GB）冷加载后吞吐不可用级，
且 15.3GB 内存限制同时常驻模型 ≤ 2。

## Decision
RouteTier = {rules, small, enhanced, large}；默认只在 rules/small/enhanced 间自动路由，
large 仅在 `enable_large=True` 且显式 `force_tier` 时可用。

## Consequences
- 正面：p95 延迟可控；内存峰值可控；
- 负面：复杂推理质量上限受 1.5b 限制（由 C1 校验 + C2 修复兜底，必要时拒答）。

---

# ADR-004: 词法索引用 BM25Plus 而非 BM25Okapi

## Status: Accepted (2026-09-22)

## Background
rank_bm25 的 BM25Okapi IDF = log(N-n+0.5) - log(n+0.5)，在 N≤2 的微型语料
（demo / golden 集）下全部 ≤ 0，epsilon 修正后得分恒 0，检索空转。

## Decision
使用同库的 BM25Plus（IDF = log(1 + (N-n+0.5)/(n+0.5)) 恒正），
并在查询侧过滤与查询词零交集的文档（BM25Plus 对未命中词也给 delta 下限分）。

## Consequences
- 正面：小语料下检索有效，行为可预测；
- 负面：大语料下 BM25Plus 与 Okapi 的排序差异可忽略（有 RRF 融合兜底）。

---

# ADR-005: 离线轨为默认轨，CI 零模型零网络全绿

## Status: Accepted (2026-09-22)

## Background
「干净环境可复现」需要不依赖任何外部资源的验证路径；在线轨质量数据无法在 CI 复现。

## Decision
默认 `VERITAS_TRACK=offline`：HashEmbedder + MockLLM + InMemoryMemory。
MockLLM 支持注入 hallucinate / schema_violate 失败模式，供不变量"噪声必被捕获"测试使用。
生产部署通过环境变量切在线轨，链路代码零改动。

## Consequences
- 正面：CI 在 ubuntu/windows 干净 runner 上分钟级全绿；贡献者 clone 即可跑通全流程；
- 负面：离线轨的检索/生成质量不代表在线轨（文档已声明，质量结论以在线轨为准）。

---
作者：晨星
