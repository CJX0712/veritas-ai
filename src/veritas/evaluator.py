"""M12 evaluator — golden 集离线评测：NDCG@10 / Recall@5 / MRR。

铁律（防"指标全错但不报错"）：
  - 每次评测新建【独立内存索引】重灌语料，绝不复用生产索引；
  - 往生产索引灌干扰文档后，评测指标必须逐位不变（INV-EV-002）。
"""
from __future__ import annotations

import math
import time
from typing import Any

from pydantic import BaseModel, Field

from .embedder import HashEmbedder
from .lexical import BM25Index
from .retriever import HybridRetriever, make_chunks
from .telemetry import get_logger
from .vectorstore import InMemoryVectorStore

log = get_logger(__name__)


class GoldenCase(BaseModel):
    case_id: str
    query: str
    relevant: list[str] = Field(description="相关 chunk_id 列表")


class GoldenSet(BaseModel):
    name: str
    corpus: list[dict[str, Any]] = Field(description="[{doc_id,title,text}]")
    cases: list[GoldenCase]


class EvalReport(BaseModel):
    dataset: str
    n_cases: int
    ndcg_at_10: float
    recall_at_5: float
    mrr: float
    duration_ms: float
    track: str = "offline"
    ts: float = Field(default_factory=time.time)


def build_fresh_index(corpus: list[dict[str, Any]],
                      embedder: Any) -> tuple[HybridRetriever, InMemoryVectorStore]:
    """新建独立索引并灌语料（绝不复用生产索引）。"""
    store: InMemoryVectorStore = InMemoryVectorStore()
    lex = BM25Index()
    texts = [c["text"] for c in corpus]
    vecs = embedder.embed(texts) if texts else []
    for doc, vec in zip(corpus, vecs, strict=False):
        for chunk in make_chunks(doc["doc_id"], doc["title"], doc["text"]):
            store.upsert(chunk, vec)
            lex.add(chunk)
    return HybridRetriever(store, lex, embedder), store


def ndcg_at_k(ranked_ids: list[str], relevant: set[str], k: int = 10) -> float:
    dcg = 0.0
    for i, cid in enumerate(ranked_ids[:k]):
        rel = 1.0 if cid in relevant else 0.0
        dcg += rel / math.log2(i + 2)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def evaluate(golden: GoldenSet, embedder: Any,
             k_retrieval: int = 10) -> EvalReport:
    retriever, _ = build_fresh_index(golden.corpus, embedder)
    t0 = time.perf_counter()
    ndcgs: list[float] = []
    recalls: list[float] = []
    mrrs: list[float] = []
    for case in golden.cases:
        hits, _meta = retriever.retrieve(case.query, k_retrieval)
        ranked = [h.chunk.chunk_id for h in hits]
        relevant = set(case.relevant)
        ndcgs.append(ndcg_at_k(ranked, relevant, 10))
        top5 = set(ranked[:5])
        recalls.append(len(top5 & relevant) / len(relevant) if relevant else 0.0)
        rr = 0.0
        for i, cid in enumerate(ranked):
            if cid in relevant:
                rr = 1.0 / (i + 1)
                break
        mrrs.append(rr)
    dt = (time.perf_counter() - t0) * 1000
    n = max(len(golden.cases), 1)
    return EvalReport(
        dataset=golden.name, n_cases=len(golden.cases),
        ndcg_at_10=round(sum(ndcgs) / n, 4),
        recall_at_5=round(sum(recalls) / n, 4),
        mrr=round(sum(mrrs) / n, 4),
        duration_ms=round(dt, 1),
    )


def default_embedder() -> HashEmbedder:
    return HashEmbedder(dim=512)
