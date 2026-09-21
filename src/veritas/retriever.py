"""M05 retriever — 混合检索：向量 + BM25，RRF 融合，重排只改顺序不改集合。

不变量：
  INV-RET-001 重排（rerank）只改变顺序，不改变命中集合
  INV-RET-002 RRF 结果集合 ⊇ 各路 top-k 命中的交集贡献（实现上从并集取分，
              故每条 RRF 命中必来自至少一路检索）
  INV-RET-003 空库/空查询必返回空列表
"""
from __future__ import annotations

from typing import Protocol

from .contracts import Chunk, Embedder, Hit, LexicalIndex, VectorStore
from .telemetry import SpanContext

RRF_K = 60


class Reranker(Protocol):
    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]: ...


class OverlapReranker:
    """词汇重叠重排（零成本、确定性）：按查询词覆盖率与原始分联合排序。"""

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        def cov(h: Hit) -> float:
            t = h.chunk.text.lower()
            words = [w for w in query.lower().split() if w]
            if not words:
                return 0.0
            return sum(1 for w in words if w in t) / len(words)

        return sorted(hits, key=lambda h: (cov(h), h.score), reverse=True)


class HybridRetriever:
    def __init__(self, store: VectorStore, lexical: LexicalIndex,
                 embedder: Embedder, reranker: Reranker | None = None,
                 parent_span: SpanContext | None = None) -> None:
        self._store = store
        self._lexical = lexical
        self._embedder = embedder
        self._reranker = reranker or OverlapReranker()
        self._span = parent_span

    def retrieve(self, query: str, k: int = 5) -> tuple[list[Hit], dict[str, object]]:
        """返回 (hits, meta)。meta 含 trace 用的中间统计。"""
        meta: dict[str, object] = {"top1_similarity": 0.0, "vector_hits": 0,
                                   "lexical_hits": 0, "pool": 0}
        if not query.strip() or (self._store.count() == 0 and self._lexical.count() == 0):
            return [], meta  # INV-RET-003

        v_hits: list[Hit] = []
        if self._store.count() > 0:
            qvec = self._embedder.embed([query])[0]
            v_hits = self._store.search(qvec, k * 2)

        l_hits = self._lexical.search(query, k * 2)
        meta["vector_hits"] = len(v_hits)
        meta["lexical_hits"] = len(l_hits)
        meta["top1_similarity"] = v_hits[0].score if v_hits else 0.0

        merged = _rrf_merge(v_hits, l_hits, k * 2)
        meta["pool"] = len(merged)
        final = self._reranker.rerank(query, merged)[:k]

        # INV-RET-001：重排不得增删元素
        pool_ids = {h.chunk.chunk_id for h in merged}
        final_ids = {h.chunk.chunk_id for h in final}
        if not final_ids.issubset(pool_ids):
            raise AssertionError("rerank changed the hit set (INV-RET-001)")
        return final, meta


def _rrf_merge(v_hits: list[Hit], l_hits: list[Hit], k: int) -> list[Hit]:
    """Reciprocal Rank Fusion。结果只来自两路输入（INV-RET-002）。"""
    best: dict[str, tuple[float, Hit]] = {}
    for hits in (v_hits, l_hits):
        for rank, h in enumerate(hits):
            rrf = 1.0 / (RRF_K + rank + 1)
            cur = best.get(h.chunk.chunk_id)
            if cur is None:
                best[h.chunk.chunk_id] = (rrf, h.model_copy(update={"score": rrf,
                                                                     "source": "rrf"}))
            else:
                cur0 = best[h.chunk.chunk_id]
                best[h.chunk.chunk_id] = (cur0[0] + rrf, cur0[1])
    out = sorted(best.values(), key=lambda x: x[0], reverse=True)[:k]
    return [h.model_copy(update={"score": round(s, 8)}) for s, h in out]


def chunk_text(text: str, max_chars: int = 400) -> list[str]:
    """按段落聚合的简单分块（零依赖，中文友好）。"""
    paras = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 2 <= max_chars:
            buf = f"{buf}\n{p}" if buf else p
        else:
            if buf:
                chunks.append(buf)
            while len(p) > max_chars:
                chunks.append(p[:max_chars])
                p = p[max_chars:]
            buf = p
    if buf:
        chunks.append(buf)
    return chunks or ([text[:max_chars]] if text.strip() else [])


def make_chunks(doc_id: str, title: str, text: str) -> list[Chunk]:
    parts = chunk_text(text)
    return [
        Chunk(doc_id=doc_id, chunk_id=f"{doc_id}#{i}",
              text=p, section=f"{title}#{i}", meta={"title": title})
        for i, p in enumerate(parts)
    ]
