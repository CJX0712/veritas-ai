"""M03 vectorstore — 内存余弦向量库（默认，零服务零依赖）。

qdrant 本地模式作为可选实现（extras: qdrant），接口不变，import 失败自动降级。
不变量：
  INV-VEC-001 upsert 后必能以自身向量检索命中该 chunk 且 rank=1
  INV-VEC-002 count 与 upsert/delete 精确一致
"""
from __future__ import annotations

import threading

from .contracts import Chunk, Hit
from .embedder import cosine


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._chunks: dict[str, Chunk] = {}
        self._vectors: dict[str, list[float]] = {}

    def upsert(self, chunk: Chunk, vector: list[float]) -> None:
        with self._lock:
            self._chunks[chunk.chunk_id] = chunk
            self._vectors[chunk.chunk_id] = vector

    def delete(self, chunk_id: str) -> bool:
        with self._lock:
            existed = chunk_id in self._chunks
            self._chunks.pop(chunk_id, None)
            self._vectors.pop(chunk_id, None)
            return existed

    def count(self) -> int:
        with self._lock:
            return len(self._chunks)

    def all_chunks(self) -> list[Chunk]:
        with self._lock:
            return list(self._chunks.values())

    def search(self, vector: list[float], k: int) -> list[Hit]:
        with self._lock:
            items = list(self._vectors.items())
            chunks = dict(self._chunks)
        if k <= 0 or not items:
            return []
        scored = [
            (cid, cosine(vector, vec)) for cid, vec in items
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        out = []
        for cid, score in scored[:k]:
            chunk = chunks.get(cid)
            if chunk is not None:
                out.append(Hit(chunk=chunk, score=round(score, 6), source="vector"))
        return out


class VectorStoreBundle:
    """向量库 + 词法索引 + 嵌入器的检索侧组合，供 retriever 使用。"""

    def __init__(self, store: InMemoryVectorStore) -> None:
        self.store = store
