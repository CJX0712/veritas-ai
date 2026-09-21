"""M04 lexical — BM25 词法索引（rank-bm25）+ CJK 单字/ASCII 分词。

不变量：
  INV-LEX-001 空查询必返回空列表
  INV-LEX-002 包含查询词的文档得分高于不包含的
"""
from __future__ import annotations

import threading

from rank_bm25 import BM25Plus

from .contracts import Chunk, Hit
from .embedder import tokenize


class BM25Index:
    """词法索引。

    选 BM25Plus 而非 BM25Okapi（ADR-004）：Okapi 的 IDF = log(N-n+0.5)-log(n+0.5)
    在小语料（N<=2）下全部 <= 0，epsilon 修正后得分恒 0 —— demo/golden 规模必踩；
    BM25Plus 的 IDF = log(1 + (N-n+0.5)/(n+0.5)) 恒正，且对未命中词给出 delta 下限。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._chunks: list[Chunk] = []
        self._corpus: list[list[str]] = []
        self._bm25: BM25Plus | None = None
        self._dirty = False

    def add(self, chunk: Chunk) -> None:
        with self._lock:
            self._chunks.append(chunk)
            self._corpus.append(tokenize(chunk.text))
            self._dirty = True

    def _rebuild(self) -> None:
        if self._dirty or self._bm25 is None:
            if self._corpus:
                self._bm25 = BM25Plus(self._corpus)
            else:
                self._bm25 = None
            self._dirty = False

    def count(self) -> int:
        with self._lock:
            return len(self._chunks)

    def search(self, query: str, k: int) -> list[Hit]:
        tokens = tokenize(query)
        if not tokens:
            return []  # INV-LEX-001
        with self._lock:
            self._rebuild()
            bm25 = self._bm25
            chunks = list(self._chunks)
        if bm25 is None or k <= 0:
            return []
        scores = bm25.get_scores(tokens)
        qset = set(tokens)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        out = []
        for i in order:
            # BM25Plus 对未命中词也给 delta 下限分，必须过滤零交集文档
            if scores[i] > 0 and qset & set(self._corpus[i]):
                out.append(Hit(
                    chunk=chunks[i], score=round(float(scores[i]), 6),
                    source="lexical",
                ))
        return out
