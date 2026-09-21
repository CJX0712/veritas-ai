"""M02 embedder — 文本向量化。Ollama 在线轨 / 哈希离线轨，同 Protocol。

不变量（guardian 注册）：
  INV-EMB-001 输出维度恒等于 self.dim
  INV-EMB-002 同输入必得同输出（确定性）
  INV-EMB-003 零向量必须被拒绝
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Any

import httpx

from .telemetry import get_logger

log = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """通用分词：CJK 单字 + ASCII 词。供哈希嵌入与词法索引共用。"""
    return _TOKEN_RE.findall(text.lower())


class HashEmbedder:
    """确定性哈希嵌入（离线轨 / CI）：字符 bigram + 词，FNV 特征散列，L2 归一化。"""

    def __init__(self, dim: int = 512) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"hash-bigram-{self._dim}"

    def _feats(self, text: str) -> list[str]:
        chars = [c for c in text.lower() if not c.isspace()]
        bigrams = ["".join(p) for p in zip(chars, chars[1:], strict=False)]
        return tokenize(text) + bigrams

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            vec = [0.0] * self._dim
            for feat in self._feats(t):
                h = int.from_bytes(
                    hashlib.md5(feat.encode("utf-8")).digest()[:8], "little"
                )
                idx = h % self._dim
                sign = 1.0 if (h >> 63) & 1 else -1.0
                vec[idx] += sign
            norm = math.sqrt(sum(x * x for x in vec))
            if norm == 0:
                raise ValueError("zero vector: empty or blank text")  # INV-EMB-003
            vec = [x / norm for x in vec]
            out.append(vec)
        return out


class OllamaEmbedder:
    """在线轨：Ollama /api/embed（bge-m3 1024 维 / nomic 768 维，已实测确定性）。"""

    def __init__(self, model: str = "bge-m3",
                 base_url: str = "http://127.0.0.1:11434",
                 timeout: float = 120.0) -> None:
        self._model = model
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._dim = 0

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"ollama:{self._model}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(
                f"{self._base}/api/embed",
                json={"model": self._model, "input": texts},
            )
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
        embs = data.get("embeddings")
        if not embs or len(embs) != len(texts):
            raise ValueError("ollama embed returned mismatched embeddings")
        self._dim = len(embs[0])
        for v in embs:
            if all(abs(x) < 1e-12 for x in v):
                raise ValueError("zero vector returned by embedder")  # INV-EMB-003
        log.info("embed_ok", model=self._model, n=len(texts), dim=self._dim)
        return embs


def cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return num / (na * nb)
