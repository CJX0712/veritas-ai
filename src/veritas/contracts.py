"""M00 contracts — 类型与 Protocol 定义，零业务逻辑。

所有跨模块数据结构集中在此；模块之间只依赖本文件中的 Protocol，
运行时由 pipeline 装配注入具体实现（依赖严格单向向下）。
"""
from __future__ import annotations

import enum
import time
import uuid
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

# ---------------------------------------------------------------- errors


class ErrorFamily(enum.StrEnum):
    """C2 错误家族（产品契约，命名必须稳定）。"""

    SCHEMA = "schema"
    RETRIEVAL = "retrieval"
    TIMEOUT = "timeout"
    HALLUCINATION = "hallucination"
    TOOL = "tool"


class RouteTier(enum.StrEnum):
    """C3 算力路由档位（本机 CPU-only 实测后锁定为三档 + 手动大档）。"""

    RULES = "rules"          # 规则/缓存直出
    SMALL = "small"          # qwen2.5:1.5b（本地默认生成档）
    ENHANCED = "enhanced"    # 检索增强 + 1.5b
    LARGE = "large"          # qwen2.5:7b，仅手动触发（CPU 吞吐过低）


# ---------------------------------------------------------------- core data


class Chunk(BaseModel):
    doc_id: str
    chunk_id: str
    text: str
    section: str = ""
    meta: dict[str, Any] = Field(default_factory=dict)


class Hit(BaseModel):
    chunk: Chunk
    score: float
    source: str  # "vector" | "lexical" | "rrf"


class Citation(BaseModel):
    index: int
    chunk_id: str
    doc_id: str
    section: str
    snippet: str
    score: float


class Assertion(BaseModel):
    text: str
    citation_index: int | None = None


class RouteDecision(BaseModel):
    tier: RouteTier
    complexity: float
    latency_budget_ms: int
    reasons: list[str] = Field(default_factory=list)
    downgraded_from: RouteTier | None = None


class InvariantResult(BaseModel):
    invariant_id: str
    name: str
    module: str
    passed: bool
    family: ErrorFamily | None = None
    detail: str = ""
    duration_ms: float = 0.0
    skipped: bool = False


class TraceSpan(BaseModel):
    span_id: str
    parent_id: str | None
    name: str
    module: str
    started_at: float
    ended_at: float | None = None
    status: str = "running"  # running|passed|failed
    attrs: dict[str, Any] = Field(default_factory=dict)


class Answer(BaseModel):
    query: str
    trace_id: str
    refused: bool = False
    refuse_reason: str = ""
    assertions: list[Assertion] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    removed_count: int = 0
    route: RouteDecision | None = None
    invariants: list[InvariantResult] = Field(default_factory=list)
    total_ms: float = 0.0
    model: str = ""


# ---------------------------------------------------------------- llm types


class ChatMessage(BaseModel):
    role: str
    content: str


class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]


class ChatResponse(BaseModel):
    content: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    model: str = ""
    eval_count: int = 0


# ---------------------------------------------------------------- protocols


@runtime_checkable
class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        """文本 -> 向量。不变量：同输入必得同输出；维度恒定。"""
        ...

    @property
    def dim(self) -> int: ...

    @property
    def name(self) -> str: ...


@runtime_checkable
class VectorStore(Protocol):
    def upsert(self, chunk: Chunk, vector: list[float]) -> None: ...
    def search(self, vector: list[float], k: int) -> list[Hit]: ...
    def count(self) -> int: ...
    def all_chunks(self) -> list[Chunk]: ...


@runtime_checkable
class LexicalIndex(Protocol):
    def add(self, chunk: Chunk) -> None: ...
    def search(self, query: str, k: int) -> list[Hit]: ...
    def count(self) -> int: ...


@runtime_checkable
class LLMClient(Protocol):
    def chat(
        self, messages: list[ChatMessage], tools: list[ToolSpec] | None = None
    ) -> ChatResponse: ...

    @property
    def model_name(self) -> str: ...


@runtime_checkable
class MemoryStore(Protocol):
    def write(self, query: str, answer: Answer) -> str: ...
    def recall(self, query: str, k: int) -> list[Answer]: ...
    def count(self) -> int: ...


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def now_ms() -> float:
    return time.perf_counter() * 1000
