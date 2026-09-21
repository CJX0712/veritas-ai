"""M01 telemetry — 轻量 trace 树 + structlog 结构化日志。

每棵 trace 是一棵 span 树；span 状态与耗时在结束时落定，
最终整体 dump 为 dict（供 API /traces/{id} 与控制台瀑布图消费）。
"""
from __future__ import annotations

import threading
import time
from typing import Any

import structlog

from .contracts import TraceSpan, new_id


def configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(colors=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(20),
        cache_logger_on_first_use=True,
    )


class Tracer:
    """线程安全的单请求 span 树收集器。"""

    def __init__(self, trace_id: str | None = None) -> None:
        self.trace_id = trace_id or new_id("tr")
        self._lock = threading.Lock()
        self._spans: dict[str, TraceSpan] = {}
        self._order: list[str] = []
        root = TraceSpan(
            span_id=new_id("sp"), parent_id=None, name="request",
            module="pipeline", started_at=time.perf_counter(),
        )
        self._root_id = root.span_id
        self._add(root)

    def _add(self, span: TraceSpan) -> None:
        with self._lock:
            self._spans[span.span_id] = span
            self._order.append(span.span_id)

    def start(self, name: str, module: str, parent_id: str | None = None,
              **attrs: Any) -> str:
        span = TraceSpan(
            span_id=new_id("sp"), parent_id=parent_id or self._root_id,
            name=name, module=module, started_at=time.perf_counter(), attrs=attrs,
        )
        self._add(span)
        return span.span_id

    def end(self, span_id: str, status: str = "passed", **attrs: Any) -> float:
        with self._lock:
            span = self._spans.get(span_id)
            if span is None or span.ended_at is not None:
                return 0.0  # 幂等：手动 end 后 __exit__ 不重复结算
            span.ended_at = time.perf_counter()
            span.status = status
            span.attrs.update(attrs)
        return (span.ended_at - span.started_at) * 1000

    def context(self, span_id: str) -> SpanContext:
        return SpanContext(self, span_id)

    def tree(self) -> dict[str, Any]:
        with self._lock:
            spans = [self._spans[i].model_copy() for i in self._order]
        for s in spans:
            if s.ended_at is not None:
                s.attrs["duration_ms"] = round((s.ended_at - s.started_at) * 1000, 2)
        by_parent: dict[str | None, list[TraceSpan]] = {}
        for s in spans:
            by_parent.setdefault(s.parent_id, []).append(s)
        for v in by_parent.values():
            v.sort(key=lambda x: x.started_at)

        def build(span_id: str | None) -> list[dict[str, Any]]:
            return [s.model_dump() | {"children": build(s.span_id)}
                    for s in by_parent.get(span_id, [])]

        root = self._spans[self._root_id]
        total = 0.0
        if root.ended_at is not None:
            total = (root.ended_at - root.started_at) * 1000
        else:
            # 根 span 未显式结束时，以最后一个已结束 span 收口
            ended = [s.ended_at for s in spans if s.ended_at is not None]
            if ended:
                total = (max(ended) - root.started_at) * 1000
        return {
            "trace_id": self.trace_id,
            "total_ms": round(total, 2),
            "spans": build(None),
            "flat": [s.model_dump() for s in spans],
        }

    def fail_count(self) -> int:
        return sum(1 for s in self._spans.values() if s.status == "failed")


class SpanContext:
    """with 语法包装：异常自动标记 failed。"""

    def __init__(self, tracer: Tracer, parent_id: str) -> None:
        self._tracer = tracer
        self._parent = parent_id
        self.span_id = ""

    def __enter__(self) -> str:
        raise NotImplementedError

    def span(self, name: str, module: str, **attrs: Any) -> OpenSpan:
        sid = self._tracer.start(name, module, self._parent, **attrs)
        return OpenSpan(self._tracer, sid)


class OpenSpan:
    def __init__(self, tracer: Tracer, span_id: str) -> None:
        self._tracer = tracer
        self.span_id = span_id

    def __enter__(self) -> OpenSpan:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc is not None:
            self._tracer.end(self.span_id, "failed", error=f"{type(exc).__name__}: {exc}")
        else:
            self._tracer.end(self.span_id, "passed")


def get_logger(name: str) -> Any:
    return structlog.get_logger(name)
