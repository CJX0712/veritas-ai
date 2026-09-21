"""M11 orchestrator — 幂等执行与 run 记录。

不变量：
  INV-ORC-001 相同幂等键重复执行必返回完全相同结果，且只真正执行一次
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from .contracts import new_id
from .telemetry import Tracer, get_logger

log = get_logger(__name__)


class Orchestrator:
    def __init__(self, max_cache: int = 256) -> None:
        self._lock = threading.Lock()
        self._cache: dict[str, dict[str, Any]] = {}
        self._executions: dict[str, int] = {}
        self._max_cache = max_cache
        self._runs: dict[str, dict[str, Any]] = {}

    def execute(self, fn: Callable[[Tracer], Any], idempotency_key: str,
                tracer: Tracer | None = None) -> tuple[Any, str, bool]:
        """返回 (result, trace_id, replayed)。replayed=True 表示命中缓存。"""
        with self._lock:
            if idempotency_key in self._cache:
                cached = self._cache[idempotency_key]
                self._executions[idempotency_key] = (
                    self._executions.get(idempotency_key, 0) + 1)
                return cached["result"], cached["trace_id"], True

        tr = tracer or Tracer()
        result = fn(tr)
        with self._lock:
            if len(self._cache) >= self._max_cache:
                oldest = next(iter(self._cache))
                self._cache.pop(oldest, None)
            self._cache[idempotency_key] = {
                "result": result, "trace_id": tr.trace_id}
            self._executions[idempotency_key] = (
                self._executions.get(idempotency_key, 0) + 1)
        return result, tr.trace_id, False

    def executions(self, idempotency_key: str) -> int:
        with self._lock:
            return self._executions.get(idempotency_key, 0)

    def register_run(self, trace_id: str, record: dict[str, Any]) -> str:
        run_id = new_id("run")
        with self._lock:
            self._runs[run_id] = record | {"run_id": run_id, "trace_id": trace_id}
        return run_id

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            rec = self._runs.get(run_id)
        return dict(rec) if rec else None

    def runs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._runs.values())
        return items[-limit:]
