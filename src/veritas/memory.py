"""M06 memory — SQLite 情景记忆 + 会话召回。

不变量：
  INV-MEM-001 write 后 recall 必能召回刚写入的条目
  INV-MEM-002 forget 后 count 单调减，且被遗忘条目不再出现
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from .contracts import Answer


class SQLiteMemory:
    def __init__(self, db_path: str | Path = "data/memory.db") -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._init()

    def _init(self) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS memory ("
                " id TEXT PRIMARY KEY, created_at REAL NOT NULL,"
                " query TEXT NOT NULL, query_norm TEXT NOT NULL,"
                " answer_json TEXT NOT NULL)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_qnorm ON memory(query_norm)"
            )

    @staticmethod
    def _norm(q: str) -> str:
        return "".join(q.lower().split())

    def write(self, query: str, answer: Answer) -> str:
        import time as _t
        import uuid as _u
        mid = f"mem_{_u.uuid4().hex[:12]}"
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO memory (id, created_at, query, query_norm, answer_json)"
                " VALUES (?,?,?,?,?)",
                (mid, _t.time(), query, self._norm(query), answer.model_dump_json()),
            )
        return mid

    def recall(self, query: str, k: int = 3) -> list[Answer]:
        norm = self._norm(query)
        if not norm:
            return []
        with self._lock:
            rows = self._conn.execute(
                "SELECT answer_json FROM memory WHERE query_norm LIKE ?"
                " ORDER BY created_at DESC LIMIT ?",
                (f"%{norm[:24]}%", k),
            ).fetchall()
        return [Answer.model_validate_json(r[0]) for r in rows]

    def count(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM memory").fetchone()
        return int(row[0])

    def forget(self, memory_id: str) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute("DELETE FROM memory WHERE id = ?", (memory_id,))
        return cur.rowcount > 0

    def dump_stats(self) -> dict[str, object]:
        return {"count": self.count()}


class InMemoryMemory:
    """零依赖实现（CI / 单测）：接口与 SQLiteMemory 完全一致。"""

    def __init__(self) -> None:
        self._items: dict[str, tuple[str, Answer]] = {}
        self._order: list[str] = []

    def write(self, query: str, answer: Answer) -> str:
        import uuid as _u
        mid = f"mem_{_u.uuid4().hex[:12]}"
        self._items[mid] = (self._norm(query), answer)
        self._order.insert(0, mid)
        return mid

    @staticmethod
    def _norm(q: str) -> str:
        return "".join(q.lower().split())

    def recall(self, query: str, k: int = 3) -> list[Answer]:
        norm = self._norm(query)
        out: list[Answer] = []
        for mid in self._order:
            qnorm, ans = self._items[mid]
            if norm and (norm in qnorm or qnorm in norm):
                out.append(ans)
            if len(out) >= k:
                break
        return out

    def count(self) -> int:
        return len(self._items)

    def forget(self, memory_id: str) -> bool:
        existed = self._items.pop(memory_id, None) is not None
        if existed and memory_id in self._order:
            self._order.remove(memory_id)
        return existed

    def dump_stats(self) -> dict[str, object]:
        return {"count": self.count()}


def answer_to_record(ans: Answer) -> dict[str, object]:
    return json.loads(ans.model_dump_json())
