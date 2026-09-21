"""M10 guardian — 不变量校验（C1）+ 错误家族归类 + 自进化回路（C2）。

不变量在本模块统一注册与执行；校验失败产出带家族标签的结构化事件，
由 pipeline 触发定向修复；每次修复尝试落盘到 data/evolution.jsonl。
不变量：
  INV-GRD-001 注入噪声（幻觉断言/schema 违约）必被对应不变量捕获
  INV-GRD-002 校验器自身异常不得静默视为通过（记为未校验失败）
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .contracts import (
    Answer,
    Embedder,
    ErrorFamily,
    Hit,
    InvariantResult,
    LexicalIndex,
    VectorStore,
)


class InvariantSpec(BaseModel):
    invariant_id: str
    name: str
    module: str
    family: ErrorFamily


CheckFn = Callable[[dict[str, Any]], tuple[bool, str]]



class Guardian:
    def __init__(self, evolution_log: str | Path = "data/evolution.jsonl") -> None:
        self._specs: dict[str, InvariantSpec] = {}
        self._checks: dict[str, CheckFn] = {}
        self._log_path = Path(evolution_log)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------ registry
    def register(self, spec: InvariantSpec, check: CheckFn) -> None:
        self._specs[spec.invariant_id] = spec
        self._checks[spec.invariant_id] = check

    def list_invariants(self) -> list[InvariantSpec]:
        return list(self._specs.values())

    # ------------------------------------------------ run
    def check(self, ctx: dict[str, Any],
              only_ids: set[str] | None = None) -> list[InvariantResult]:
        results: list[InvariantResult] = []
        for iid, check in self._checks.items():
            if only_ids is not None and iid not in only_ids:
                continue
            spec = self._specs[iid]
            t0 = time.perf_counter()
            try:
                ok, detail = check(ctx)
            except Exception as exc:  # INV-GRD-002：校验器异常不算通过
                ok, detail = False, f"checker_error:{type(exc).__name__}:{exc}"
            results.append(InvariantResult(
                invariant_id=iid, name=spec.name, module=spec.module,
                passed=ok, family=spec.family, detail=detail,
                duration_ms=round((time.perf_counter() - t0) * 1000, 3),
            ))
        return results

    # ------------------------------------------------ diagnosis
    @staticmethod
    def diagnose(results: list[InvariantResult],
                 answer: Answer | None) -> ErrorFamily | None:
        """失败 -> 家族归类（取第一个失败项的家族）。"""
        for r in results:
            if not r.passed and not r.skipped and r.family is not None:
                return r.family
        if answer is not None and answer.refused:
            return ErrorFamily.RETRIEVAL
        return None

    # ------------------------------------------------ repair strategies
    @staticmethod
    def repair_plan(family: ErrorFamily | None) -> str:
        return {
            ErrorFamily.SCHEMA: "bounded_retry_shorter_output",
            ErrorFamily.RETRIEVAL: "expand_topk_rewrite_query",
            ErrorFamily.HALLUCINATION: "drop_unsupported_assertions",
            ErrorFamily.TIMEOUT: "downgrade_tier",
            ErrorFamily.TOOL: "fallback_no_tools",
            None: "none",
        }[family]

    # ------------------------------------------------ evolution log
    def log_episode(self, family: ErrorFamily | None, strategy: str,
                    trace_id: str, fixed: bool, detail: str = "") -> None:
        episode: dict[str, Any] = {
            "ts": time.time(), "trace_id": trace_id,
            "family": family.value if family else None,
            "strategy": strategy, "fixed": fixed, "detail": detail,
        }
        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(episode, ensure_ascii=False) + "\n")

    def episodes(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self._log_path.exists():
            return []
        lines = self._log_path.read_text(encoding="utf-8").strip().splitlines()
        return [json.loads(line) for line in lines[-limit:]]


# ------------------------------------------------- 内置不变量检查器


def make_core_invariants() -> list[tuple[InvariantSpec, CheckFn]]:
    """对核心链路可机械判定的契约集（对象由 ctx 注入，便于独立单测）。"""
    def emb_dim(ctx: dict[str, Any]) -> tuple[bool, str]:
        emb: Embedder = ctx["embedder"]
        vecs = emb.embed(["不变量校验探针"])
        ok = len(vecs[0]) == emb.dim
        return ok, f"dim={len(vecs[0])} expected={emb.dim}"

    def emb_deterministic(ctx: dict[str, Any]) -> tuple[bool, str]:
        emb: Embedder = ctx["embedder"]
        a = emb.embed(["确定性探针文本"])
        b = emb.embed(["确定性探针文本"])
        return a == b, "same input -> same vector"

    def vec_recall_self(ctx: dict[str, Any]) -> tuple[bool, str]:
        store: VectorStore = ctx["store"]
        emb: Embedder = ctx["embedder"]
        probe = ctx.get("probe_chunk")
        if probe is None or store.count() == 0:
            return True, "skipped:no_probe"
        hits = store.search(emb.embed([probe.text])[0], 3)
        ok = hits and hits[0].chunk.chunk_id == probe.chunk_id
        return bool(ok), f"top1={hits[0].chunk.chunk_id if hits else 'none'}"

    def lex_empty(ctx: dict[str, Any]) -> tuple[bool, str]:
        lex: LexicalIndex = ctx["lexical"]
        hits = lex.search("", 3)
        return len(hits) == 0, f"empty_query_hits={len(hits)}"

    def cit_supported(ctx: dict[str, Any]) -> tuple[bool, str]:
        """INV-CIT-001：每条断言与其绑定 chunk 的词重叠率 >= 阈值。"""
        answer: Answer = ctx["answer"]
        hits: list[Hit] = ctx["hits"]
        if answer.refused or not answer.assertions:
            return True, "refused_or_empty"
        by_id = {i + 1: h for i, h in enumerate(hits)}
        bad: list[str] = []
        for a in answer.assertions:
            h = by_id.get(a.citation_index or -1)
            if h is None:
                bad.append(f"no_hit:{a.text[:12]}")
                continue
            ratio = _overlap(a.text, h.chunk.text)
            if ratio < 0.12:
                bad.append(f"low_overlap({ratio:.2f}):{a.text[:12]}")
        return not bad, "; ".join(bad) or "all_supported"

    def route_valid(ctx: dict[str, Any]) -> tuple[bool, str]:
        rd = ctx.get("route")
        if rd is None:
            return True, "skipped:no_route"
        ok = rd.tier in ("rules", "small", "enhanced", "large")
        if rd.downgraded_from is not None and rd.tier != "rules":
            return False, "downgrade_not_applied"
        return ok, f"tier={rd.tier}"

    return [
        (InvariantSpec(invariant_id="INV-EMB-001", name="嵌入维度恒定",
                       module="m02_embedder", family=ErrorFamily.SCHEMA), emb_dim),
        (InvariantSpec(invariant_id="INV-EMB-002", name="嵌入确定性",
                       module="m02_embedder", family=ErrorFamily.SCHEMA), emb_deterministic),
        (InvariantSpec(invariant_id="INV-VEC-001", name="向量库自召回",
                       module="m03_vectorstore", family=ErrorFamily.RETRIEVAL), vec_recall_self),
        (InvariantSpec(invariant_id="INV-LEX-001", name="空查询返空",
                       module="m04_lexical", family=ErrorFamily.RETRIEVAL), lex_empty),
        (InvariantSpec(invariant_id="INV-CIT-001", name="断言有证据支撑",
                       module="m08_reasoner", family=ErrorFamily.HALLUCINATION), cit_supported),
        (InvariantSpec(invariant_id="INV-RT-001", name="路由决策合法",
                       module="m09_router", family=ErrorFamily.TIMEOUT), route_valid),
    ]


def _overlap(text: str, evidence: str) -> float:
    """字符 bigram Jaccard（零依赖、确定性、中文友好）。"""
    def grams(s: str) -> set[str]:
        chars = [c for c in s if not c.isspace()]
        if not chars:
            return set()
        if len(chars) == 1:
            return {chars[0]}
        return {a + b for a, b in zip(chars, chars[1:], strict=False)}

    g1, g2 = grams(text), grams(evidence)
    if not g1:
        return 1.0
    inter = len(g1 & g2)
    return inter / max(len(g1), 1)
