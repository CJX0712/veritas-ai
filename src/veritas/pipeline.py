"""pipeline — 端到端装配：ingest + ask（含 C1 校验与 C2 自进化修复回路）。

唯一负责组装具体实现并执行完整链路；模块之间不直接互相 import 实现。
依赖注入：embedder / llm / memory / store / lexical 全部可替换（双轨的基础）。
"""
from __future__ import annotations

from typing import Any

from .contracts import (
    Answer,
    Chunk,
    Embedder,
    ErrorFamily,
    Hit,
    LLMClient,
    MemoryStore,
    RouteTier,
)
from .embedder import HashEmbedder, cosine
from .guardian import Guardian, make_core_invariants
from .lexical import BM25Index
from .llm import MockLLM
from .memory import InMemoryMemory
from .orchestrator import Orchestrator
from .reasoner import Reasoner
from .retriever import HybridRetriever, make_chunks
from .router import Router, query_hash
from .telemetry import SpanContext, Tracer, get_logger
from .vectorstore import InMemoryVectorStore

log = get_logger(__name__)


class VeritasApp:
    def __init__(self, embedder: Embedder, llm: LLMClient,
                 store: InMemoryVectorStore | None = None,
                 memory: MemoryStore | None = None,
                 enable_large: bool = False,
                 min_similarity: float = 0.12) -> None:
        self.embedder = embedder
        self.llm = llm
        self.store = store or InMemoryVectorStore()
        self.lexical = BM25Index()
        self.memory: MemoryStore = memory or InMemoryMemory()
        self.router = Router(enable_large=enable_large)
        self.orchestrator = Orchestrator()
        self.guardian = Guardian()
        self.min_similarity = min_similarity
        self._spans: dict[str, dict[str, Any]] = {}
        self._vectors: dict[str, list[float]] = {}
        self._probe_chunk: Chunk | None = None
        for spec, check in make_core_invariants():
            self.guardian.register(spec, check)

    # ------------------------------------------------------------ ingest
    def ingest(self, doc_id: str, title: str, text: str) -> int:
        chunks = make_chunks(doc_id, title, text)
        if not chunks:
            return 0
        vecs = self.embedder.embed([c.text for c in chunks])
        for chunk, vec in zip(chunks, vecs, strict=False):
            self.store.upsert(chunk, vec)
            self.lexical.add(chunk)
            self._vectors[chunk.chunk_id] = vec
        if self._probe_chunk is None and chunks:
            self._probe_chunk = chunks[0]
        return len(chunks)

    # ------------------------------------------------------------ ask
    def ask(self, query: str, idempotency_key: str | None = None,
            force_tier: str | None = None) -> tuple[Answer, dict[str, Any]]:
        key = idempotency_key or f"q:{query_hash(query)}"

        def run(tracer: Tracer) -> tuple[Answer, dict[str, Any]]:
            root = tracer.context(tracer.start("ask", "pipeline"))
            with root.span("route", "m09_router") as sp:
                route = self.router.route(query, force_tier)
                tracer.end(sp.span_id, "passed", tier=route.tier.value)

            answer, meta = self._ask_once(tracer, root, query, route, topk=5)
            family = Guardian.diagnose([r for r in answer.invariants], answer)
            strategy = Guardian.repair_plan(family)
            repaired = False
            if family is not None and family != ErrorFamily.TIMEOUT:
                answer, family = self._repair(
                    tracer, root, query, route, answer, family)
                repaired = True
            if repaired:
                self.guardian.log_episode(
                    family, strategy, tracer.trace_id,
                    fixed=(family is None),
                    detail=answer.refuse_reason or "repaired",
                )

            with root.span("memory_write", "m06_memory") as sp:
                self.memory.write(query, answer)
                tracer.end(sp.span_id, "passed", memory_count=self.memory.count())

            total = tracer.end(root.span_id, "passed")
            answer.trace_id = tracer.trace_id
            answer.route = route
            answer.total_ms = round(total, 1)
            meta_out = {
                "trace": tracer.tree(), "route": route.model_dump(),
                "repaired": repaired, "family": family.value if family else None,
                "strategy": strategy,
            }
            return answer, meta_out

        result, _trace_id, replayed = self.orchestrator.execute(run, key)
        answer, meta = result
        if replayed:
            meta["replayed"] = True
        run_id = self.orchestrator.register_run(answer.trace_id, {
            "query": query, "refused": answer.refused,
            "tier": answer.route.tier.value if answer.route else "",
            "repaired": meta.get("repaired", False),
        })
        meta["run_id"] = run_id
        return answer, meta

    # ------------------------------------------------------------ internal
    def _ask_once(self, tracer: Tracer, root: SpanContext, query: str,
                  route: Any, topk: int) -> tuple[Answer, dict[str, Any]]:
        hits: list[Hit] = []
        meta: dict[str, Any] = {}
        if route.tier != RouteTier.RULES:
            with root.span("retrieve", "m05_retriever") as sp:
                retriever = HybridRetriever(self.store, self.lexical, self.embedder)
                hits, meta = retriever.retrieve(query, topk)
                tracer.end(sp.span_id, "passed", hits=len(hits), **{
                    k2: v for k2, v in meta.items() if isinstance(v, (int, float))})
            # PRD F1：top-1 相似度低于阈值 -> retrieval 家族失败 -> 拒答
            top1_raw = meta.get("top1_similarity", 0.0)
            top1 = float(top1_raw) if isinstance(top1_raw, (int, float)) else 0.0
            if not hits or top1 < self.min_similarity:
                with root.span("refuse_low_confidence", "m10_guardian") as sp:
                    reasoner = Reasoner(self.llm)
                    answer = reasoner._refuse(  # noqa: SLF001
                        query, hits, f"retrieval_miss:top1={top1:.3f}")
                    tracer.end(sp.span_id, "passed", top1=round(top1, 3))
                    return answer, meta

        with root.span("generate", "m08_reasoner") as sp:
            reasoner = Reasoner(self.llm)
            answer = reasoner.generate(query, hits, route.tier)
            tracer.end(sp.span_id, "passed", assertions=len(answer.assertions),
                       model=answer.model)

        with root.span("guard", "m10_guardian") as sp:
            ctx = self._guard_ctx(query, hits, answer, route)
            results = self.guardian.check(ctx)
            answer.invariants = results
            failed = [r.invariant_id for r in results if not r.passed]
            tracer.end(sp.span_id, "passed" if not failed else "failed",
                       failed=failed)
        return answer, meta

    def _repair(
            self, tracer: Tracer, root: SpanContext, query: str,
            route: Any, answer: Answer, family: ErrorFamily,
    ) -> tuple[Answer, ErrorFamily | None]:
        """C2 定向修复：按家族出手；修复后再校验，通过即闭环。"""
        if family == ErrorFamily.HALLUCINATION:
            with root.span("repair_drop_unsupported", "m10_guardian") as sp:
                kept, removed = self._drop_unsupported(answer)
                answer = answer.model_copy(update={
                    "assertions": kept, "removed_count": removed,
                    "refused": not kept,
                    "refuse_reason": "all_assertions_unsupported" if not kept else "",
                })
                hits, _ = HybridRetriever(
                    self.store, self.lexical, self.embedder).retrieve(query, 5)
                ctx = self._guard_ctx(query, hits, answer, route)
                results = self.guardian.check(ctx)
                answer = answer.model_copy(update={"invariants": results})
                fixed = all(r.passed for r in results)
                tracer.end(sp.span_id, "passed" if fixed else "failed",
                           removed=removed)
                return answer, (None if fixed else family)

        if family == ErrorFamily.RETRIEVAL:
            with root.span("repair_expand_topk", "m10_guardian") as sp:
                route2 = route.model_copy(update={"tier": RouteTier.ENHANCED})
                answer2, _ = self._ask_once(tracer, root, query, route2, topk=10)
                fixed = all(r.passed for r in answer2.invariants) and not answer2.refused
                tracer.end(sp.span_id, "passed" if fixed else "failed")
                return answer2, (None if fixed else family)

        return answer, family

    def _drop_unsupported(self, answer: Answer) -> tuple[list[Any], int]:
        kept: list[Any] = []
        removed = 0
        for a in answer.assertions:
            if a.citation_index is None:
                removed += 1
                continue
            kept.append(a)
        return kept, removed

    def _guard_ctx(self, query: str, hits: list[Hit], answer: Answer,
                   route: Any) -> dict[str, Any]:
        return {
            "query": query, "hits": hits, "answer": answer, "route": route,
            "embedder": self.embedder, "store": self.store,
            "lexical": self.lexical, "probe_chunk": self._probe_chunk,
        }

    # ------------------------------------------------------------ expose
    def verify_retrieval_invariants(self) -> list[dict[str, Any]]:
        """独立触发核心不变量校验（不依赖一次提问）。"""
        probe = self._probe_chunk
        ctx = self._guard_ctx("", [], Answer(query="", trace_id=""), None)
        if probe is not None:
            vec = self._vectors.get(probe.chunk_id)
            if vec:
                top = self.store.search(vec, 1)
                ctx["probe_ok"] = bool(top) and top[0].chunk.chunk_id == probe.chunk_id
        return [r.model_dump() for r in self.guardian.check(ctx)]

    def similarity(self, a: str, b: str) -> float:
        va, vb = self.embedder.embed([a, b])
        return cosine(va, vb)


def build_offline_app() -> VeritasApp:
    """离线轨：哈希嵌入 + MockLLM + 内存记忆。CI / 无网络 / 无模型全绿。"""
    return VeritasApp(embedder=HashEmbedder(dim=512), llm=MockLLM(),
                      memory=InMemoryMemory())
