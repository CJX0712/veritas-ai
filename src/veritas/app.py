"""M13 api — FastAPI 入口（装配层，零业务逻辑）。

双轨切换：VERITAS_TRACK=offline（默认，全离线可复现）| online（Ollama）。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .embedder import HashEmbedder, OllamaEmbedder
from .evaluator import GoldenSet, evaluate
from .llm import OllamaLLM
from .memory import InMemoryMemory
from .pipeline import VeritasApp, build_offline_app
from .telemetry import configure_logging

configure_logging()

_GOLDEN_PATH = Path(__file__).resolve().parents[2] / "golden" / "golden.json"


class IngestRequest(BaseModel):
    doc_id: str
    title: str
    text: str = Field(min_length=1)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    idempotency_key: str | None = None
    force_tier: str | None = Field(default=None,
                                   pattern="^(rules|small|enhanced|large)$")


def _build_app() -> VeritasApp:
    track = os.environ.get("VERITAS_TRACK", "offline")
    if track == "online":
        app = VeritasApp(embedder=OllamaEmbedder(model="bge-m3"),
                         llm=OllamaLLM(model="qwen2.5:1.5b-instruct"),
                         memory=InMemoryMemory(),
                         min_similarity=0.55)
    else:
        app = build_offline_app()
    app.track = track  # type: ignore[attr-defined]
    return app


def create_app() -> FastAPI:
    vapp = _build_app()
    traces: dict[str, dict[str, Any]] = {}
    last_eval: dict[str, Any] = {}

    api = FastAPI(
        title="Veritas AI", version="0.1.0",
        description="Invariant-first, self-evolving AI Agent runtime",
    )

    @api.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "track": getattr(vapp, "track", "offline"),
            "embedder": vapp.embedder.name,
            "llm": vapp.llm.model_name,
            "chunks": vapp.store.count(),
            "memory": vapp.memory.count(),
        }

    @api.post("/api/v1/ingest")
    def ingest(req: IngestRequest) -> dict[str, Any]:
        n = vapp.ingest(req.doc_id, req.title, req.text)
        return {"doc_id": req.doc_id, "chunks": n, "total_chunks": vapp.store.count()}

    @api.post("/api/v1/query")
    def query(req: QueryRequest) -> dict[str, Any]:
        answer, meta = vapp.ask(req.query, req.idempotency_key, req.force_tier)
        traces[answer.trace_id] = meta["trace"]
        return {"answer": json.loads(answer.model_dump_json()),
                "replayed": meta.get("replayed", False),
                "repaired": meta.get("repaired", False),
                "family": meta.get("family"),
                "strategy": meta.get("strategy"),
                "run_id": meta.get("run_id")}

    @api.get("/api/v1/traces")
    def list_traces(limit: int = 20) -> list[dict[str, Any]]:
        return vapp.orchestrator.runs(limit)

    @api.get("/api/v1/traces/{trace_id}")
    def get_trace(trace_id: str) -> dict[str, Any]:
        tr = traces.get(trace_id)
        if tr is None:
            raise HTTPException(status_code=404, detail="trace not found")
        return tr

    @api.get("/api/v1/invariants")
    def invariants() -> list[dict[str, Any]]:
        return [s.model_dump() for s in vapp.guardian.list_invariants()]

    @api.post("/api/v1/invariants/verify")
    def verify_now() -> list[dict[str, Any]]:
        return vapp.verify_retrieval_invariants()

    @api.get("/api/v1/evolution")
    def evolution(limit: int = 50) -> dict[str, Any]:
        episodes = vapp.guardian.episodes(limit)
        dist: dict[str, int] = {}
        for ep in episodes:
            fam = ep.get("family") or "none"
            dist[fam] = dist.get(fam, 0) + 1
        fixed = sum(1 for e in episodes if e.get("fixed"))
        return {"episodes": episodes, "family_distribution": dist,
                "total": len(episodes), "fixed": fixed}

    @api.post("/api/v1/eval/run")
    def eval_run() -> dict[str, Any]:
        if not _GOLDEN_PATH.exists():
            raise HTTPException(status_code=404,
                                detail="golden set missing: golden/golden.json")
        golden = GoldenSet.model_validate(
            json.loads(_GOLDEN_PATH.read_text(encoding="utf-8")))
        report = evaluate(golden, HashEmbedder(dim=512))
        last_eval.clear()
        last_eval.update(json.loads(report.model_dump_json()))
        return last_eval

    @api.get("/api/v1/eval/latest")
    def eval_latest() -> dict[str, Any]:
        if not last_eval:
            raise HTTPException(status_code=404, detail="no eval run yet")
        return last_eval

    return api


app = create_app()
