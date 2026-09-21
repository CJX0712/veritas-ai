"""API E2E：进程内 ASGI 跑 HTTP 全链路（含错误流）。"""
import pytest
from fastapi.testclient import TestClient

from veritas.app import create_app


@pytest.fixture()
def client():
    api = create_app()
    return TestClient(api)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["track"] == "offline"


def test_ingest_and_query_flow(client):
    r = client.post("/api/v1/ingest", json={
        "doc_id": "d1", "title": "差旅",
        "text": "市内交通费按实际发生额的 80% 报销，单日上限 120 元。"})
    assert r.status_code == 200
    assert r.json()["chunks"] >= 1

    r = client.post("/api/v1/query", json={"query": "市内交通费报销上限是多少？"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"]["refused"] is False
    assert body["answer"]["assertions"]
    trace_id = body["answer"]["trace_id"]

    r = client.get(f"/api/v1/traces/{trace_id}")
    assert r.status_code == 200
    assert r.json()["trace_id"] == trace_id


def test_query_refusal_on_unknown_topic(client):
    client.post("/api/v1/ingest", json={
        "doc_id": "d1", "title": "差旅", "text": "市内交通费按实际发生额的 80% 报销。"})
    r = client.post("/api/v1/query", json={"query": "木星的卫星有几颗？"})
    assert r.status_code == 200
    assert r.json()["answer"]["refused"] is True


def test_trace_404(client):
    r = client.get("/api/v1/traces/tr_missing")
    assert r.status_code == 404


def test_query_validation_error(client):
    r = client.post("/api/v1/query", json={"query": ""})
    assert r.status_code == 422


def test_query_force_tier_rejected_value(client):
    r = client.post("/api/v1/query", json={"query": "q", "force_tier": "turbo"})
    assert r.status_code == 422


def test_invariants_listing_and_verify(client):
    r = client.get("/api/v1/invariants")
    assert r.status_code == 200
    ids = [i["invariant_id"] for i in r.json()]
    assert "INV-CIT-001" in ids

    r = client.post("/api/v1/invariants/verify")
    assert r.status_code == 200
    assert all(item["passed"] for item in r.json())


def test_evolution_endpoint(client):
    client.post("/api/v1/ingest", json={
        "doc_id": "d1", "title": "t", "text": "市内交通费报销上限 120 元。"})
    client.post("/api/v1/query", json={"query": "市内交通费报销上限是多少？"})
    r = client.get("/api/v1/evolution")
    assert r.status_code == 200
    body = r.json()
    assert "episodes" in body and "family_distribution" in body


def test_eval_flow(client):
    r = client.post("/api/v1/eval/run")
    assert r.status_code == 200
    body = r.json()
    assert body["dataset"]
    assert 0.0 <= body["ndcg_at_10"] <= 1.0
    r = client.get("/api/v1/eval/latest")
    assert r.status_code == 200
