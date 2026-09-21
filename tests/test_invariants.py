"""核心不变量与噪声捕获测试（C1）：每个不变量注入噪声必被捕获。"""
import pytest

from veritas.contracts import Answer, Assertion, ErrorFamily, Hit
from veritas.embedder import HashEmbedder
from veritas.guardian import Guardian, InvariantSpec
from veritas.llm import MockLLM
from veritas.memory import InMemoryMemory
from veritas.pipeline import build_offline_app
from veritas.retriever import HybridRetriever, make_chunks
from veritas.vectorstore import InMemoryVectorStore

CORPUS = [
    {"doc_id": "d1", "title": "差旅",
     "text": "市内交通费按实际发生额的 80% 报销，单日上限 120 元。"},
    {"doc_id": "d2", "title": "报销",
     "text": "报销单需在费用发生后 30 天内提交，逾期不予受理。"},
]


def _filled_app() -> tuple[object, list[dict]]:
    app = build_offline_app()
    for d in CORPUS:
        app.ingest(d["doc_id"], d["title"], d["text"])
    return app, CORPUS


# ------------------------------------------------- M02 embedder

def test_emb_dimension_constant():
    emb = HashEmbedder(dim=512)
    vecs = emb.embed(["a", "更长的中文文本测试", "x" * 500])
    assert all(len(v) == 512 for v in vecs)


def test_emb_deterministic():
    emb = HashEmbedder(dim=512)
    assert emb.embed(["确定性探针"]) == emb.embed(["确定性探针"])


def test_emb_rejects_zero_vector():
    # 空文本无特征 -> 全零 -> 必须拒绝（INV-EMB-003）
    with pytest.raises(ValueError):
        HashEmbedder(dim=512).embed([""])
    with pytest.raises(ValueError):
        HashEmbedder(dim=512).embed(["   "])


# ------------------------------------------------- M03 vectorstore

def test_vec_recall_self():
    store = InMemoryVectorStore()
    emb = HashEmbedder(dim=512)
    chunks = make_chunks("d", "标题", "关于企业知识库检索与问答的设计要点。")
    for c in chunks:
        store.upsert(c, emb.embed([c.text])[0])
    hits = store.search(emb.embed([chunks[0].text])[0], 3)
    assert hits[0].chunk.chunk_id == chunks[0].chunk_id


def test_vec_count_exact():
    store = InMemoryVectorStore()
    emb = HashEmbedder(dim=512)
    chunk = make_chunks("d", "t", "内容内容内容。")[0]
    store.upsert(chunk, emb.embed([chunk.text])[0])
    assert store.count() == 1
    assert store.delete(chunk.chunk_id) is True
    assert store.count() == 0


# ------------------------------------------------- M04 lexical

def test_lex_empty_query_returns_empty():
    app, _ = _filled_app()
    assert app.lexical.search("", 5) == []
    assert app.lexical.search("   ", 5) == []


def test_lex_relevant_scores_higher():
    app, _ = _filled_app()
    hit_rel = app.lexical.search("交通费 报销", 5)
    hit_irr = app.lexical.search("年假 天数", 5)
    assert hit_rel and hit_rel[0].score > 0
    assert not hit_irr or hit_irr[0].score < hit_rel[0].score


# ------------------------------------------------- M05 retriever

def test_ret_rerank_preserves_set():
    app, _ = _filled_app()
    ret = HybridRetriever(app.store, app.lexical, app.embedder)
    hits, _ = ret.retrieve("交通费报销上限", 2)
    ids1 = {h.chunk.chunk_id for h in hits}
    hits2, _ = ret.retrieve("交通费报销上限", 2)
    ids2 = {h.chunk.chunk_id for h in hits2}
    assert ids1 == ids2  # 确定性：同输入同集合


def test_ret_rrf_hits_from_union_only():
    """噪声注入：手工构造 RRF 结果若含未知 chunk 必然异常。"""
    from veritas.retriever import _rrf_merge
    c1 = make_chunks("d1", "t", "内容一。")[0]
    c2 = make_chunks("d2", "t", "内容二。")[0]
    h1 = Hit(chunk=c1, score=1.0, source="vector")
    h2 = Hit(chunk=c2, score=1.0, source="lexical")
    merged = _rrf_merge([h1], [h2], 5)
    union = {c1.chunk_id, c2.chunk_id}
    assert all(h.chunk.chunk_id in union for h in merged)


def test_ret_empty_store_returns_empty():
    app = build_offline_app()
    from veritas.retriever import HybridRetriever
    ret = HybridRetriever(app.store, app.lexical, app.embedder)
    hits, meta = ret.retrieve("任何问题", 5)
    assert hits == []


# ------------------------------------------------- M06 memory

def test_mem_write_then_recall():
    mem = InMemoryMemory()
    ans = Answer(query="测试问题", trace_id="tr_x")
    mem.write("测试问题", ans)
    got = mem.recall("测试问题")
    assert got and got[0].trace_id == "tr_x"


def test_mem_forget_monotonic():
    mem = InMemoryMemory()
    a = Answer(query="q", trace_id="t")
    m1 = mem.write("q", a)
    before = mem.count()
    assert mem.forget(m1) is True
    assert mem.count() == before - 1
    assert mem.recall("q") == []


# ------------------------------------------------- M09 router

def test_route_always_valid():
    from veritas.router import Router
    router = Router()
    for q in ["", "  ", "短", "a" * 500, "P1 故障响应时限是多久，周末算不算？"]:
        rd = router.route(q)
        assert rd.tier in ("rules", "small", "enhanced", "large")


def test_route_downgrade_marks_origin():
    from veritas.router import Router
    router = Router(budget_ms={"rules": 200, "small": 1, "enhanced": 1, "large": 1})
    rd = router.route("这是一个需要检索增强的较长问题", elapsed_ms=999)
    assert rd.tier == "rules" and rd.downgraded_from is not None


# ------------------------------------------------- M10 guardian（噪声必被捕获）

def _run_checks(extra_answer: Answer | None = None):
    app, _ = _filled_app()
    hits, _ = HybridRetriever(app.store, app.lexical, app.embedder).retrieve(
        "交通费报销上限", 5)
    answer = extra_answer or Answer(query="交通费报销上限", trace_id="tr_t")
    ctx = {"query": "q", "hits": hits, "answer": answer, "route": None,
           "embedder": app.embedder, "store": app.store,
           "lexical": app.lexical, "probe_chunk": app._probe_chunk}
    return {r.invariant_id: r for r in app.guardian.check(ctx)}


def test_grd_hallucination_noise_captured():
    """INV-GRD-001：幻觉断言必被 INV-CIT-001 捕获。"""
    bad = Answer(query="q", trace_id="t", assertions=[
        Assertion(text="此句与任何证据毫无关联。", citation_index=1)])
    res = _run_checks(bad)
    assert res["INV-CIT-001"].passed is False
    assert res["INV-CIT-001"].family == ErrorFamily.HALLUCINATION


def test_grd_good_answer_passes():
    app, _ = _filled_app()
    hits, _ = HybridRetriever(app.store, app.lexical, app.embedder).retrieve(
        "交通费报销上限", 5)
    good = Answer(query="交通费报销上限", trace_id="t", assertions=[
        Assertion(text="市内交通费按实际发生额的 80% 报销。", citation_index=1)])
    ctx = {"query": "q", "hits": hits, "answer": good, "route": None,
           "embedder": app.embedder, "store": app.store,
           "lexical": app.lexical, "probe_chunk": app._probe_chunk}
    res = {r.invariant_id: r for r in app.guardian.check(ctx)}
    assert res["INV-CIT-001"].passed is True


def test_grd_checker_error_not_silent_pass():
    """INV-GRD-002：校验器自身异常不得静默视为通过。"""
    g = Guardian()

    def boom(ctx):
        raise RuntimeError("checker exploded")

    g.register(InvariantSpec(invariant_id="INV-X", name="x", module="m",
                             family=ErrorFamily.SCHEMA), boom)
    results = g.check({})
    assert results[0].passed is False
    assert "checker_error" in results[0].detail


def test_diagnose_maps_family():
    app, _ = _filled_app()
    bad = Answer(query="q", trace_id="t", assertions=[
        Assertion(text="完全无关的幻觉内容句子。", citation_index=1)])
    hits, _ = HybridRetriever(app.store, app.lexical, app.embedder).retrieve(
        "交通费报销上限", 5)
    ctx = {"query": "q", "hits": hits, "answer": bad, "route": None,
           "embedder": app.embedder, "store": app.store,
           "lexical": app.lexical, "probe_chunk": app._probe_chunk}
    results = app.guardian.check(ctx)
    assert Guardian.diagnose(results, bad) == ErrorFamily.HALLUCINATION


# ------------------------------------------------- M11 orchestrator 幂等

def test_orch_idempotent_single_execution():
    from veritas.orchestrator import Orchestrator
    orch = Orchestrator()
    calls = []

    def fn(tr):
        calls.append(1)
        return "result"

    r1, t1, rep1 = orch.execute(fn, "key-a")
    r2, t2, rep2 = orch.execute(fn, "key-a")
    assert r1 == r2 == "result"
    assert t1 == t2
    assert rep1 is False and rep2 is True
    assert len(calls) == 1
    assert orch.executions("key-a") == 2


# ------------------------------------------------- M12 evaluator 独立索引

def test_eval_independent_index_noise_free():
    from veritas.evaluator import GoldenSet, evaluate
    golden = GoldenSet(name="t", corpus=CORPUS, cases=[
        {"case_id": "c1", "query": "交通费报销上限", "relevant": ["d1#0"]},
    ])
    emb = HashEmbedder(dim=512)
    r1 = evaluate(golden, emb)
    # 生产索引灌入大量干扰
    app = build_offline_app()
    for i in range(50):
        app.ingest(f"noise{i}", "干扰", f"完全无关的干扰文档内容 {i}。")
    r2 = evaluate(golden, emb)
    assert r1.ndcg_at_10 == r2.ndcg_at_10
    assert r1.recall_at_5 == r2.recall_at_5


# ------------------------------------------------- 端到端（离线轨）

def test_e2e_ask_refuse_and_repair():
    app, _ = _filled_app()
    ans, meta = app.ask("市内交通费报销上限是多少？")
    assert ans.refused is False
    assert ans.assertions
    assert all(r.passed for r in ans.invariants)

    ans2, meta2 = app.ask("公司年假有几天？")
    assert ans2.refused is True  # 检索置信度不足 -> 拒答


def test_e2e_mock_hallucination_repaired():
    app = build_offline_app()
    for d in CORPUS:
        app.ingest(d["doc_id"], d["title"], d["text"])
    app.llm = MockLLM(mode="hallucinate")  # 注入幻觉噪声
    # 用超过检索置信度门槛的查询，确保进入生成/校验阶段而非 retrieval 拒答
    ans, meta = app.ask("市内交通费报销上限是多少？", idempotency_key="hall-1")
    assert meta["repaired"] is True
    # 幻觉引用在生成层即被过滤 -> 有效断言为空 -> 拒答（家族归为 retrieval-miss）。
    # 关键安全属性：幻觉内容绝不放行。
    assert meta["family"] in (None, "hallucination", "retrieval")
    assert ans.refused or all(
        r.passed for r in ans.invariants if r.invariant_id == "INV-CIT-001")
    assert not ans.assertions  # 幻觉 mock 只会产生无效引用断言，必须全部被拦


def test_e2e_mock_schema_violation_no_bad_output():
    app = build_offline_app()
    for d in CORPUS:
        app.ingest(d["doc_id"], d["title"], d["text"])
    app.llm = MockLLM(mode="schema_violate")
    ans, meta = app.ask("交通费报销上限", idempotency_key="schema-1")
    # schema 违约有界重试后必须拒答，不得输出未校验内容
    assert ans.refused is True


def test_e2e_trace_tree_complete():
    app, _ = _filled_app()
    ans, meta = app.ask("交通费报销上限", idempotency_key="trace-1")
    tree = meta["trace"]
    names = [s["name"] for s in tree["flat"]]
    for expected in ("route", "retrieve", "generate", "guard", "memory_write"):
        assert expected in names
    assert tree["total_ms"] > 0
