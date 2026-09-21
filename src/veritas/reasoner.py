"""M08 reasoner — 生成与引用绑定。JSON 输出 + schema 校验 + 有界重试。

不变量：
  INV-GEN-001 每条断言的 citation 必须指向本次检索返回的 chunk（解析层强制）
  INV-GEN-002 同输入同配置必得同输出（temperature=0 + seed，PoT 确定性）
"""
from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from .contracts import (
    Answer,
    Assertion,
    ChatMessage,
    Citation,
    Hit,
    LLMClient,
    RouteTier,
)
from .llm import extract_context
from .telemetry import SpanContext

MAX_RETRIES = 2
MAX_ANSWER_MS = {"rules": 200, "small": 6000, "enhanced": 6000, "large": 25000}


class AssertionPayload(BaseModel):
    """LLM 输出的 schema 契约（INV-GEN 之上的形状校验）。"""

    text: str = Field(min_length=4)
    citation: str | None = None


class Reasoner:
    def __init__(self, llm: LLMClient, parent_span: SpanContext | None = None) -> None:
        self._llm = llm
        self._span = parent_span

    @property
    def model_name(self) -> str:
        return self._llm.model_name

    def generate(self, query: str, hits: list[Hit], tier: RouteTier) -> Answer:
        """从检索命中生成逐句带引用的答案。JSON 解析失败 -> 有界重试 -> 拒答。"""
        if tier == RouteTier.RULES or not hits:
            return self._refuse(query, hits, "no_evidence")
        prompt = self._build_prompt(query, hits)
        messages = [ChatMessage(role="user", content=prompt)]
        last_err = ""
        for _attempt in range(MAX_RETRIES + 1):
            resp = self._llm.chat(messages)
            parsed, err = self._parse(resp.content)
            if parsed is None:
                last_err = err
                messages.append(ChatMessage(role="assistant", content=resp.content[:200]))
                messages.append(ChatMessage(
                    role="user",
                    content=('输出不合法，重新输出。只输出 JSON：'
                             '{"assertions":[{"text":"...","citation":"chunk_id"}]}'),
                ))
                continue
            valid = self._bind_citations(parsed, hits)
            if valid:
                return self._compose(query, valid, hits, tier)
            last_err = "citations_out_of_scope"
        return self._refuse(query, hits, f"schema:{last_err}")

    def _build_prompt(self, query: str, hits: list[Hit]) -> str:
        lines = [f"问题：{query}", "仅依据以下证据回答，每句必须标注来源编号：", ""]
        for h in hits:
            lines.append(f"CITE|{h.chunk.chunk_id}|{h.chunk.text}")
        lines += [
            "",
            '只输出 JSON，不要其他文字：{"assertions":[{"text":"句子","citation":"chunk_id"}]}',
            "若证据不足，输出 {\"assertions\":[]}",
        ]
        return "\n".join(lines)

    @staticmethod
    def _parse(content: str) -> tuple[list[AssertionPayload] | None, str]:
        try:
            data: dict[str, Any] = json.loads(content)
            items = data.get("assertions")
            if not isinstance(items, list):
                return None, "assertions_not_list"
            return [AssertionPayload.model_validate(i) for i in items], ""
        except (json.JSONDecodeError, ValidationError, AttributeError) as exc:
            return None, type(exc).__name__

    @staticmethod
    def _bind_citations(items: list[AssertionPayload],
                        hits: list[Hit]) -> list[Assertion]:
        """citation 必须指向本次检索命中的 chunk（INV-GEN-001）。"""
        valid_ids = {h.chunk.chunk_id for h in hits}
        idx_of = {h.chunk.chunk_id: i + 1 for i, h in enumerate(hits)}
        out: list[Assertion] = []
        for item in items:
            cid = item.citation
            if cid is None or cid not in valid_ids:
                continue
            out.append(Assertion(text=item.text, citation_index=idx_of[cid]))
        return out

    def _compose(self, query: str, assertions: list[Assertion],
                 hits: list[Hit], tier: RouteTier) -> Answer:
        citations: list[Citation] = []
        for idx, h in enumerate(hits):
            citations.append(Citation(
                index=idx + 1, chunk_id=h.chunk.chunk_id, doc_id=h.chunk.doc_id,
                section=h.chunk.section, snippet=h.chunk.text[:160],
                score=h.score,
            ))
        return Answer(
            query=query, trace_id="", assertions=assertions, citations=citations,
            model=self.model_name,
            route=None if tier == RouteTier.RULES else None,
        )

    def _refuse(self, query: str, hits: list[Hit], reason: str) -> Answer:
        citations = [
            Citation(index=i + 1, chunk_id=h.chunk.chunk_id, doc_id=h.chunk.doc_id,
                     section=h.chunk.section, snippet=h.chunk.text[:160], score=h.score)
            for i, h in enumerate(hits)
        ]
        return Answer(query=query, trace_id="", refused=True,
                      refuse_reason=reason, citations=citations,
                      model=self.model_name)


def context_marker_used(prompt: str) -> bool:
    return bool(extract_context(prompt))
