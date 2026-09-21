"""M07 llm — LLM 客户端 Protocol + Ollama 实现 + 确定性 Mock（离线轨/CI）。

关键设计：
  - Ollama 不用 format=json（本机实测会超时）；改用提示词约束 + reasoner 后校验重试。
  - MockLLM 从 prompt 中的 CITE 上下文标记确定性生成断言，并支持注入失败模式
    （hallucinate / schema_violate / refuse），供 guardian 的"噪声必被捕获"测试使用。
  - PoT/生成确定性：Ollana options temperature=0, seed=42。
"""
from __future__ import annotations

import json
import re
from typing import Any

import httpx

from .contracts import ChatMessage, ChatResponse, ToolSpec
from .telemetry import get_logger

log = get_logger(__name__)

_CTX_RE = re.compile(r"^CITE\|([^|]+)\|(.*)$", re.MULTILINE)


def extract_context(prompt: str) -> list[tuple[str, str]]:
    """从 prompt 提取 CITE|chunk_id|text 上下文行（MockLLM 的知识来源）。"""
    return _CTX_RE.findall(prompt)


class OllamaLLM:
    def __init__(self, model: str = "qwen2.5:1.5b-instruct",
                 base_url: str = "http://127.0.0.1:11434",
                 timeout: float = 90.0,
                 seed: int = 42) -> None:
        self._model = model
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._seed = seed

    @property
    def model_name(self) -> str:
        return self._model

    def chat(self, messages: list[ChatMessage],
             tools: list[ToolSpec] | None = None) -> ChatResponse:
        body: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {"temperature": 0, "seed": self._seed},
        }
        if tools:
            body["tools"] = [
                {"type": "function",
                 "function": {"name": t.name, "description": t.description,
                              "parameters": t.parameters}}
                for t in tools
            ]
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(f"{self._base}/api/chat", json=body)
                resp.raise_for_status()
                data: dict[str, Any] = resp.json()
        except httpx.HTTPError as exc:
            log.error("ollama_chat_failed", err=str(exc))
            raise RuntimeError(f"ollama chat failed: {exc}") from exc
        msg = data.get("message", {})
        return ChatResponse(
            content=msg.get("content", ""),
            tool_calls=msg.get("tool_calls") or [],
            model=data.get("model", self._model),
            eval_count=int(data.get("eval_count") or 0),
        )


class MockLLM:
    """确定性离线 LLM。

    行为：
      - prompt 含 CITE 上下文 -> 从上下文生成逐句断言（每句绑定其来源 chunk）。
      - mode="hallucinate"    -> 生成一条绑定不存在 chunk 的断言（噪声）。
      - mode="schema_violate" -> 输出非 JSON（schema 噪声）。
      - mode="refuse"         -> 空上下文 -> 空断言（触发拒答路径）。
    同输入必同输出：answer JSON 的构造完全确定。
    """

    def __init__(self, mode: str = "normal") -> None:
        self._mode = mode

    @property
    def model_name(self) -> str:
        return f"mock-{self._mode}"

    def chat(self, messages: list[ChatMessage],
             tools: list[ToolSpec] | None = None) -> ChatResponse:
        prompt = messages[-1].content if messages else ""
        ctx = extract_context(prompt)
        if self._mode == "schema_violate":
            return ChatResponse(content="这不是JSON", model=self.model_name,
                                eval_count=3)
        if self._mode == "hallucinate":
            payload = {"assertions": [
                {"text": "此断言没有任何上下文支撑，属于幻觉测试噪声。",
                 "citation": "chunk_does_not_exist#0"},
            ]}
            return ChatResponse(content=json.dumps(payload, ensure_ascii=False),
                                model=self.model_name, eval_count=10)
        if not ctx:
            empty: dict[str, Any] = {"assertions": []}
            return ChatResponse(content=json.dumps(empty, ensure_ascii=False),
                                model=self.model_name, eval_count=2)

        assertions: list[dict[str, Any]] = []
        for cid, text in ctx[:3]:
            for sent in _split_sentences(text)[:2]:
                assertions.append({"text": sent, "citation": cid})
        payload = {"assertions": assertions}
        return ChatResponse(content=json.dumps(payload, ensure_ascii=False),
                            model=self.model_name, eval_count=len(assertions) * 8)


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？!?；;])\s*", text.strip())
    return [p.strip() for p in parts if len(p.strip()) >= 6]
