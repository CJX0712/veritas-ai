"""M09 router — 复杂度/预算感知路由（C3）。

本机实测约束（ADR-003）：1.5b 约 2.6 tok/s，7b 不可作为默认档。
档位：RULES（短/缓存类） -> ENHANCED（检索增强 + 1.5b） -> LARGE（仅手动）。

不变量：
  INV-RT-001 任何输入都必须产出合法档位（不抛异常）
  INV-RT-002 超预算请求必须携带 downgraded_from 降级标记
"""
from __future__ import annotations

import hashlib
import math

from .contracts import RouteDecision, RouteTier

FAQ_MAX_LEN = 12          # 超短查询走规则档
LARGE_TIER_TOKENS = 3.0   # 1.5b tok/s 实测下限，用于预算估算


def complexity_score(query: str) -> float:
    """确定性复杂度评分：长度 + 疑问结构 + 条件词。0~1。"""
    n = len(query.strip())
    if n == 0:
        return 0.0
    length_part = min(n / 60.0, 1.0)
    conditional = any(w in query for w in ("如果", "是否", "周末", "超过", "多久", "多少", "区别"))
    multi = query.count("？") + query.count("?") > 1 or "，" in query
    return min(0.25 + 0.5 * length_part + 0.15 * conditional + 0.1 * multi, 1.0)


class Router:
    def __init__(self, enable_large: bool = False,
                 budget_ms: dict[str, int] | None = None) -> None:
        self._enable_large = enable_large
        self._budget = budget_ms or {
            "rules": 200, "small": 6000, "enhanced": 8000, "large": 25000,
        }

    def route(self, query: str, force_tier: str | None = None,
              elapsed_ms: float = 0.0) -> RouteDecision:
        score = complexity_score(query)
        if force_tier == "large" and self._enable_large:
            tier = RouteTier.LARGE
        elif force_tier:
            tier = RouteTier(force_tier)
        elif score < 0.30 and len(query.strip()) <= FAQ_MAX_LEN:
            tier = RouteTier.RULES
        else:
            tier = RouteTier.ENHANCED

        budget = self._budget[tier.value]
        reasons = [f"complexity={score:.2f}", f"len={len(query.strip())}"]
        if elapsed_ms > budget:
            downgraded = RouteTier.RULES if tier != RouteTier.RULES else None
            reasons.append(f"elapsed={elapsed_ms:.0f}ms>{budget}ms, degrade")
            return RouteDecision(
                tier=RouteTier.RULES if downgraded else tier,
                complexity=score, latency_budget_ms=budget,
                reasons=reasons, downgraded_from=tier if downgraded else None,
            )
        reasons.append(f"budget={budget}ms")
        return RouteDecision(tier=tier, complexity=score,
                             latency_budget_ms=budget, reasons=reasons)

    def estimate_tokens(self, query: str) -> int:
        """按实测吞吐估算 1.5b 生成预算（用于预算内答案长度控制）。"""
        return max(32, int(self._budget["enhanced"] / 1000 * LARGE_TIER_TOKENS
                           * math.sqrt(complexity_score(query) + 0.5)))


def query_hash(query: str) -> str:
    return hashlib.md5(query.strip().encode("utf-8")).hexdigest()[:12]
