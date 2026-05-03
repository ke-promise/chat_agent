"""HTTP reranker client for memory retrieval."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import aiohttp

from chat_agent.config import RerankerConfig

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RerankResult:
    """单条 rerank 结果。"""

    index: int
    score: float


class HttpReranker:
    """OpenAI 风格 HTTP reranker。

    请求固定为 POST {base_url}/rerank，body 包含 model、query、documents、top_n。
    """

    def __init__(self, config: RerankerConfig) -> None:
        self.config = config

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled and self.config.model and self.config.base_url)

    async def rerank(self, query: str, documents: list[str], top_n: int | None = None) -> list[RerankResult]:
        """调用远端 reranker 并返回按相关性排序的 index/score。"""
        if not self.enabled or not query.strip() or not documents:
            return []
        url = self.config.base_url.rstrip("/") + "/rerank"
        payload = {
            "model": self.config.model,
            "query": query,
            "documents": documents,
            "top_n": top_n or self.config.top_n or len(documents),
        }
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload, headers=headers) as response:
                response.raise_for_status()
                data = await response.json()
        return _parse_rerank_response(data, len(documents))


def _parse_rerank_response(data: Any, document_count: int) -> list[RerankResult]:
    """解析 OpenAI 风格 rerank 响应。"""
    if not isinstance(data, dict):
        return []
    raw_results = data.get("results")
    if not isinstance(raw_results, list):
        return []
    results: list[RerankResult] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("index"))
            score = float(item.get("relevance_score", item.get("score", 0.0)))
        except (TypeError, ValueError):
            continue
        if 0 <= index < document_count:
            results.append(RerankResult(index=index, score=score))
    results.sort(key=lambda item: item.score, reverse=True)
    return results
