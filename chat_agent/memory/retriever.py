"""长期记忆检索器。

MemoryRetriever 负责在线召回、重排和解释长期记忆命中结果。当前在线链路不再读取
Markdown 记忆文件，只依赖 SQLite 中的长期记忆、摘要与可选向量索引。
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any

from chat_agent.agent.provider import LLMProvider
from chat_agent.memory.embeddings import EmbeddingProvider
from chat_agent.memory.store import SQLiteStore, re_split_query
from chat_agent.memory.vector_store import VectorStore

logger = logging.getLogger(__name__)


class MemoryRetriever:
    """长期记忆检索器。"""

    def __init__(
        self,
        store: SQLiteStore,
        enabled: bool = True,
        fast_provider: LLMProvider | None = None,
        query_rewrite_enabled: bool = False,
        hyde_enabled: bool = False,
        embedding_provider: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
        vector_top_k: int = 5,
        vector_min_score: float = 0.2,
    ) -> None:
        """初始化 `MemoryRetriever` 实例。

        参数:
            store: 初始化 `MemoryRetriever` 时需要的 `store` 参数。
            enabled: 初始化 `MemoryRetriever` 时需要的 `enabled` 参数。
            fast_provider: 初始化 `MemoryRetriever` 时需要的 `fast_provider` 参数。
            query_rewrite_enabled: 初始化 `MemoryRetriever` 时需要的 `query_rewrite_enabled` 参数。
            hyde_enabled: 初始化 `MemoryRetriever` 时需要的 `hyde_enabled` 参数。
            embedding_provider: 初始化 `MemoryRetriever` 时需要的 `embedding_provider` 参数。
            vector_store: 初始化 `MemoryRetriever` 时需要的 `vector_store` 参数。
            vector_top_k: 初始化 `MemoryRetriever` 时需要的 `vector_top_k` 参数。
            vector_min_score: 初始化 `MemoryRetriever` 时需要的 `vector_min_score` 参数。
        """
        self.store = store
        self.enabled = enabled
        self.fast_provider = fast_provider
        self.query_rewrite_enabled = query_rewrite_enabled
        self.hyde_enabled = hyde_enabled
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.vector_top_k = vector_top_k
        self.vector_min_score = vector_min_score
        self.last_trace: dict[str, Any] = {}

    async def retrieve(self, chat_id: str, query: str, top_k: int) -> list[dict[str, Any]]:
        """召回并重排与当前问题相关的长期记忆。"""
        self.last_trace = {"route_decision": False, "hyde_used": False, "candidates_considered": 0}
        if not self.enabled:
            return []

        should_retrieve = await self._route(query)
        self.last_trace["route_decision"] = should_retrieve
        logger.info("Memory route_decision=%s", should_retrieve)
        if not should_retrieve:
            return []

        rewritten = await self._rewrite(query)
        hyde = await self._hyde(query) if self.hyde_enabled else ""
        self.last_trace["hyde_used"] = bool(hyde)
        logger.info("Memory rewritten_query=%r hyde_generated=%s", rewritten, bool(hyde))

        candidates: dict[int, dict[str, Any]] = {}
        for candidate_query in _unique_queries([query, rewritten, hyde]):
            keyword_hits = await self.store.search_memories(chat_id, candidate_query, limit=20)
            vector_hits = await self._vector_search(chat_id, candidate_query, top_k=20)
            for item in keyword_hits + vector_hits:
                memory_id = int(item["id"])
                current = candidates.get(memory_id)
                if current is None:
                    current = dict(item)
                    current["_keyword_score"] = 0.0
                    current["_vector_score"] = 0.0
                    candidates[memory_id] = current
                else:
                    current.update({k: v for k, v in item.items() if not k.startswith("_")})
                current["_keyword_score"] = max(
                    float(current.get("_keyword_score", 0.0)),
                    _keyword_match_score(current, candidate_query),
                )
                current["_vector_score"] = max(float(current.get("_vector_score", 0.0)), float(item.get("_vector_score", 0.0)))

        self.last_trace["candidates_considered"] = len(candidates)
        reranked: list[dict[str, Any]] = []
        for item in candidates.values():
            scored = dict(item)
            importance = _clip01(float(scored.get("importance", 0.0)))
            reinforcement_score = min(math.log1p(max(int(scored.get("reinforcement", 1)), 0)) / math.log(6), 1.0)
            recency_score = _recency_score(scored)
            keyword_score = _clip01(float(scored.get("_keyword_score", 0.0)))
            vector_score = _clip01(float(scored.get("_vector_score", 0.0)))
            match_score = (
                vector_score * 0.45
                + keyword_score * 0.30
                + importance * 0.10
                + reinforcement_score * 0.10
                + recency_score * 0.05
            )
            match_reason = _match_reason(keyword_score, vector_score, str(scored.get("source_kind") or "inferred"), match_score)
            threshold = 0.35 if match_reason == "explicit_low_threshold" else 0.45
            scored["_match_score"] = round(match_score, 4)
            scored["_match_reason"] = match_reason
            scored["_rerank_features"] = {
                "keyword_score": round(keyword_score, 4),
                "vector_score": round(vector_score, 4),
                "importance": round(importance, 4),
                "reinforcement_score": round(reinforcement_score, 4),
                "recency_score": round(recency_score, 4),
                "threshold": threshold,
            }
            if match_score >= threshold:
                reranked.append(scored)

        reranked.sort(key=lambda item: (float(item["_match_score"]), float(item.get("importance", 0.0)), int(item.get("reinforcement", 1))), reverse=True)
        result = reranked[:top_k]
        logger.info("Memory memory_hits=%s considered=%s", len(result), len(candidates))
        return result

    async def _vector_search(self, chat_id: str, query: str, top_k: int) -> list[dict[str, Any]]:
        """处理`search`。

        参数:
            chat_id: 参与处理`search`的 `chat_id` 参数。
            query: 参与处理`search`的 `query` 参数。
            top_k: 参与处理`search`的 `top_k` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        if not self.embedding_provider or not self.vector_store:
            return []
        try:
            embedding = await self.embedding_provider.embed(query)
            if not embedding:
                return []
            hits = await self.vector_store.search(
                chat_id,
                embedding,
                top_k=min(top_k, max(self.vector_top_k, top_k)),
                min_score=self.vector_min_score,
            )
            logger.info("Memory vector_hits=%s", len(hits))
            return hits
        except NotImplementedError as exc:
            logger.warning("Memory vector search unavailable: %s", exc)
        except Exception:
            logger.exception("Memory vector search failed")
        return []

    async def _route(self, query: str) -> bool:
        """路由相关逻辑。

        参数:
            query: 参与路由相关逻辑的 `query` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        if not self.fast_provider:
            return True
        try:
            result = await self.fast_provider.chat(
                [
                    {"role": "system", "content": "判断用户问题是否需要检索长期记忆。只回答 yes 或 no。"},
                    {"role": "user", "content": query},
                ]
            )
            if not result.ok:
                return True
            return "no" not in result.content.lower()
        except Exception:
            logger.exception("Memory route failed; fallback to retrieve")
            return True

    async def _rewrite(self, query: str) -> str:
        """处理相关逻辑。

        参数:
            query: 参与处理相关逻辑的 `query` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        if not self.query_rewrite_enabled or not self.fast_provider:
            return query
        try:
            result = await self.fast_provider.chat(
                [
                    {"role": "system", "content": "把用户问题改写成适合检索长期记忆的简短关键词查询。只输出查询。"},
                    {"role": "user", "content": query},
                ]
            )
            if result.ok and result.content.strip():
                return result.content.strip()
        except Exception:
            logger.exception("Memory query rewrite failed")
        return query

    async def _hyde(self, query: str) -> str:
        """处理相关逻辑。

        参数:
            query: 参与处理相关逻辑的 `query` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        if not self.fast_provider:
            return ""
        try:
            result = await self.fast_provider.chat(
                [
                    {"role": "system", "content": "生成一条假想长期记忆，如果真实记忆存在，它可能会怎样描述。只输出一句话。"},
                    {"role": "user", "content": query},
                ]
            )
            if result.ok:
                return result.content.strip()
        except Exception:
            logger.exception("Memory HyDE failed")
        return ""


def _unique_queries(queries: list[str]) -> list[str]:
    """处理查询列表。

    参数:
        queries: 参与处理查询列表的 `queries` 参数。

    返回:
        返回与本函数处理结果对应的数据。
    """
    result: list[str] = []
    seen: set[str] = set()
    for query in queries:
        text = query.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _keyword_match_score(item: dict[str, Any], query: str) -> float:
    """处理`match`、`score`。

    参数:
        item: 参与处理`match`、`score`的 `item` 参数。
        query: 参与处理`match`、`score`的 `query` 参数。

    返回:
        返回与本函数处理结果对应的数据。
    """
    terms = [term.lower() for term in re_split_query(query) if term.strip()]
    if not terms:
        return 0.0
    haystacks = [
        str(item.get("content") or "").lower(),
        " ".join(str(tag).lower() for tag in item.get("tags") or []),
        str(item.get("type") or "").lower(),
    ]
    matched = 0
    for term in terms:
        if any(term in haystack for haystack in haystacks):
            matched += 1
    return matched / len(terms)


def _recency_score(item: dict[str, Any]) -> float:
    """处理`score`。

    参数:
        item: 参与处理`score`的 `item` 参数。

    返回:
        返回与本函数处理结果对应的数据。
    """
    stamp = item.get("updated_at") or item.get("last_used_at") or item.get("created_at")
    if not stamp:
        return 0.0
    try:
        dt = datetime.fromisoformat(str(stamp)).astimezone(timezone.utc)
    except ValueError:
        return 0.0
    age = datetime.now(timezone.utc) - dt
    if age <= timedelta(days=7):
        return 1.0
    if age <= timedelta(days=30):
        return 0.5
    return 0.0


def _match_reason(keyword_score: float, vector_score: float, source_kind: str, final_score: float) -> str:
    """处理`reason`。

    参数:
        keyword_score: 参与处理`reason`的 `keyword_score` 参数。
        vector_score: 参与处理`reason`的 `vector_score` 参数。
        source_kind: 参与处理`reason`的 `source_kind` 参数。
        final_score: 参与处理`reason`的 `final_score` 参数。

    返回:
        返回与本函数处理结果对应的数据。
    """
    if source_kind == "explicit" and final_score >= 0.35 and final_score < 0.45:
        return "explicit_low_threshold"
    if keyword_score > 0 and vector_score > 0:
        return "keyword+vector"
    if vector_score > 0:
        return "vector_only"
    return "keyword_only"


def _clip01(value: float) -> float:
    """处理相关逻辑。

    参数:
        value: 参与处理相关逻辑的 `value` 参数。

    返回:
        返回与本函数处理结果对应的数据。
    """
    return max(0.0, min(value, 1.0))
