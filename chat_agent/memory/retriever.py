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
        """初始化长期记忆检索器。

        参数:
            store: 长期记忆和会话状态的 SQLite 存储。
            enabled: 是否启用在线记忆召回。
            fast_provider: 轻量模型，用于路由、query rewrite 和 HyDE。
            query_rewrite_enabled: 是否启用检索查询改写。
            hyde_enabled: 是否生成假想记忆辅助召回。
            embedding_provider: 可选 embedding provider。
            vector_store: 可选向量存储。
            vector_top_k: 向量召回的参考数量。
            vector_min_score: 向量命中的最低相似度。
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
        """对查询文本执行向量召回，并把命中回表成记忆条目。

        参数:
            chat_id: 当前会话 ID。
            query: 待向量化的查询文本。
            top_k: 本次最多返回多少条候选。

        返回:
            带 `_vector_score` 的记忆条目列表；向量服务不可用时返回空列表。
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
        """判断当前用户输入是否值得检索长期记忆。

        参数:
            query: 用户当前输入。

        返回:
            True 表示继续检索；False 表示跳过记忆召回。
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
        """把用户原话改写成更适合检索长期记忆的短查询。

        参数:
            query: 用户当前输入。

        返回:
            改写后的查询；失败或未启用时返回原查询。
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
        """生成一条假想长期记忆，用于扩展召回查询。

        参数:
            query: 用户当前输入。

        返回:
            假想记忆文本；失败时返回空字符串。
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
    """去除空查询和重复查询，并保留原始顺序。

    参数:
        queries: 待合并的查询文本列表。

    返回:
        去重后的非空查询列表。
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
    """计算查询词在记忆正文、标签和类型中的关键词覆盖率。

    参数:
        item: 候选记忆条目。
        query: 当前检索查询。

    返回:
        0 到 1 之间的关键词匹配分数。
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
    """根据记忆更新时间估算新鲜度分数。

    参数:
        item: 候选记忆条目。

    返回:
        0、0.5 或 1.0 的新鲜度分数。
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
    """根据重排特征生成便于 trace 展示的命中原因。

    参数:
        keyword_score: 关键词匹配分。
        vector_score: 向量相似度分。
        source_kind: 记忆来源类型，例如 explicit 或 inferred。
        final_score: 最终重排分。

    返回:
        命中原因字符串。
    """
    if source_kind == "explicit" and final_score >= 0.35 and final_score < 0.45:
        return "explicit_low_threshold"
    if keyword_score > 0 and vector_score > 0:
        return "keyword+vector"
    if vector_score > 0:
        return "vector_only"
    return "keyword_only"


def _clip01(value: float) -> float:
    """把浮点数裁剪到 0 到 1 区间。

    参数:
        value: 原始浮点数。

    返回:
        裁剪后的数值。
    """
    return max(0.0, min(value, 1.0))
