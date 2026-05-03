"""长期记忆检索器。

在线召回链路使用向量检索、SQLite FTS5 BM25、RRF 融合和可选 HTTP reranker。
Markdown 记忆文件只作为审计导出，不参与在线 prompt 构建。
"""

from __future__ import annotations

import logging
from typing import Any

from chat_agent.agent.provider import LLMProvider
from chat_agent.memory.embeddings import EmbeddingProvider
from chat_agent.memory.reranker import HttpReranker
from chat_agent.memory.store import SQLiteStore
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
        vector_top_k: int = 50,
        vector_min_score: float = 0.2,
        bm25_top_k: int = 50,
        rrf_top_k: int = 20,
        rrf_k: int = 60,
        reranker: HttpReranker | None = None,
    ) -> None:
        self.store = store
        self.enabled = enabled
        self.fast_provider = fast_provider
        self.query_rewrite_enabled = query_rewrite_enabled
        self.hyde_enabled = hyde_enabled
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.vector_top_k = vector_top_k
        self.vector_min_score = vector_min_score
        self.bm25_top_k = bm25_top_k
        self.rrf_top_k = rrf_top_k
        self.rrf_k = rrf_k
        self.reranker = reranker
        self.last_trace: dict[str, Any] = {}

    async def retrieve(self, chat_id: str, query: str, top_k: int) -> list[dict[str, Any]]:
        """召回并重排与当前问题相关的长期记忆。"""
        self.last_trace = {
            "route_decision": False,
            "hyde_used": False,
            "candidates_considered": 0,
            "bm25_hits": 0,
            "vector_hits": 0,
            "rrf_candidates": 0,
            "rerank_used": False,
            "rerank_error": "",
        }
        if not self.enabled:
            return []
        if not query.strip():
            return await self._recent_memories(chat_id, top_k)

        should_retrieve = await self._route(query)
        self.last_trace["route_decision"] = should_retrieve
        logger.info("Memory route_decision=%s", should_retrieve)
        if not should_retrieve:
            return []

        rewritten = await self._rewrite(query)
        hyde = await self._hyde(query) if self.hyde_enabled else ""
        self.last_trace["hyde_used"] = bool(hyde)
        logger.info("Memory rewritten_query=%r hyde_generated=%s", rewritten, bool(hyde))

        ranked_lists: list[list[dict[str, Any]]] = []
        candidates: dict[int, dict[str, Any]] = {}
        for candidate_query in _unique_queries([query, rewritten, hyde]):
            bm25_hits = await self.store.search_bm25_memories(chat_id, candidate_query, limit=self.bm25_top_k)
            vector_hits = await self._vector_search(chat_id, candidate_query, top_k=self.vector_top_k)
            self.last_trace["bm25_hits"] += len(bm25_hits)
            self.last_trace["vector_hits"] += len(vector_hits)
            ranked_lists.extend([bm25_hits, vector_hits])
            for item in bm25_hits:
                current = _merge_candidate(candidates, item)
                current["_bm25_score"] = max(float(current.get("_bm25_score", 0.0)), float(item.get("_bm25_score", 0.0)))
            for item in vector_hits:
                current = _merge_candidate(candidates, item)
                current["_vector_score"] = max(float(current.get("_vector_score", 0.0)), float(item.get("_vector_score", 0.0)))

        self.last_trace["candidates_considered"] = len(candidates)
        fused_ids = _rrf_fuse(ranked_lists, self.rrf_k)
        fused: list[dict[str, Any]] = []
        for memory_id, score in fused_ids[: self.rrf_top_k]:
            item = candidates.get(memory_id)
            if not item:
                continue
            scored = dict(item)
            scored["_rrf_score"] = round(score, 6)
            scored["_match_score"] = round(score, 6)
            scored["_match_reason"] = _match_reason(scored, reranked=False)
            scored["_rerank_features"] = {
                "bm25_score": round(float(scored.get("_bm25_score", 0.0)), 6),
                "vector_score": round(float(scored.get("_vector_score", 0.0)), 6),
                "rrf_score": round(score, 6),
                "rerank_score": None,
            }
            fused.append(scored)

        self.last_trace["rrf_candidates"] = len(fused)
        if not fused:
            return []

        result = await self._rerank(query, fused, top_k)
        logger.info("Memory memory_hits=%s considered=%s rrf_candidates=%s", len(result), len(candidates), len(fused))
        return result

    async def _recent_memories(self, chat_id: str, top_k: int) -> list[dict[str, Any]]:
        """空查询时返回最近/重要记忆，并补齐统一 trace 字段。"""
        memories = await self.store.list_recent_memories(chat_id, limit=top_k)
        for index, item in enumerate(memories):
            score = 1.0 / (index + 1)
            item["_rrf_score"] = score
            item["_match_score"] = score
            item["_match_reason"] = "recent"
            item["_rerank_features"] = {"rrf_score": score, "rerank_score": None}
        self.last_trace["candidates_considered"] = len(memories)
        self.last_trace["rrf_candidates"] = len(memories)
        return memories

    async def _rerank(self, query: str, fused: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """用 HTTP reranker 精排；不可用时降级为 RRF 排序。"""
        fallback = sorted(fused, key=lambda item: float(item.get("_rrf_score", 0.0)), reverse=True)[:top_k]
        if not self.reranker or not self.reranker.enabled:
            return fallback
        try:
            documents = [str(item.get("content") or "") for item in fused]
            reranked = await self.reranker.rerank(query, documents, top_n=min(len(fused), max(top_k, 1)))
        except Exception as exc:
            logger.warning("Memory rerank failed; falling back to RRF: %s", exc)
            self.last_trace["rerank_error"] = str(exc)
            return fallback
        if not reranked:
            return fallback

        self.last_trace["rerank_used"] = True
        result: list[dict[str, Any]] = []
        used: set[int] = set()
        for entry in reranked:
            if entry.index in used or entry.index >= len(fused):
                continue
            used.add(entry.index)
            item = dict(fused[entry.index])
            item["_rerank_score"] = entry.score
            item["_match_score"] = round(entry.score, 6)
            item["_match_reason"] = _match_reason(item, reranked=True)
            features = dict(item.get("_rerank_features") or {})
            features["rerank_score"] = round(entry.score, 6)
            item["_rerank_features"] = features
            result.append(item)
            if len(result) >= top_k:
                break
        if len(result) < top_k:
            for item in fallback:
                memory_id = int(item["id"])
                if memory_id not in {int(existing["id"]) for existing in result}:
                    result.append(item)
                if len(result) >= top_k:
                    break
        return result[:top_k]

    async def _vector_search(self, chat_id: str, query: str, top_k: int) -> list[dict[str, Any]]:
        """对查询文本执行向量召回，并把命中回表成记忆条目。"""
        if not self.embedding_provider or not self.vector_store:
            return []
        try:
            embedding = await self.embedding_provider.embed(query)
            if not embedding:
                return []
            hits = await self.vector_store.search(
                chat_id,
                embedding,
                top_k=top_k,
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
        """判断当前用户输入是否值得检索长期记忆。"""
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
        """把用户原话改写成更适合检索长期记忆的短查询。"""
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
        """生成一条假想长期记忆，用于扩展召回查询。"""
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


def _merge_candidate(candidates: dict[int, dict[str, Any]], item: dict[str, Any]) -> dict[str, Any]:
    """按 memory_id 合并召回候选。"""
    memory_id = int(item["id"])
    current = candidates.get(memory_id)
    if current is None:
        current = dict(item)
        current["_bm25_score"] = float(item.get("_bm25_score", 0.0))
        current["_vector_score"] = float(item.get("_vector_score", 0.0))
        candidates[memory_id] = current
    else:
        current.update({k: v for k, v in item.items() if not k.startswith("_")})
    return current


def _rrf_fuse(ranked_lists: list[list[dict[str, Any]]], k: int) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion。"""
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            memory_id = int(item["id"])
            scores[memory_id] = scores.get(memory_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def _unique_queries(queries: list[str]) -> list[str]:
    """去除空查询和重复查询，并保留原始顺序。"""
    result: list[str] = []
    seen: set[str] = set()
    for query in queries:
        text = query.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _match_reason(item: dict[str, Any], reranked: bool) -> str:
    """生成 trace 中的命中原因。"""
    if reranked:
        return "bge_rerank"
    has_bm25 = float(item.get("_bm25_score", 0.0)) > 0
    has_vector = float(item.get("_vector_score", 0.0)) > 0
    if has_bm25 and has_vector:
        return "rrf_bm25+vector"
    if has_vector:
        return "rrf_vector"
    if has_bm25:
        return "rrf_bm25"
    return "rrf"
