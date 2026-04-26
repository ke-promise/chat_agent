"""长期记忆向量存储适配层。

SQLiteStore 仍然保存完整 memory 正文和业务元数据；本模块只负责把 memory_id 与
embedding 写入可检索的向量后端。当前支持轻量 sqlite_json 和外部 Chroma，
其他 provider 先通过占位类明确提示“尚未实现”。
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlparse

from chat_agent.config import EmbeddingConfig
from chat_agent.memory.store import SQLiteStore

logger = logging.getLogger(__name__)


class VectorStore(Protocol):
    """向量存储接口。

    第一阶段由 SQLiteJsonVectorStore 实现；第二阶段可新增 QdrantVectorStore、
    ChromaVectorStore 或 PgVectorStore，并保持 MemoryRetriever 调用方式不变。
    """

    async def upsert_memory(self, chat_id: str, memory_id: int, embedding: list[float]) -> None:
        """写入或更新某条 memory 的向量。"""

    async def search(self, chat_id: str, query_embedding: list[float], top_k: int, min_score: float) -> list[dict]:
        """按向量相似度搜索 memory。"""


class SQLiteJsonVectorStore:
    """SQLite JSON 向量存储。

    适用阶段:
        第一阶段。个人助手记忆量不大时，用 SQLite 保存 JSON 向量并在 Python 中计算余弦相似度，
        不需要额外部署外部服务。
    """

    def __init__(self, store: SQLiteStore) -> None:
        """初始化 SQLite JSON 向量存储。

        Args:
            store: 主 SQLiteStore，用于读写 memory_embeddings 表。
        """
        self.store = store

    async def upsert_memory(self, chat_id: str, memory_id: int, embedding: list[float]) -> None:
        """把 memory embedding 写入 SQLite。"""
        await self.store.upsert_memory_embedding(chat_id, memory_id, embedding)

    async def search(self, chat_id: str, query_embedding: list[float], top_k: int, min_score: float) -> list[dict]:
        """读取当前 chat 的所有 memory embedding 并做余弦相似度排序。"""
        rows = await self.store.list_memory_embeddings(chat_id)
        scored: list[tuple[float, int]] = []
        for row in rows:
            score = cosine_similarity(query_embedding, row["embedding"])
            if score >= min_score:
                scored.append((score, int(row["memory_id"])))
        scored.sort(reverse=True, key=lambda item: item[0])
        memory_ids = [memory_id for _, memory_id in scored[:top_k]]
        memories = await self.store.get_memories_by_ids(chat_id, memory_ids)
        by_id = {int(item["id"]): item for item in memories}
        result: list[dict] = []
        for score, memory_id in scored[:top_k]:
            if memory_id in by_id:
                item = dict(by_id[memory_id])
                item["_vector_score"] = score
                result.append(item)
        return result


class ExternalVectorStorePlaceholder:
    """第二阶段外部向量数据库占位实现。

    当前不会真正连接外部库。它让配置和依赖方向先稳定下来：以后只需要替换这个类，
    不需要重写 ContextBuilder、AgentLoop 或 Reasoner。
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        """保存外部向量库配置，调用时明确抛出未实现错误。"""
        self.config = config

    async def upsert_memory(self, chat_id: str, memory_id: int, embedding: list[float]) -> None:
        """占位写入接口；当前 provider 尚未接入。"""
        raise NotImplementedError(f"External vector provider {self.config.provider} is planned for stage 2")

    async def search(self, chat_id: str, query_embedding: list[float], top_k: int, min_score: float) -> list[dict]:
        """占位搜索接口；当前 provider 尚未接入。"""
        raise NotImplementedError(f"External vector provider {self.config.provider} is planned for stage 2")


@dataclass(frozen=True)
class ChromaConnection:
    """Chroma HTTP 连接参数。"""

    host: str
    port: int
    ssl: bool
    headers: dict[str, str] | None = None


class VectorStoreUnavailable(RuntimeError):
    """向量存储暂时不可用。"""


class ChromaVectorStore:
    """Chroma 外部向量数据库实现。

    存储策略:
        - SQLite 继续保存完整 memory 正文和元数据。
        - Chroma 只保存 memory_id、chat_id 和 embedding。
        - 查询时先从 Chroma 找到 memory_id，再回 SQLite 读取完整记忆。

    这样外部向量库只负责相似度检索，业务数据仍以 SQLite 为准，便于备份和迁移。
    """

    def __init__(self, config: EmbeddingConfig, store: SQLiteStore) -> None:
        """初始化 Chroma 向量存储。

        Args:
            config: embedding/vector_store 配置，包含 Chroma URL、collection 和密钥。
            store: 主 SQLiteStore，用于根据 Chroma 返回的 memory_id 回表读取正文。
        """
        self.config = config
        self.store = store
        self.connection = _parse_chroma_connection(config)
        self._collection = None
        self._retry_after = 0.0
        self._retry_cooldown_seconds = 60.0
        self._last_unavailable_message = ""
        self._last_warning_message = ""

    async def upsert_memory(self, chat_id: str, memory_id: int, embedding: list[float]) -> None:
        """把 memory embedding 写入 Chroma collection。"""
        try:
            await asyncio.to_thread(self._upsert_sync, chat_id, memory_id, embedding)
        except VectorStoreUnavailable as exc:
            self._warn_unavailable(exc)

    async def search(self, chat_id: str, query_embedding: list[float], top_k: int, min_score: float) -> list[dict]:
        """从 Chroma 检索相似 memory，再回 SQLite 读取完整条目。"""
        try:
            scored = await asyncio.to_thread(self._search_sync, chat_id, query_embedding, top_k, min_score)
        except VectorStoreUnavailable as exc:
            self._warn_unavailable(exc)
            return []
        memory_ids = [memory_id for _, memory_id in scored]
        memories = await self.store.get_memories_by_ids(chat_id, memory_ids)
        by_id = {int(item["id"]): item for item in memories}
        result: list[dict] = []
        for score, memory_id in scored:
            if memory_id in by_id:
                item = dict(by_id[memory_id])
                item["_vector_score"] = score
                result.append(item)
        return result

    def _collection_handle(self):
        """懒加载 Chroma collection。

        chromadb 是第二阶段外部向量库依赖，只有 provider=chroma 时才导入。
        """
        if self._collection is not None:
            return self._collection
        if time.monotonic() < self._retry_after:
            raise VectorStoreUnavailable(self._last_unavailable_message or self._unavailable_message())
        try:
            import chromadb
        except ImportError as exc:
            self._mark_unavailable(
                'Chroma support requires installing dependency: pip install "chromadb>=0.5"',
                exc,
            )

        try:
            client = chromadb.HttpClient(
                host=self.connection.host,
                port=self.connection.port,
                ssl=self.connection.ssl,
                headers=self.connection.headers,
            )
            self._collection = client.get_or_create_collection(
                name=self.config.collection,
                metadata={"hnsw:space": "cosine"},
            )
            self._retry_after = 0.0
            self._last_unavailable_message = ""
            self._last_warning_message = ""
            return self._collection
        except Exception as exc:
            self._mark_unavailable(self._unavailable_message(), exc)

    def _upsert_sync(self, chat_id: str, memory_id: int, embedding: list[float]) -> None:
        """同步写入 Chroma，供 asyncio.to_thread 调用。"""
        collection = self._collection_handle()
        collection.upsert(
            ids=[str(memory_id)],
            embeddings=[[float(value) for value in embedding]],
            metadatas=[{"chat_id": chat_id, "memory_id": memory_id}],
        )

    def _search_sync(self, chat_id: str, query_embedding: list[float], top_k: int, min_score: float) -> list[tuple[float, int]]:
        """同步查询 Chroma 并转换为 (score, memory_id) 列表。"""
        collection = self._collection_handle()
        response = collection.query(
            query_embeddings=[[float(value) for value in query_embedding]],
            n_results=top_k,
            where={"chat_id": chat_id},
            include=["metadatas", "distances"],
        )
        metadatas = response.get("metadatas", [[]])[0] if isinstance(response, dict) else [[]]
        distances = response.get("distances", [[]])[0] if isinstance(response, dict) else [[]]
        scored: list[tuple[float, int]] = []
        for metadata, distance in zip(metadatas, distances, strict=False):
            if not isinstance(metadata, dict):
                continue
            memory_id = int(metadata.get("memory_id") or 0)
            score = 1.0 - float(distance)
            if memory_id > 0 and score >= min_score:
                scored.append((score, memory_id))
        scored.sort(reverse=True, key=lambda item: item[0])
        return scored

    def _warn_unavailable(self, exc: Exception) -> None:
        """记录告警`unavailable`。

        参数:
            exc: 参与记录告警`unavailable`的 `exc` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        message = str(exc)
        if message and message != self._last_warning_message:
            logger.warning("%s", message)
            self._last_warning_message = message

    def _mark_unavailable(self, message: str, exc: Exception) -> None:
        """标记`unavailable`。

        参数:
            message: 参与标记`unavailable`的 `message` 参数。
            exc: 参与标记`unavailable`的 `exc` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        self._collection = None
        self._retry_after = time.monotonic() + self._retry_cooldown_seconds
        self._last_unavailable_message = message
        raise VectorStoreUnavailable(message) from exc

    def _unavailable_message(self) -> str:
        """处理消息。

        返回:
            返回与本函数处理结果对应的数据。"""
        target = self.config.external_url or "http://localhost:8000"
        return (
            f"Chroma server is unavailable at {target}. "
            'Start Chroma first or switch [embedding].provider to "sqlite_json".'
        )


def create_vector_store(config: EmbeddingConfig, store: SQLiteStore) -> VectorStore:
    """根据配置创建向量存储。

    参数:
        config: embedding 配置。
        store: SQLiteStore，sqlite_json 阶段复用主数据库。
    """
    if config.provider == "sqlite_json":
        return SQLiteJsonVectorStore(store)
    if config.provider == "chroma":
        return ChromaVectorStore(config, store)
    return ExternalVectorStorePlaceholder(config)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """计算两个向量的余弦相似度。"""
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _parse_chroma_connection(config: EmbeddingConfig) -> ChromaConnection:
    """从 embedding.external_url 解析 Chroma HTTP 连接参数。

    支持示例:
        http://localhost:8000
        https://chroma.example.com:443
    """
    raw_url = config.external_url or "http://localhost:8000"
    parsed = urlparse(raw_url if "://" in raw_url else f"http://{raw_url}")
    ssl = parsed.scheme == "https"
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if ssl else 8000)
    headers = {"Authorization": f"Bearer {config.external_api_key}"} if config.external_api_key else None
    return ChromaConnection(host=host, port=port, ssl=ssl, headers=headers)
