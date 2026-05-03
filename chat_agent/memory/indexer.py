"""长期记忆索引辅助组件。"""

from __future__ import annotations

import logging

from chat_agent.memory.embeddings import EmbeddingProvider
from chat_agent.memory.vector_store import VectorStore

logger = logging.getLogger(__name__)


class MemoryIndexer:
    """把长期记忆同步到可选向量索引。

    SQLiteStore 会在写入 memories 时同步 BM25/FTS 索引；这个组件只负责可选 embedding。
    """

    def __init__(self, embedding_provider: EmbeddingProvider | None = None, vector_store: VectorStore | None = None) -> None:
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    async def index_memory(self, chat_id: str, memory_id: int, content: str) -> None:
        """为一条长期记忆生成 embedding 并写入向量后端。"""
        if not self.embedding_provider or not self.vector_store:
            return
        try:
            embedding = await self.embedding_provider.embed(content)
            if embedding:
                await self.vector_store.upsert_memory(chat_id, memory_id, embedding)
        except NotImplementedError as exc:
            logger.warning("Memory vector index unavailable: %s", exc)
        except Exception:
            logger.exception("Failed to index memory id=%s chat_id=%s", memory_id, chat_id)
