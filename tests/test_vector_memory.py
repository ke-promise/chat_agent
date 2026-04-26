from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from chat_agent.config import EmbeddingConfig
from chat_agent.memory.retriever import MemoryRetriever
from chat_agent.memory.store import SQLiteStore
from chat_agent.memory.vector_store import ChromaVectorStore, SQLiteJsonVectorStore, cosine_similarity


class FakeEmbeddingProvider:
    async def embed(self, text: str) -> list[float]:
        if "简洁" in text:
            return [1.0, 0.0]
        return [0.0, 1.0]


@pytest.mark.asyncio
async def test_vector_retriever_merges_embedding_hits(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    memory_id = await store.add_memory("chat-1", "用户喜欢简洁回答", tags=["preference"])
    vector_store = SQLiteJsonVectorStore(store)
    await vector_store.upsert_memory("chat-1", memory_id, [1.0, 0.0])

    retriever = MemoryRetriever(
        store,
        fast_provider=None,
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=vector_store,
        vector_top_k=5,
        vector_min_score=0.1,
    )

    hits = await retriever.retrieve("chat-1", "回答要简洁", top_k=5)

    assert hits[0]["id"] == memory_id
    assert hits[0]["_vector_score"] > 0.9


def test_cosine_similarity() -> None:
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_chroma_vector_store_upsert_and_search(tmp_path: Path, monkeypatch) -> None:
    class FakeCollection:
        def __init__(self) -> None:
            self.items = {}

        def upsert(self, ids, embeddings, metadatas):
            for item_id, embedding, metadata in zip(ids, embeddings, metadatas, strict=True):
                self.items[item_id] = {"embedding": embedding, "metadata": metadata}

        def query(self, query_embeddings, n_results, where, include):
            query = query_embeddings[0]
            rows = []
            for item in self.items.values():
                if item["metadata"].get("chat_id") != where.get("chat_id"):
                    continue
                score = cosine_similarity(query, item["embedding"])
                rows.append((1.0 - score, item["metadata"]))
            rows.sort(key=lambda row: row[0])
            rows = rows[:n_results]
            return {"distances": [[distance for distance, _ in rows]], "metadatas": [[metadata for _, metadata in rows]]}

    class FakeClient:
        collection = FakeCollection()

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def get_or_create_collection(self, name, metadata):
            return self.collection

    monkeypatch.setitem(sys.modules, "chromadb", SimpleNamespace(HttpClient=FakeClient))

    store = SQLiteStore(tmp_path / "agent.sqlite3")
    memory_id = await store.add_memory("chat-1", "用户喜欢简洁回答", tags=["preference"])
    config = EmbeddingConfig(
        enabled=True,
        provider="chroma",
        model="text-embedding-v4",
        api_key="embedding-key",
        base_url="https://example.test/v1",
        timeout_seconds=30,
        dimension=2,
        top_k=5,
        min_score=0.1,
        external_url="http://localhost:8000",
        external_api_key="",
        collection="chat_agent_memories",
    )
    vector_store = ChromaVectorStore(config, store)

    await vector_store.upsert_memory("chat-1", memory_id, [1.0, 0.0])
    hits = await vector_store.search("chat-1", [1.0, 0.0], top_k=5, min_score=0.1)

    assert hits[0]["id"] == memory_id
    assert hits[0]["_vector_score"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_chroma_vector_store_gracefully_degrades_when_server_is_unavailable(tmp_path: Path, monkeypatch, caplog) -> None:
    class BrokenClient:
        def __init__(self, **kwargs):
            raise ValueError("Could not connect to a Chroma server. Are you sure it is running?")

    monkeypatch.setitem(sys.modules, "chromadb", SimpleNamespace(HttpClient=BrokenClient))

    store = SQLiteStore(tmp_path / "agent.sqlite3")
    memory_id = await store.add_memory("chat-1", "用户喜欢简洁回答", tags=["preference"])
    config = EmbeddingConfig(
        enabled=True,
        provider="chroma",
        model="text-embedding-v4",
        api_key="embedding-key",
        base_url="https://example.test/v1",
        timeout_seconds=30,
        dimension=2,
        top_k=5,
        min_score=0.1,
        external_url="http://localhost:8000",
        external_api_key="",
        collection="chat_agent_memories",
    )
    vector_store = ChromaVectorStore(config, store)

    with caplog.at_level(logging.WARNING):
        await vector_store.upsert_memory("chat-1", memory_id, [1.0, 0.0])
        hits = await vector_store.search("chat-1", [1.0, 0.0], top_k=5, min_score=0.1)

    assert hits == []
    warnings = [record.message for record in caplog.records if "Chroma server is unavailable" in record.message]
    assert len(warnings) == 1
