from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from chat_agent.config import EmbeddingConfig
from chat_agent.memory.reranker import RerankResult
from chat_agent.memory.retriever import MemoryRetriever
from chat_agent.memory.store import SQLiteStore
from chat_agent.memory.vector_store import ChromaVectorStore, SQLiteJsonVectorStore, cosine_similarity


class FakeEmbeddingProvider:
    async def embed(self, text: str) -> list[float]:
        if "简洁" in text:
            return [1.0, 0.0]
        return [0.0, 1.0]


class FakeReranker:
    enabled = True

    async def rerank(self, query: str, documents: list[str], top_n: int | None = None) -> list[RerankResult]:
        beta_index = next(index for index, document in enumerate(documents) if "beta" in document)
        other_index = next(index for index, document in enumerate(documents) if "beta" not in document)
        return [RerankResult(index=beta_index, score=0.99), RerankResult(index=other_index, score=0.2)]


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


@pytest.mark.asyncio
async def test_bm25_retriever_hits_and_ignores_deleted_or_superseded(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    active_id = await store.add_memory("chat-1", "用户喜欢简洁回答", tags=["preference"])
    deleted_id = await store.add_memory("chat-1", "用户喜欢简洁回答的旧说法", tags=["preference"])
    replacement_id = await store.add_memory("chat-1", "用户喜欢详细回答", tags=["preference"])

    assert await store.delete_memory("chat-1", deleted_id)
    assert await store.supersede_memory("chat-1", active_id, replacement_id, "test")

    hits = await store.search_bm25_memories("chat-1", "简洁回答", limit=10)
    assert all(item["id"] not in {active_id, deleted_id} for item in hits)

    detail_hits = await store.search_bm25_memories("chat-1", "详细回答", limit=10)
    assert detail_hits[0]["id"] == replacement_id


@pytest.mark.asyncio
async def test_rrf_boosts_memory_seen_by_bm25_and_vector(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    both_id = await store.add_memory("chat-1", "用户喜欢简洁回答", tags=["preference"])
    vector_only_id = await store.add_memory("chat-1", "用户喜欢安静的早晨", tags=["preference"])
    vector_store = SQLiteJsonVectorStore(store)
    await vector_store.upsert_memory("chat-1", both_id, [1.0, 0.0])
    await vector_store.upsert_memory("chat-1", vector_only_id, [1.0, 0.0])

    retriever = MemoryRetriever(
        store,
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=vector_store,
        vector_top_k=5,
        bm25_top_k=5,
        rrf_top_k=5,
        vector_min_score=0.1,
    )

    hits = await retriever.retrieve("chat-1", "简洁回答", top_k=5)

    assert hits[0]["id"] == both_id
    assert hits[0]["_match_reason"] == "rrf_bm25+vector"
    assert hits[0]["_rrf_score"] > next(item["_rrf_score"] for item in hits if item["id"] == vector_only_id)


@pytest.mark.asyncio
async def test_reranker_reorders_rrf_candidates(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    first_id = await store.add_memory("chat-1", "alpha memory about reply style", tags=["preference"])
    second_id = await store.add_memory("chat-1", "beta memory about reply style", tags=["preference"])
    retriever = MemoryRetriever(store, bm25_top_k=5, rrf_top_k=5, reranker=FakeReranker())

    hits = await retriever.retrieve("chat-1", "memory reply style", top_k=2)

    assert {first_id, second_id} == {item["id"] for item in hits}
    assert hits[0]["id"] == second_id
    assert hits[0]["_match_reason"] == "bge_rerank"
    assert retriever.last_trace["rerank_used"] is True


@pytest.mark.asyncio
async def test_reranker_failure_falls_back_to_rrf(tmp_path: Path) -> None:
    class BrokenReranker:
        enabled = True

        async def rerank(self, query: str, documents: list[str], top_n: int | None = None) -> list[RerankResult]:
            raise RuntimeError("rerank down")

    store = SQLiteStore(tmp_path / "agent.sqlite3")
    await store.add_memory("chat-1", "alpha memory about reply style", tags=["preference"])
    retriever = MemoryRetriever(store, bm25_top_k=5, rrf_top_k=5, reranker=BrokenReranker())

    hits = await retriever.retrieve("chat-1", "memory reply style", top_k=1)

    assert hits
    assert hits[0]["_match_reason"] == "rrf_bm25"
    assert "rerank down" in retriever.last_trace["rerank_error"]


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
