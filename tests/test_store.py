from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from chat_agent.memory.store import SQLiteStore


@pytest.mark.asyncio
async def test_memory_write_search_delete(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    memory_id = await store.add_memory(
        "chat-1",
        "我喜欢喝乌龙茶",
        tags=["preference", "drink"],
        memory_type="preference",
        importance=0.9,
    )

    results = await store.search_memories("chat-1", "乌龙茶", limit=5)
    assert results[0]["id"] == memory_id
    assert results[0]["type"] == "preference"
    assert "drink" in results[0]["tags"]

    assert await store.delete_memory("chat-1", memory_id)
    assert await store.search_memories("chat-1", "乌龙茶", limit=5) == []


@pytest.mark.asyncio
async def test_reminder_due_complete_and_cancel(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    due_id = await store.add_reminder(
        "chat-1",
        "user-1",
        "喝水",
        datetime.now(timezone.utc) - timedelta(seconds=1),
    )
    future_id = await store.add_reminder(
        "chat-1",
        "user-1",
        "站起来",
        datetime.now(timezone.utc) + timedelta(hours=1),
    )

    due = await store.get_due_reminders()
    assert [item["id"] for item in due] == [due_id]

    assert await store.cancel_reminder("chat-1", future_id)
    await store.mark_reminder_delivered(due_id)
    assert await store.get_due_reminders() == []


@pytest.mark.asyncio
async def test_trace_and_summary(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    await store.upsert_summary("chat-1", "用户喜欢简洁回答", 42)
    await store.add_message_trace("chat-1", "hi", "hello", ["memorize"], [1], 123)

    assert await store.get_summary("chat-1") == "用户喜欢简洁回答"


@pytest.mark.asyncio
async def test_user_profile_update_merge(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "agent.sqlite3")

    await store.update_user_profile("chat-1", {"name": "alice", "preferences": ["简洁回答"]})
    profile = await store.update_user_profile("chat-1", {"preferences": ["天气提醒", "简洁回答"]})

    assert profile["name"] == "alice"
    assert profile["preferences"] == ["简洁回答", "天气提醒"]


@pytest.mark.asyncio
async def test_memory_embedding_store(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    memory_id = await store.add_memory("chat-1", "我喜欢简洁回答", tags=["preference"])
    await store.upsert_memory_embedding("chat-1", memory_id, [1.0, 0.0, 0.0])

    embeddings = await store.list_memory_embeddings("chat-1")
    memories = await store.get_memories_by_ids("chat-1", [memory_id])

    assert embeddings == [{"memory_id": memory_id, "embedding": [1.0, 0.0, 0.0]}]
    assert memories[0]["content"] == "我喜欢简洁回答"
