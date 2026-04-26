from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from chat_agent.agent.provider import LLMResult
from chat_agent.context import ContextBuilder
from chat_agent.loop import AgentLoop
from chat_agent.memory.consolidation import ConsolidationService
from chat_agent.memory.files import MemoryFiles
from chat_agent.memory.optimizer import MemoryOptimizer
from chat_agent.memory.retriever import MemoryRetriever
from chat_agent.memory.store import SQLiteStore
from chat_agent.messages import InboundMessage
from chat_agent.observe.trace import TraceRecorder
from chat_agent.presence import PresenceTracker
from chat_agent.reasoner import Reasoner
from chat_agent.tools.builtin import build_default_registry


class FakeProvider:
    async def chat(self, messages, tools=None):
        return LLMResult(content="assistant reply", tool_calls=[])


class FakeConsolidationProvider:
    async def chat(self, messages, tools=None):
        return LLMResult(
            content=(
                '{"memories":['
                '{"type":"preference","content":"User prefers concise replies.","tags":["preference"],"importance":0.82,"confidence":0.8},'
                '{"type":"fact","content":"User may care about weather alerts.","tags":["candidate"],"importance":0.55,"confidence":0.5}'
                '],'
                '"recent_context":"Recent context: concise replies matter."}'
            ),
            tool_calls=[],
        )


def _inbound(text: str) -> InboundMessage:
    return InboundMessage(channel="telegram", chat_id="chat-1", sender="user-1", content=text)


def _agent_loop(store: SQLiteStore) -> AgentLoop:
    tools = build_default_registry(store)
    context = ContextBuilder(store, MemoryRetriever(store), tools)
    reasoner = Reasoner(FakeProvider(), tools)
    return AgentLoop(store, context, reasoner, TraceRecorder(store), PresenceTracker(store))


@pytest.mark.asyncio
async def test_commit_writes_plain_user_and_assistant_messages(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    loop = _agent_loop(store)

    await loop.handle_message(_inbound("hello"))
    history = await store.get_recent_session_messages("chat-1", limit=10)

    assert [item["role"] for item in history[-2:]] == ["user", "assistant"]
    assert history[-2]["content"] == "hello"
    assert history[-1]["content"] == "assistant reply"


@pytest.mark.asyncio
async def test_store_reinforcement_checkpoint_and_supersede(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "agent.sqlite3")

    first = await store.add_memory("chat-1", "User prefers concise replies.", tags=["preference"], source_kind="inferred", confidence=0.7)
    second = await store.add_memory("chat-1", "User prefers concise replies.", tags=["preference"], source_kind="explicit", confidence=1.0)
    memories = await store.get_memories_by_ids("chat-1", [first])

    assert second == first
    assert memories[0]["reinforcement"] == 2
    assert memories[0]["content_hash"]
    assert memories[0]["source_kind"] == "explicit"
    assert memories[0]["confidence"] == pytest.approx(1.0)

    replacement = await store.add_memory("chat-1", "User prefers detailed replies.", tags=["preference"])
    assert await store.supersede_memory("chat-1", first, replacement, "user correction")
    assert await store.get_memories_by_ids("chat-1", [first]) == []

    assert await store.get_last_consolidated("chat-1") == 0
    await store.set_last_consolidated("chat-1", 12)
    await store.set_last_consolidated("chat-1", 3)
    assert await store.get_last_consolidated("chat-1") == 12

    assert await store.add_consolidation_event("chat-1", "session:chat-1:1-12")
    assert not await store.add_consolidation_event("chat-1", "session:chat-1:1-12")


def test_memory_files_export_snapshot(tmp_path: Path) -> None:
    files = MemoryFiles(tmp_path / "memory")
    root = files.export_chat_snapshot(
        "chat-1",
        "summary text",
        [{"id": 1, "type": "preference", "content": "简洁回答", "source_kind": "explicit", "confidence": 1.0, "importance": 0.8, "reinforcement": 2}],
        [{"id": 2, "type": "fact", "content": "可能关心天气提醒", "status": "pending", "confidence": 0.5, "evidence_count": 1}],
        [{"old_memory_id": 1, "new_memory_id": 3, "reason": "correction", "created_at": "2026-01-01T00:00:00+00:00"}],
        user_profile={"preferences": ["天气提醒"]},
    )

    assert (root / "SUMMARY.md").exists()
    assert "summary text" in (root / "SUMMARY.md").read_text(encoding="utf-8")
    assert "简洁回答" in (root / "MEMORIES.md").read_text(encoding="utf-8")
    assert "天气提醒" in (root / "CANDIDATES.md").read_text(encoding="utf-8")
    assert "天气提醒" in (root / "INTERESTS.md").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_consolidation_updates_summary_candidates_and_export(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    files = MemoryFiles(tmp_path / "memory")
    for index in range(8):
        await store.add_session_message("chat-1", "user" if index % 2 == 0 else "assistant", f"message {index}")

    service = ConsolidationService(
        store,
        files,
        provider=FakeConsolidationProvider(),
        keep_recent=2,
        max_window=10,
    )
    result = await service.run_once("chat-1")

    assert result.ran
    assert result.message_count == 6
    assert result.memory_count == 1
    assert result.candidate_count == 1
    assert await store.get_last_consolidated("chat-1") == 6
    assert "Recent context" in (files.chat_root("chat-1") / "SUMMARY.md").read_text(encoding="utf-8")
    memories = await store.search_memories("chat-1", "concise", limit=10)
    assert any(item["type"] == "preference" for item in memories)
    candidates = await store.get_memory_candidates("chat-1")
    assert any("weather alerts" in item["content"] for item in candidates)

    duplicate = await service.run_once("chat-1")
    assert not duplicate.ran


@pytest.mark.asyncio
async def test_memory_optimizer_exports_chat_snapshots(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    files = MemoryFiles(tmp_path / "memory")
    await store.upsert_summary("chat-1", "summary", 2)
    await store.add_memory("chat-1", "User prefers concise replies.", tags=["preference"], source_kind="explicit", confidence=1.0)
    await store.add_memory_candidate("chat-1", "User might care about weather alerts.", tags=["candidate"])

    result = await MemoryOptimizer(store, files).run_once(["chat-1"])

    assert result.ran
    assert result.exported_chats == 1
    assert "summary" in (files.chat_root("chat-1") / "SUMMARY.md").read_text(encoding="utf-8")
    assert "concise replies" in (files.chat_root("chat-1") / "INTERESTS.md").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_context_builder_ignores_missing_export_files(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    files = MemoryFiles(tmp_path / "memory")
    files.export_chat_snapshot("chat-1", "summary", [], [], [])
    shutil.rmtree(files.chat_root("chat-1"))

    await store.upsert_summary("chat-1", "db summary", 1)
    await store.add_memory("chat-1", "User likes concise replies.", tags=["preference"], source_kind="explicit", confidence=1.0)
    builder = ContextBuilder(
        store,
        MemoryRetriever(store),
        build_default_registry(store),
        memory_files=files,
    )
    bundle = await builder.build(_inbound("concise replies"))
    system_text = "\n".join(str(item["content"]) for item in bundle.messages if item["role"] == "system")

    assert "db summary" in system_text
    assert "RECENT_CONTEXT.md" not in system_text
    assert bundle.trace["memory_file_chars"] == 0


@pytest.mark.asyncio
async def test_candidate_promotes_after_repeated_evidence(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    loop = _agent_loop(store)

    await loop.handle_message(_inbound("回答风格希望简洁"))
    await loop.handle_message(_inbound("回答风格希望简洁"))

    candidates = await store.get_memory_candidates("chat-1")
    memories = await store.search_memories("chat-1", "简洁", limit=10)

    assert candidates == []
    assert any(item["source_kind"] == "promoted" for item in memories)


@pytest.mark.asyncio
async def test_natural_correction_supersedes_old_memory(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    loop = _agent_loop(store)

    remembered = await loop.handle_message(_inbound("记住：我喜欢简洁回答"))
    corrected = await loop.handle_message(_inbound("你记错了，不是简洁回答，我喜欢详细回答"))

    memories = await store.search_memories("chat-1", "详细回答", limit=10)
    old = await store.search_memories("chat-1", "简洁回答", limit=10)

    assert "记住" in remembered.content
    assert "改成新版" in corrected.content
    assert any("详细回答" in item["content"] for item in memories)
    assert old == []
