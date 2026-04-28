from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from chat_agent.agent.provider import LLMResult
from chat_agent.context import ContextBuilder
from chat_agent.loop import AgentLoop
from chat_agent.memory.retriever import MemoryRetriever
from chat_agent.memory.store import SQLiteStore
from chat_agent.messages import InboundMessage
from chat_agent.observe.trace import TraceRecorder
from chat_agent.presence import PresenceTracker
from chat_agent.reasoner import Reasoner
from chat_agent.tools.builtin import build_default_registry


class FakeProvider:
    async def chat(self, messages, tools=None):
        return LLMResult(content="LLM 回复", tool_calls=[])


def _message(text: str) -> InboundMessage:
    return InboundMessage(channel="telegram", chat_id="chat-1", sender="user-1", content=text, metadata={"username": "alice"})


def _loop(store: SQLiteStore, model_main: str = "", model_fast: str = "") -> AgentLoop:
    tools = build_default_registry(store)
    context = ContextBuilder(store, MemoryRetriever(store), tools)
    reasoner = Reasoner(FakeProvider(), tools)
    return AgentLoop(
        store,
        context,
        reasoner,
        TraceRecorder(store),
        PresenceTracker(store),
        model_main=model_main,
        model_fast=model_fast,
    )


@pytest.mark.asyncio
async def test_agent_loop_direct_memory_and_reminder(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    loop = _loop(store)

    remembered = await loop.handle_message(_message("记住：我喜欢简洁的回答"))
    recalled = await loop.handle_message(_message("你记得我喜欢什么？"))
    reminder = await loop.handle_message(_message("1分钟后提醒我喝水"))

    assert "记住" in remembered.content
    assert "简洁的回答" in recalled.content
    assert "提醒你 喝水" in reminder.content
    assert await store.get_due_reminders() == []


@pytest.mark.asyncio
async def test_agent_loop_records_model_names_in_trace(tmp_path: Path) -> None:
    db_path = tmp_path / "agent.sqlite3"
    store = SQLiteStore(db_path)
    loop = _loop(store, model_main="main-model", model_fast="fast-model")

    await loop.handle_message(_message("hello"))

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT model_main, model_fast FROM message_trace ORDER BY id DESC LIMIT 1"
        ).fetchone()

    assert row == ("main-model", "fast-model")
