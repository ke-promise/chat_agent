from __future__ import annotations

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


def _loop(store: SQLiteStore) -> AgentLoop:
    tools = build_default_registry(store)
    context = ContextBuilder(store, MemoryRetriever(store), tools)
    reasoner = Reasoner(FakeProvider(), tools)
    return AgentLoop(store, context, reasoner, TraceRecorder(store), PresenceTracker(store))


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
