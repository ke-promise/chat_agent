from __future__ import annotations

from pathlib import Path

import pytest

from chat_agent.context import ContextBuilder
from chat_agent.memory.retriever import MemoryRetriever
from chat_agent.memory.store import SQLiteStore
from chat_agent.messages import InboundMessage
from chat_agent.messages import Attachment
from chat_agent.tools.builtin import build_default_registry


@pytest.mark.asyncio
async def test_context_builder_assembles_history_and_memory(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    await store.add_session_message("chat-1", "user", "你好")
    await store.add_session_message("chat-1", "assistant", "你好呀")
    await store.add_memory(
        "chat-1",
        "用户喜欢简洁回答",
        tags=["preference"],
        memory_type="preference",
        importance=0.9,
        source_kind="explicit",
        confidence=1.0,
    )
    builder = ContextBuilder(store, MemoryRetriever(store), build_default_registry(store), history_window=2, memory_top_k=3)

    bundle = await builder.build(InboundMessage(channel="telegram", chat_id="chat-1", sender="user-1", content="简洁回答"))

    assert bundle.trace["history_count"] == 2
    assert bundle.trace["memory_count"] == 1
    assert any("用户喜欢简洁回答" in item["content"] for item in bundle.messages)


@pytest.mark.asyncio
async def test_context_builder_injects_image_block(tmp_path: Path) -> None:
    image_path = tmp_path / "a.jpg"
    image_path.write_bytes(b"fake-image")
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    builder = ContextBuilder(
        store,
        MemoryRetriever(store),
        build_default_registry(store),
        vision_enabled=True,
    )

    bundle = await builder.build(
        InboundMessage(
            channel="telegram",
            chat_id="chat-1",
            sender="user-1",
            content="看图",
            attachments=[Attachment(kind="image", file_id="f1", local_path=str(image_path), mime_type="image/jpeg")],
        )
    )

    assert isinstance(bundle.messages[-1]["content"], list)
    assert bundle.messages[-1]["content"][1]["type"] == "image_url"
    assert bundle.messages[-1]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
