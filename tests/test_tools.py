from __future__ import annotations

from pathlib import Path

import pytest

from chat_agent.memory.store import SQLiteStore
from chat_agent.messages import InboundMessage
from chat_agent.tools.builtin import build_default_registry, register_message_push_tool


def _message() -> InboundMessage:
    return InboundMessage(channel="telegram", chat_id="chat-1", sender="user-1", content="hello")


class FakeChannel:
    def __init__(self) -> None:
        self.sent = []

    async def send(self, message) -> None:
        self.sent.append(message)


class FakeMemoryIndexer:
    def __init__(self) -> None:
        self.indexed = []

    async def index_memory(self, chat_id: str, memory_id: int, content: str) -> None:
        self.indexed.append((chat_id, memory_id, content))


class FakeMemoryRetriever:
    async def retrieve(self, chat_id: str, query: str, top_k: int):
        return [{"id": 7, "content": f"retrieved via pipeline: {query}"}]


@pytest.mark.asyncio
async def test_tool_registry_register_and_execute(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    registry = build_default_registry(store)

    saved = await registry.execute("memorize", {"content": "likes quiet mornings", "tags": ["preference"]}, _message())
    recalled = await registry.execute("recall_memory", {"query": "quiet"}, _message())

    assert "memorize" in registry.list_descriptions()
    assert "quiet mornings" in recalled
    assert saved


@pytest.mark.asyncio
async def test_memory_tools_use_indexer_and_retriever(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    indexer = FakeMemoryIndexer()
    registry = build_default_registry(store, memory_indexer=indexer, memory_retriever=FakeMemoryRetriever())

    saved = await registry.execute("memorize", {"content": "likes quiet mornings", "tags": ["preference"]}, _message())
    recalled = await registry.execute("recall_memory", {"query": "quiet", "limit": 3}, _message())

    assert saved
    assert indexer.indexed[0][2] == "likes quiet mornings"
    assert "retrieved via pipeline: quiet" in recalled


@pytest.mark.asyncio
async def test_reminder_tools(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    registry = build_default_registry(store)

    created = await registry.execute("create_reminder", {"content": "drink water", "delay_seconds": 60}, _message())
    listed = await registry.execute("list_reminders", {}, _message())

    assert created
    assert "drink water" in listed


def test_tool_registry_visibility_and_search(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    registry = build_default_registry(store)

    visible = set(registry.visible_names())
    assert "tool_search" in visible
    assert "web_fetch" not in visible
    assert "write_file" not in visible

    schema_names = {schema["function"]["name"] for schema in registry.get_schema()}
    assert "web_fetch" not in schema_names
    assert "write_file" not in schema_names

    matches = registry.search("web", exposures={"discoverable"})
    assert any(tool.name == "web_fetch" for tool in matches)
    assert all(tool.exposure == "discoverable" for tool in matches)
    assert all(tool.name != "write_file" for tool in matches)


@pytest.mark.asyncio
async def test_file_tools_are_limited_to_workspace(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    registry = build_default_registry(store, file_workspace=tmp_path / "files")

    written = await registry.execute("write_file", {"path": "notes/today.txt", "content": "hello"}, _message())
    listed = await registry.execute("list_files", {"path": "notes"}, _message())
    content = await registry.execute("read_file", {"path": "notes/today.txt"}, _message())

    assert written
    assert "notes/today.txt" in listed
    assert content == "hello"


@pytest.mark.asyncio
async def test_send_message_tool_is_restricted(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    registry = build_default_registry(store)
    channel = FakeChannel()
    register_message_push_tool(registry, channel, default_chat_id="target")

    ok = await registry.execute("send_message", {"content": "hello"}, _message())
    denied = await registry.execute("send_message", {"chat_id": "other", "content": "x"}, _message())

    assert ok
    assert channel.sent[0].content == "hello"
    assert channel.sent[0].chat_id == "target"
    assert denied
    assert len(channel.sent) == 1


@pytest.mark.asyncio
async def test_send_emoji_hidden_but_send_meme_discoverable(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    registry = build_default_registry(store)
    channel = FakeChannel()

    memes_dir = tmp_path / "files" / "memes" / "happy"
    memes_dir.mkdir(parents=True)
    (memes_dir / "001.png").write_bytes(b"fake-png")
    (tmp_path / "files" / "memes" / "empty").mkdir(parents=True)
    (tmp_path / "files" / "memes" / "manifest.json").write_text(
        (
            '{"version":1,"categories":{"happy":{"desc":"happy","aliases":["joy"],"mood_tags":["celebrate"],'
            '"enabled":true,"files":["001.png"]},"empty":{"desc":"empty","enabled":true,"files":["404.png"]}}}'
        ),
        encoding="utf-8",
    )

    register_message_push_tool(registry, channel, default_chat_id="target", file_workspace=tmp_path / "files")

    visible = set(registry.visible_names())
    assert "send_emoji" not in visible
    assert "send_meme" not in visible
    assert "list_memes" in visible
    assert any(tool.name == "send_meme" for tool in registry.search("meme", exposures={"discoverable"}))

    listed = await registry.execute("list_memes", {}, _message())
    emoji_result = await registry.execute("send_emoji", {"emoji": "OK", "text": "hello"}, _message())
    meme_result = await registry.execute("send_meme", {"category": "happy", "caption": "saved"}, _message())

    assert "happy" in listed
    assert "empty" not in listed
    assert emoji_result
    assert channel.sent[0].content == "OK hello"
    assert meme_result
    assert channel.sent[1].attachments
    assert channel.sent[1].attachments[0].local_path.endswith("001.png")


def test_extra_model_tools_can_surface_hidden_tools(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    registry = build_default_registry(store, extra_model_tools=["write_file"])
    register_message_push_tool(registry, FakeChannel(), default_chat_id="target")

    visible = set(registry.visible_names())

    assert "write_file" in visible
    assert "send_message" not in visible
