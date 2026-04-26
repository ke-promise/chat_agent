from __future__ import annotations

import json
from pathlib import Path

import pytest

from chat_agent.agent.provider import LLMResult
from chat_agent.memory.store import SQLiteStore
from chat_agent.proactive.drift import DriftManager
from chat_agent.tools.registry import Tool, ToolContext, ToolRegistry


class FakeProvider:
    async def chat(self, messages, tools=None):
        user_prompt = str(messages[-1]["content"])
        title = "Review context" if "Review context" in user_prompt else "X"
        return LLMResult(
            f'<candidate>{{"shareable": false, "title": "{title}", "body": "observation from drift", "priority": 0.4, "confidence": 0.6, "novelty": 0.3, "user_fit": 0.7}}</candidate>\n'
            "<artifact>observation from drift</artifact>",
            [],
        )


@pytest.mark.asyncio
async def test_drift_manager_runs_task_and_writes_output(tmp_path: Path) -> None:
    tasks = tmp_path / "drift_tasks.json"
    tasks.write_text(
        json.dumps({"tasks": [{"id": "review", "title": "Review context", "prompt": "Write one observation.", "enabled": True}]}),
        encoding="utf-8",
    )
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    await store.add_memory("chat-1", "User is debugging a Telegram bot", tags=["project"])
    manager = DriftManager(
        store=store,
        provider=FakeProvider(),
        tasks_path=tasks,
        output_dir=tmp_path / "drift_runs",
        run_cooldown_minutes=0,
        daily_run_cap=1,
        target_chat_id="chat-1",
    )

    result = await manager.run_once()
    last = await store.get_last_drift_run()

    assert result.ran is True
    assert result.candidate is None
    assert last is not None
    assert Path(last["output_path"]).exists()


@pytest.mark.asyncio
async def test_drift_manager_daily_max(tmp_path: Path) -> None:
    tasks = tmp_path / "drift_tasks.json"
    tasks.write_text('{"tasks":[{"id":"x","title":"X","prompt":"do","enabled":true}]}', encoding="utf-8")
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    manager = DriftManager(
        store=store,
        provider=FakeProvider(),
        tasks_path=tasks,
        output_dir=tmp_path / "drift_runs",
        run_cooldown_minutes=0,
        daily_run_cap=1,
        target_chat_id="chat-1",
    )

    first = await manager.run_once()
    second = await manager.run_once()

    assert first.ran is True
    assert second.ran is False
    assert second.reason == "daily_run_cap"


@pytest.mark.asyncio
async def test_drift_manager_rotates_tasks_instead_of_always_picking_first(tmp_path: Path) -> None:
    tasks = tmp_path / "drift_tasks.json"
    tasks.write_text(
        json.dumps(
            {
                "tasks": [
                    {"id": "first", "title": "First", "prompt": "one", "enabled": True},
                    {"id": "second", "title": "Second", "prompt": "two", "enabled": True},
                ]
            }
        ),
        encoding="utf-8",
    )
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    manager = DriftManager(
        store=store,
        provider=FakeProvider(),
        tasks_path=tasks,
        output_dir=tmp_path / "drift_runs",
        run_cooldown_minutes=999,
        daily_run_cap=2,
        target_chat_id="chat-1",
    )

    first = await manager.run_once()
    second = await manager.run_once()
    last = await store.get_last_drift_run()

    assert first.ran is True
    assert second.ran is True
    assert first.task_id != second.task_id
    assert last is not None
    assert last["task_id"] == second.task_id


class ToolCallingProvider:
    def __init__(self) -> None:
        self.calls = 0
        self.config = type("Config", (), {"enable_vision": False})()

    async def chat(self, messages, tools=None):
        self.calls += 1
        if self.calls == 1:
            return LLMResult(
                "",
                [{"id": "call-1", "name": "fake_search", "arguments": {"query": "weather"}, "raw_arguments": '{"query":"weather"}'}],
            )
        return LLMResult(
            '<candidate>{"shareable": true, "title": "Drift candidate", "body": "tool result used in drift", "priority": 0.6, "confidence": 0.8, "novelty": 0.4, "user_fit": 0.7}</candidate>\n'
            "<artifact>tool result used in drift</artifact>",
            [],
        )


@pytest.mark.asyncio
async def test_drift_manager_can_use_tool_loop(tmp_path: Path) -> None:
    tasks = tmp_path / "drift_tasks.json"
    tasks.write_text('{"tasks":[{"id":"x","title":"X","prompt":"search first","enabled":true}]}', encoding="utf-8")
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    registry = ToolRegistry(store)

    async def fake_search(_: ToolContext, args: dict) -> str:
        return "search result: " + args["query"]

    registry.register(
        Tool("fake_search", "fake search", {"type": "object", "properties": {"query": {"type": "string"}}}, fake_search, exposure="always")
    )
    manager = DriftManager(
        store=store,
        provider=ToolCallingProvider(),
        tasks_path=tasks,
        output_dir=tmp_path / "drift_runs",
        run_cooldown_minutes=0,
        daily_run_cap=1,
        target_chat_id="chat-1",
        tools=registry,
    )

    result = await manager.run_once()
    last = await store.get_last_drift_run()

    assert result.ran is True
    assert result.candidate is not None
    assert last is not None
    assert "tool result used in drift" in last["result"]


class EnumMetaProvider:
    async def chat(self, messages, tools=None):
        return LLMResult(
            '<candidate>{"shareable": "yes", "title": "Drift candidate", "body": "枚举型元数据也能发", "priority": "medium", "confidence": "high", "novelty": "0.4", "user_fit": "very_high"}</candidate>\n'
            "<artifact>枚举型元数据也能发</artifact>",
            [],
        )


@pytest.mark.asyncio
async def test_drift_manager_coerces_enum_like_candidate_meta(tmp_path: Path) -> None:
    tasks = tmp_path / "drift_tasks.json"
    tasks.write_text('{"tasks":[{"id":"x","title":"X","prompt":"share if useful","enabled":true}]}', encoding="utf-8")
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    manager = DriftManager(
        store=store,
        provider=EnumMetaProvider(),
        tasks_path=tasks,
        output_dir=tmp_path / "drift_runs",
        run_cooldown_minutes=0,
        daily_run_cap=1,
        target_chat_id="chat-1",
    )

    result = await manager.run_once()

    assert result.ran is True
    assert result.candidate is not None
    assert result.candidate.priority == pytest.approx(0.55)
    assert result.candidate.confidence == pytest.approx(0.82)
    assert result.candidate.novelty == pytest.approx(0.4)
    assert result.candidate.user_fit == pytest.approx(0.93)
