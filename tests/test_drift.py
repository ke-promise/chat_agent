from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from chat_agent.agent.provider import LLMResult
from chat_agent.memory.store import SQLiteStore
from chat_agent.proactive.drift import DriftManager, _DriftToolRegistry, _drift_dedupe_key, _stale_search_query_reason
from chat_agent.skills import SkillsLoader
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


class DegradedSearchProvider:
    def __init__(self) -> None:
        self.calls = 0
        self.config = type("Config", (), {"enable_vision": False})()

    async def chat(self, messages, tools=None):
        self.calls += 1
        if self.calls == 1:
            return LLMResult(
                "",
                [
                    {
                        "id": "call-1",
                        "name": "duckduckgo_web_search",
                        "arguments": {"query": "鸣潮 3.3 2026 最新"},
                        "raw_arguments": '{"query":"鸣潮 3.3 2026 最新"}',
                    }
                ],
            )
        return LLMResult(
            '<candidate>{"shareable": true, "title": "鸣潮3.3明天更新", "body": "刚看到鸣潮明天要更新3.3二周年版本，新角色也会上线。", "priority": 0.9, "confidence": 0.9, "novelty": 0.9, "user_fit": 0.9}</candidate>\n'
            "<artifact>搜索失败后没有可靠来源。</artifact>",
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


@pytest.mark.asyncio
async def test_drift_manager_does_not_withhold_candidate_after_degraded_search(tmp_path: Path) -> None:
    tasks = tmp_path / "drift_tasks.json"
    tasks.write_text('{"tasks":[{"id":"x","title":"X","prompt":"search fresh game news","enabled":true}]}', encoding="utf-8")
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    registry = ToolRegistry(store)

    async def degraded_search(_: ToolContext, args: dict) -> str:
        return json.dumps(
            {"query": args["query"], "results": [], "provider": "duckduckgo", "degraded": True, "error": "HTTP 202"},
            ensure_ascii=False,
        )

    registry.register(
        Tool(
            "duckduckgo_web_search",
            "web search",
            {"type": "object", "properties": {"query": {"type": "string"}}},
            degraded_search,
            exposure="always",
        )
    )
    manager = DriftManager(
        store=store,
        provider=DegradedSearchProvider(),
        tasks_path=tasks,
        output_dir=tmp_path / "drift_runs",
        run_cooldown_minutes=0,
        daily_run_cap=1,
        target_chat_id="chat-1",
        tools=registry,
    )

    result = await manager.run_once()

    assert result.ran is True
    assert result.candidate is not None


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


class StaleShareProvider:
    async def chat(self, messages, tools=None):
        return LLMResult(
            '<candidate>{"shareable": true, "title": "旧版本安排", "body": "2025 年 2.5 版本 7月24号已经开了，10月9号还有旧活动。", "priority": 0.9, "confidence": 0.9, "novelty": 0.9, "user_fit": 0.9}</candidate>\n'
            "<artifact>旧版本安排归档</artifact>",
            [],
        )


@pytest.mark.asyncio
async def test_drift_manager_keeps_shareable_candidate_without_post_parse_stale_filter(tmp_path: Path) -> None:
    tasks = tmp_path / "drift_tasks.json"
    tasks.write_text('{"tasks":[{"id":"x","title":"X","prompt":"share old version info","enabled":true}]}', encoding="utf-8")
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    manager = DriftManager(
        store=store,
        provider=StaleShareProvider(),
        tasks_path=tasks,
        output_dir=tmp_path / "drift_runs",
        run_cooldown_minutes=0,
        daily_run_cap=1,
        target_chat_id="chat-1",
    )

    result = await manager.run_once()

    assert result.ran is True
    assert result.candidate is not None


def test_drift_manager_merges_skill_and_json_tasks(tmp_path: Path) -> None:
    skills = tmp_path / "skills"
    skill_path = skills / "review" / "SKILL.md"
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text(
        "\n".join(
            [
                "---",
                "name: review",
                "description: review skill",
                'metadata: {"chat_agent":{"always":false,"drift":true,"triggers":[],"requires":{"bins":[],"env":[],"tools":[]}}}',
                "---",
                "",
                "Skill body",
            ]
        ),
        encoding="utf-8",
    )
    tasks = tmp_path / "drift_tasks.json"
    tasks.write_text('{"tasks":[{"id":"json-task","title":"JSON task","prompt":"JSON body","enabled":true}]}', encoding="utf-8")
    manager = DriftManager(
        store=SQLiteStore(tmp_path / "agent.sqlite3"),
        provider=FakeProvider(),
        tasks_path=tasks,
        output_dir=tmp_path / "drift_runs",
        skills_loader=SkillsLoader(workspace=skills),
    )

    loaded = manager.load_tasks()

    assert [task.id for task in loaded] == ["skill-review", "json-task"]


def test_drift_search_query_rejects_old_year() -> None:
    reason = _stale_search_query_reason("duckduckgo_web_search", {"query": "鸣潮 2.6 新角色 2025"}, now=datetime(2026, 4, 29))

    assert "旧年份 2025" in reason
    assert "2026" in reason


def test_drift_tool_registry_delegates_search(tmp_path: Path) -> None:
    registry = ToolRegistry(SQLiteStore(tmp_path / "agent.sqlite3"))

    async def noop(_: ToolContext, args: dict) -> str:
        return ""

    registry.register(Tool("discover_tool", "discover target", {"type": "object", "properties": {}}, noop, exposure="discoverable"))
    wrapped = _DriftToolRegistry(registry)

    assert [tool.name for tool in wrapped.search("discover")] == ["discover_tool"]


def test_drift_dedupe_key_tracks_candidate_identity() -> None:
    first = _drift_dedupe_key(
        "interest_search",
        "鸣潮 3.3 二周年更新",
        "诶话说你玩鸣潮对吧？我刚看到明天（4/30）3.3版本「自星海尽处回响」二周年大更新要上了，新地图黯原、新角色绯雪和达妮娅双UP池，而且好像送超过50抽的福利还有兑换码～",
    )
    second = _drift_dedupe_key(
        "followup_draft",
        "鸣潮3.3二周年版本跟进",
        "嘿，刚看到鸣潮明天（4/30）上3.3版本啦，叫「自星海尽处回响」，二周年庆送五十多抽，还出了两个新五星——绯雪和达妮娅～",
    )

    assert first != second
