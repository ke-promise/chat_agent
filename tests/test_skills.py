from __future__ import annotations

from pathlib import Path

import pytest

from chat_agent.agent.provider import LLMResult
from chat_agent.context import ContextBuilder
from chat_agent.memory.retriever import MemoryRetriever
from chat_agent.memory.store import SQLiteStore
from chat_agent.messages import InboundMessage
from chat_agent.proactive.drift import DriftManager
from chat_agent.skills import SkillsLoader
from chat_agent.tools.builtin import build_default_registry


def _write_skill(root: Path, name: str, description: str, body: str, metadata: str | None = None) -> Path:
    path = root / name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_text = metadata or '{"chat_agent":{"always":false,"drift":false,"triggers":[],"requires":{"bins":[],"env":[],"tools":[]}}}'
    path.write_text(
        "\n".join(
            [
                "---",
                f"name: {name}",
                f"description: {description}",
                f"metadata: {metadata_text}",
                "---",
                "",
                body,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_skills_loader_scans_and_workspace_overrides(tmp_path: Path) -> None:
    builtin = tmp_path / "builtin"
    workspace = tmp_path / "workspace"
    _write_skill(builtin, "weather", "builtin weather", "builtin body")
    _write_skill(workspace, "weather", "workspace weather", "workspace body")

    loader = SkillsLoader(workspace=workspace, builtin_skills_dir=builtin)
    skills = loader.list_skills()

    assert len(skills) == 1
    assert skills[0]["source"] == "workspace"
    assert "workspace body" in (loader.load_skill("weather") or "")


def test_skills_loader_marks_missing_requirements(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "workspace"
    _write_skill(
        workspace,
        "needs-stuff",
        "needs deps",
        "body",
        '{"chat_agent":{"always":false,"requires":{"bins":["definitely-missing-bin-xyz"],"env":["MISSING_SKILL_ENV"]}}}',
    )
    monkeypatch.delenv("MISSING_SKILL_ENV", raising=False)

    loader = SkillsLoader(workspace=workspace)
    all_skills = loader.list_skills(filter_unavailable=False)

    assert all_skills[0]["available"] is False
    assert "definitely-missing-bin-xyz" in all_skills[0]["missing_bins"]
    assert "MISSING_SKILL_ENV" in all_skills[0]["missing_env"]
    assert loader.list_skills(filter_unavailable=True) == []


def test_skills_loader_marks_missing_tool_requirements(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    _write_skill(
        workspace,
        "needs-tool",
        "needs tool",
        "body",
        '{"chat_agent":{"always":false,"triggers":[],"requires":{"bins":[],"env":[],"tools":["missing_tool"]}}}',
    )

    loader = SkillsLoader(workspace=workspace)
    all_skills = loader.list_skills(filter_unavailable=False, available_tools={"present_tool"})

    assert all_skills[0]["available"] is False
    assert all_skills[0]["missing_tools"] == ["missing_tool"]
    assert loader.list_skills(filter_unavailable=True, available_tools={"present_tool"}) == []
    assert loader.list_skills(filter_unavailable=True) != []


def test_skills_loader_extracts_trigger_reasons(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    _write_skill(
        workspace,
        "weather",
        "查询天气",
        "body",
        '{"chat_agent":{"always":false,"triggers":["下雨","温度"],"requires":{"bins":[],"env":[],"tools":[]}}}',
    )
    loader = SkillsLoader(workspace=workspace)

    triggered = loader.extract_triggered_skills("明天会下雨吗")

    assert triggered == [{"name": "weather", "reason": "trigger: 下雨"}]


def test_always_skill_is_detected(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    _write_skill(
        workspace,
        "house-style",
        "style",
        "always body",
        '{"chat_agent":{"always":true,"requires":{"bins":[],"env":[]}}}',
    )
    loader = SkillsLoader(workspace=workspace)

    assert loader.get_always_skills() == ["house-style"]


@pytest.mark.asyncio
async def test_context_builder_injects_skills_catalog_and_active_skill(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    _write_skill(
        workspace,
        "weather",
        "查询天气",
        "# Weather\nUse weather steps.",
        '{"chat_agent":{"always":false,"triggers":["天气"],"requires":{"bins":[],"env":[],"tools":[]}}}',
    )
    loader = SkillsLoader(workspace=workspace)
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    registry = build_default_registry(store, skills_loader=loader)
    builder = ContextBuilder(store, MemoryRetriever(store), registry, skills_loader=loader)

    bundle = await builder.build(InboundMessage(channel="telegram", chat_id="chat-1", sender="user-1", content="@weather 北京天气"))
    system_text = "\n".join(str(item["content"]) for item in bundle.messages if item["role"] == "system")

    assert "<skills>" in system_text
    assert "<triggers>" in system_text
    assert '<skill name="weather">' in system_text
    assert bundle.trace["active_skills"] == [{"name": "weather", "reason": "mention: weather"}]
    assert bundle.trace["active_skill_names"] == ["weather"]


@pytest.mark.asyncio
async def test_skill_tools_read_create_and_reject_invalid_name(tmp_path: Path) -> None:
    loader = SkillsLoader(workspace=tmp_path / "skills")
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    registry = build_default_registry(store, skills_loader=loader)
    message = InboundMessage(channel="telegram", chat_id="chat-1", sender="user-1", content="hi")

    created = await registry.execute(
        "create_skill",
        {"name": "my-skill", "description": "desc", "body": "# My Skill\nDo it."},
        message,
    )
    read = await registry.execute("read_skill", {"name": "my-skill"}, message)
    rejected = await registry.execute("create_skill", {"name": "../bad", "description": "x", "body": "x"}, message)

    assert "已创建 workspace skill" in created
    assert "# My Skill" in read
    assert '"drift": false' in read
    assert '"triggers": []' in read
    assert '"tools": []' in read
    assert "只允许小写字母" in rejected


class FakeProvider:
    async def chat(self, messages, tools=None):
        prompt = str(messages[-1]["content"])
        assert "Drift Skill Body" in prompt
        return LLMResult(
            '<candidate>{"shareable": false, "title": "review skill", "body": "drift skill result", "priority": 0.4, "confidence": 0.6, "novelty": 0.3, "user_fit": 0.7}</candidate>\n'
            "<artifact>drift skill result</artifact>",
            [],
        )


@pytest.mark.asyncio
async def test_drift_uses_drift_skill_before_json_tasks(tmp_path: Path) -> None:
    drift_skills = tmp_path / "drift_skills"
    _write_skill(
        drift_skills,
        "review",
        "review skill",
        "Drift Skill Body",
        '{"chat_agent":{"always":false,"drift":true,"triggers":[],"requires":{"bins":[],"env":[],"tools":[]}}}',
    )
    tasks = tmp_path / "drift_tasks.json"
    tasks.write_text('{"tasks":[{"id":"fallback","title":"Fallback","prompt":"fallback body","enabled":true}]}', encoding="utf-8")
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    manager = DriftManager(
        store=store,
        provider=FakeProvider(),
        tasks_path=tasks,
        output_dir=tmp_path / "runs",
        run_cooldown_minutes=0,
        daily_run_cap=1,
        skills_loader=SkillsLoader(workspace=drift_skills),
    )

    result = await manager.run_once()
    last = await store.get_last_drift_run()

    assert result.ran is True
    assert last is not None
    assert last["task_id"] == "skill-review"
