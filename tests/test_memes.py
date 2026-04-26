from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from chat_agent.config import DriftConfig, ProactiveBudgetConfig
from chat_agent.context import ContextBuilder
from chat_agent.loop import AgentLoop
from chat_agent.memes import MemeService
from chat_agent.memory.retriever import MemoryRetriever
from chat_agent.memory.store import SQLiteStore, utc_now
from chat_agent.messages import Attachment, InboundMessage, OutboundMessage
from chat_agent.observe.trace import TraceRecorder
from chat_agent.presence import PresenceTracker
from chat_agent.proactive.loop import ProactiveLoop
from chat_agent.proactive.models import ProactiveCandidate
from chat_agent.reasoner import Reasoner
from chat_agent.tools.builtin import build_default_registry
from chat_agent.agent.provider import LLMResult


class FakeProvider:
    async def chat(self, messages, tools=None):
        return LLMResult(content="好呀，给你一张", tool_calls=[])


class FakeChannel:
    def __init__(self) -> None:
        self.sent = []

    async def send(self, message) -> None:
        self.sent.append(message)


def _build_loop(store: SQLiteStore, file_workspace: Path) -> AgentLoop:
    tools = build_default_registry(store, file_workspace=file_workspace)
    context = ContextBuilder(store, MemoryRetriever(store), tools)
    reasoner = Reasoner(FakeProvider(), tools)
    meme_service = MemeService(file_workspace)
    return AgentLoop(
        store,
        context,
        reasoner,
        TraceRecorder(store),
        PresenceTracker(store),
        meme_service=meme_service,
    )


@pytest.mark.asyncio
async def test_ingest_image_into_meme_library(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    loop = _build_loop(store, tmp_path / "files")

    image = tmp_path / "attachments" / "source.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"fake-image")

    inbound = InboundMessage(
        channel="telegram",
        chat_id="chat-1",
        sender="user-1",
        content="存成表情包：happy",
        attachments=[Attachment(kind="image", file_id="f1", local_path=str(image), mime_type="image/png")],
    )

    outbound = await loop.handle_message(inbound)

    assert "收进表情包库" in outbound.content
    assert (tmp_path / "files" / "memes" / "happy" / "001.png").exists()
    manifest = (tmp_path / "files" / "memes" / "manifest.json").read_text(encoding="utf-8")
    assert '"happy"' in manifest


@pytest.mark.asyncio
async def test_duplicate_image_is_not_ingested_twice(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    loop = _build_loop(store, tmp_path / "files")

    image = tmp_path / "attachments" / "source.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"fake-image")

    inbound = InboundMessage(
        channel="telegram",
        chat_id="chat-1",
        sender="user-1",
        content="存成表情包：happy",
        attachments=[Attachment(kind="image", file_id="f1", local_path=str(image), mime_type="image/png")],
    )

    first = await loop.handle_message(inbound)
    second = await loop.handle_message(inbound)

    assert "收进表情包库" in first.content
    assert "已经在表情包库里" in second.content
    assert len(list((tmp_path / "files" / "memes" / "happy").glob("*.png"))) == 1


@pytest.mark.asyncio
async def test_passive_reply_can_auto_attach_meme(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    file_workspace = tmp_path / "files"
    (file_workspace / "memes" / "happy").mkdir(parents=True)
    (file_workspace / "memes" / "happy" / "001.png").write_bytes(b"fake-image")
    (file_workspace / "memes" / "manifest.json").write_text(
        '{"version":1,"categories":{"happy":{"desc":"开心表情","aliases":["开心"],"enabled":true,"files":["001.png"]}}}',
        encoding="utf-8",
    )
    loop = _build_loop(store, file_workspace)

    outbound = await loop.handle_message(
        InboundMessage(channel="telegram", chat_id="chat-1", sender="user-1", content="来个开心表情包")
    )

    assert outbound.attachments
    assert outbound.attachments[0].local_path.endswith("001.png")


@pytest.mark.asyncio
async def test_generic_explicit_meme_request_can_pick_any_category(tmp_path: Path) -> None:
    file_workspace = tmp_path / "files"
    (file_workspace / "memes" / "happy").mkdir(parents=True)
    (file_workspace / "memes" / "happy" / "001.png").write_bytes(b"fake-image")
    (file_workspace / "memes" / "manifest.json").write_text(
        '{"version":1,"categories":{"happy":{"desc":"开心表情","aliases":["开心"],"enabled":true,"files":["001.png"]}}}',
        encoding="utf-8",
    )
    service = MemeService(file_workspace)

    outbound = service.decorate_outbound(
        OutboundMessage(channel="telegram", chat_id="chat-1", content="给你来一张"),
        inbound_text="来个表情包",
        source="passive",
    )

    assert outbound.attachments
    assert outbound.metadata["meme_decision_explicit"] is True
    assert outbound.metadata["auto_meme_query"] == ""


@pytest.mark.asyncio
async def test_service_can_proactively_attach_meme_without_explicit_request(tmp_path: Path) -> None:
    file_workspace = tmp_path / "files"
    (file_workspace / "memes" / "hug").mkdir(parents=True)
    (file_workspace / "memes" / "hug" / "001.png").write_bytes(b"fake-image")
    (file_workspace / "memes" / "manifest.json").write_text(
        (
            '{"version":1,"categories":{"hug":{"desc":"安慰抱抱","aliases":["抱抱","安慰"],'
            '"mood_tags":["安慰","委屈"],"usage_scenarios":["安慰回复"],"enabled":true,"files":["001.png"]}}}'
        ),
        encoding="utf-8",
    )
    service = MemeService(file_workspace)

    outbound = service.decorate_outbound(
        OutboundMessage(channel="telegram", chat_id="chat-1", content="来，给你一个抱抱，今天辛苦啦"),
        inbound_text="我今天有点委屈",
        source="passive",
    )

    assert outbound.attachments
    assert outbound.metadata["meme_decision_driver"] == "priority_emotion"
    assert outbound.metadata["auto_meme_category"] == "hug"


@pytest.mark.asyncio
async def test_priority_user_emotion_prefers_meme_for_distress_messages(tmp_path: Path) -> None:
    file_workspace = tmp_path / "files"
    (file_workspace / "memes" / "hug").mkdir(parents=True)
    (file_workspace / "memes" / "hug" / "001.png").write_bytes(b"fake-image")
    (file_workspace / "memes" / "manifest.json").write_text(
        (
            '{"version":1,"categories":{"hug":{"desc":"安慰抱抱","aliases":["抱抱","安慰"],'
            '"mood_tags":["安慰","委屈"],"usage_scenarios":["安慰回复"],"enabled":true,"files":["001.png"]}}}'
        ),
        encoding="utf-8",
    )
    service = MemeService(file_workspace)

    outbound = service.decorate_outbound(
        OutboundMessage(channel="telegram", chat_id="chat-1", content="我在呢，先抱抱你。"),
        inbound_text="我今天真的有点委屈，还有点想哭",
        source="passive",
    )

    assert outbound.attachments
    assert outbound.metadata["meme_decision_driver"] == "priority_emotion"
    assert outbound.metadata["auto_meme_category"] == "hug"


@pytest.mark.asyncio
async def test_priority_user_emotion_can_override_information_gate_for_passive_reply(tmp_path: Path) -> None:
    file_workspace = tmp_path / "files"
    (file_workspace / "memes" / "hug").mkdir(parents=True)
    (file_workspace / "memes" / "hug" / "001.png").write_bytes(b"fake-image")
    (file_workspace / "memes" / "manifest.json").write_text(
        (
            '{"version":1,"categories":{"hug":{"desc":"安慰抱抱","aliases":["抱抱","安慰"],'
            '"mood_tags":["安慰","委屈"],"usage_scenarios":["安慰回复"],"enabled":true,"files":["001.png"]}}}'
        ),
        encoding="utf-8",
    )
    service = MemeService(file_workspace)

    outbound = service.decorate_outbound(
        OutboundMessage(
            channel="telegram",
            chat_id="chat-1",
            content=(
                "我在呢，再抱紧一点。🫂\n\n"
                "受委屈的时候想发呆、想安静一会儿都很正常。\n\n"
                "你不用急着好起来，我先在这里陪你。"
            ),
        ),
        inbound_text="我今天有点委屈，真的有点难受",
        source="passive",
    )

    assert outbound.attachments
    assert outbound.metadata["meme_decision_driver"] == "priority_emotion"


@pytest.mark.asyncio
async def test_service_removes_meme_offer_text_when_attachment_is_added(tmp_path: Path) -> None:
    file_workspace = tmp_path / "files"
    (file_workspace / "memes" / "hug").mkdir(parents=True)
    (file_workspace / "memes" / "hug" / "001.png").write_bytes(b"fake-image")
    (file_workspace / "memes" / "manifest.json").write_text(
        (
            '{"version":1,"categories":{"hug":{"desc":"安慰抱抱","aliases":["抱抱","安慰"],'
            '"mood_tags":["安慰","委屈"],"usage_scenarios":["安慰回复"],"enabled":true,"files":["001.png"]}}}'
        ),
        encoding="utf-8",
    )
    service = MemeService(file_workspace)

    outbound = service.decorate_outbound(
        OutboundMessage(
            channel="telegram",
            chat_id="chat-1",
            content=(
                "我在呢，再抱紧一点。🫂\n\n"
                "要不要我给你发个看起来傻乎乎但其实很治愈的表情包，稍微转换一下心情？"
            ),
        ),
        inbound_text="我今天有点委屈",
        source="passive",
    )

    assert outbound.attachments
    assert "表情包" not in outbound.content
    assert "我在呢，再抱紧一点" in outbound.content


@pytest.mark.asyncio
async def test_service_removes_attachment_contradiction_text_when_attachment_is_added(tmp_path: Path) -> None:
    file_workspace = tmp_path / "files"
    (file_workspace / "memes" / "happy").mkdir(parents=True)
    (file_workspace / "memes" / "happy" / "001.png").write_bytes(b"fake-image")
    (file_workspace / "memes" / "manifest.json").write_text(
        '{"version":1,"categories":{"happy":{"desc":"开心表情","aliases":["开心"],"enabled":true,"files":["001.png"]}}}',
        encoding="utf-8",
    )
    service = MemeService(file_workspace)

    outbound = service.decorate_outbound(
        OutboundMessage(
            channel="telegram",
            chat_id="chat-1",
            content=(
                "抱歉呀，我这里暂时还没办法直接给你发图片文件。\n\n"
                "不过我先给你来一张开心的！"
            ),
        ),
        inbound_text="来个表情包",
        source="passive",
    )

    assert outbound.attachments
    assert "没办法直接给你发图片文件" not in outbound.content
    assert "不过我先给你来一张开心的" in outbound.content


@pytest.mark.asyncio
async def test_information_heavy_feed_skips_auto_attach(tmp_path: Path) -> None:
    file_workspace = tmp_path / "files"
    (file_workspace / "memes" / "happy").mkdir(parents=True)
    (file_workspace / "memes" / "happy" / "001.png").write_bytes(b"fake-image")
    (file_workspace / "memes" / "manifest.json").write_text(
        '{"version":1,"categories":{"happy":{"desc":"开心表情","aliases":["开心"],"enabled":true,"files":["001.png"]}}}',
        encoding="utf-8",
    )
    service = MemeService(file_workspace)

    outbound = service.decorate_outbound(
        OutboundMessage(
            channel="telegram",
            chat_id="chat-1",
            content="新闻摘要：今天市场高开低走；更多信息见 https://example.com ，请理性参考。",
        ),
        inbound_text="",
        source="feed",
    )

    assert not outbound.attachments


@pytest.mark.asyncio
async def test_proactive_can_auto_attach_meme(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    file_workspace = tmp_path / "files"
    (file_workspace / "memes" / "hug").mkdir(parents=True)
    (file_workspace / "memes" / "hug" / "001.png").write_bytes(b"fake-image")
    (file_workspace / "memes" / "manifest.json").write_text(
        '{"version":1,"categories":{"hug":{"desc":"安慰抱抱","aliases":["抱抱","安慰"],"enabled":true,"files":["001.png"]}}}',
        encoding="utf-8",
    )
    channel = FakeChannel()
    candidate = ProactiveCandidate(
        candidate_id="drift-1",
        source_type="drift",
        title="抱抱",
        body="来，给你一个抱抱",
        url="",
        confidence=0.9,
        novelty=0.6,
        user_fit=0.8,
        priority=0.8,
        shareable=True,
        created_at=utc_now(),
        expires_at=utc_now() + timedelta(hours=1),
        dedupe_key="drift-1",
    )

    class FakeDriftManager:
        async def run_once(self):
            return type("Result", (), {"ran": True, "candidate": candidate, "reason": None})()

    loop = ProactiveLoop(
        store,
        channel,
        target_chat_id="chat-1",
        budget=ProactiveBudgetConfig(daily_max=6, min_interval_minutes=0, quiet_hours_start="", quiet_hours_end=""),
        drift_config=DriftConfig(
            enabled=True,
            tasks_path=tmp_path / "drift_tasks.json",
            output_dir=tmp_path / "runs",
            run_cooldown_minutes=0,
            daily_run_cap=3,
            promotion_enabled=True,
            daily_cap=2,
            skills_enabled=False,
            skills_workspace_dir=tmp_path / "drift_skills",
            skills_include_builtin=False,
        ),
        drift_manager=FakeDriftManager(),
        meme_service=MemeService(file_workspace),
    )

    await loop.tick()

    assert channel.sent
    assert channel.sent[0].attachments
    assert channel.sent[0].attachments[0].local_path.endswith("001.png")
