from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from chat_agent.config import DriftConfig, FallbackConfig, FeedConfig, ProactiveBudgetConfig
from chat_agent.memes import MemeService
from chat_agent.memory.store import SQLiteStore, utc_now
from chat_agent.messages import OutboundMessage
from chat_agent.proactive.feed import FeedSource, ProactiveFeedManager
from chat_agent.proactive.loop import ProactiveLoop
from chat_agent.proactive.models import ProactiveCandidate


class FakeChannel:
    def __init__(self) -> None:
        self.sent: list[OutboundMessage] = []

    async def send(self, message: OutboundMessage) -> None:
        self.sent.append(message)


class FakeFeedManager:
    def __init__(self, candidates: list[ProactiveCandidate]) -> None:
        self._candidates = candidates
        self.acked: list[str] = []

    def enabled_count(self) -> int:
        return 1

    def connected_count(self) -> int:
        return 1

    async def poll(self):
        return list(self._candidates)

    async def ack(self, candidate):
        self.acked.append(candidate.candidate_id)


class FakeDriftManager:
    def __init__(self, result) -> None:
        self.result = result
        self.calls = 0

    async def run_once(self):
        self.calls += 1
        return self.result


class FakeProvider:
    async def chat(self, messages, tools=None):
        return type("Result", (), {"ok": True, "content": "我在这儿，轻轻冒个泡。"})()


class RewritingProvider:
    def __init__(self, content: str) -> None:
        self.content = content

    async def chat(self, messages, tools=None):
        return type("Result", (), {"ok": True, "content": self.content})()


class CountingProvider:
    def __init__(self, content: str = "我在这儿，轻轻冒个泡。") -> None:
        self.content = content
        self.calls = 0

    async def chat(self, messages, tools=None):
        self.calls += 1
        return type("Result", (), {"ok": True, "content": self.content})()


def _candidate(
    source_type: str,
    candidate_id: str,
    body: str,
    *,
    priority: float,
    confidence: float,
    novelty: float,
    user_fit: float,
    shareable: bool = True,
) -> ProactiveCandidate:
    now = utc_now()
    return ProactiveCandidate(
        candidate_id=candidate_id,
        source_type=source_type,
        title=body,
        body=body,
        url="",
        confidence=confidence,
        novelty=novelty,
        user_fit=user_fit,
        priority=priority,
        shareable=shareable,
        created_at=now,
        expires_at=now + timedelta(hours=2),
        dedupe_key=candidate_id,
    )


def _loop(tmp_path: Path, store: SQLiteStore, channel: FakeChannel, **kwargs) -> ProactiveLoop:
    return ProactiveLoop(
        store=store,
        channel=channel,
        enabled=True,
        tick_interval_seconds=1,
        max_due_per_tick=50,
        target_chat_id="chat-1",
        budget=kwargs.get("budget", ProactiveBudgetConfig(daily_max=6, min_interval_minutes=0, quiet_hours_start="", quiet_hours_end="")),
        fallback_config=kwargs.get("fallback_config", FallbackConfig(enabled=False, probability=1.0, daily_cap=2)),
        feed_config=kwargs.get("feed_config", FeedConfig(enabled=False, sources_path=tmp_path / "sources.json", daily_cap=3)),
        drift_config=kwargs.get(
            "drift_config",
            DriftConfig(
                enabled=False,
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
        ),
        fallback_provider=kwargs.get("fallback_provider"),
        feed_manager=kwargs.get("feed_manager"),
        drift_manager=kwargs.get("drift_manager"),
    )


@pytest.mark.asyncio
async def test_proactive_sends_due_reminder_even_when_budget_exhausted(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    for idx in range(6):
        await store.add_proactive_delivery("chat-1", f"old-{idx}", "feed")
    await store.add_reminder("chat-1", "user-1", "喝水", datetime.now(timezone.utc) - timedelta(seconds=1))
    channel = FakeChannel()
    loop = _loop(tmp_path, store, channel, budget=ProactiveBudgetConfig(daily_max=0, min_interval_minutes=999, quiet_hours_start="", quiet_hours_end=""))

    await loop.tick()

    assert channel.sent[0].content == "叮咚，小提醒到啦：喝水"


@pytest.mark.asyncio
async def test_feed_candidates_are_ranked_globally_not_by_input_order(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    channel = FakeChannel()
    low = _candidate("feed", "feed-low", "旧闻", priority=0.5, confidence=0.6, novelty=0.2, user_fit=0.5)
    high = _candidate("feed", "feed-high", "新鲜高分内容", priority=0.8, confidence=0.9, novelty=0.95, user_fit=0.7)
    manager = FakeFeedManager([low, high])
    loop = _loop(
        tmp_path,
        store,
        channel,
        feed_config=FeedConfig(enabled=True, sources_path=tmp_path / "sources.json", daily_cap=3),
        feed_manager=manager,
    )

    await loop.tick()

    assert len(channel.sent) == 1
    assert channel.sent[0].content == "我刚刷到一个你可能会感兴趣的小发现，顺手塞给你看看：新鲜高分内容"
    assert manager.acked == ["feed-high"]

@pytest.mark.asyncio
async def test_fallback_can_send_when_feed_candidates_are_all_duplicates(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    channel = FakeChannel()
    duplicate = _candidate("feed", "feed-seen", "旧闻", priority=0.8, confidence=0.9, novelty=0.8, user_fit=0.7)
    await store.mark_seen_item(duplicate.dedupe_key, duplicate.source_type, duplicate.title, duplicate.url)
    loop = _loop(
        tmp_path,
        store,
        channel,
        feed_config=FeedConfig(enabled=True, sources_path=tmp_path / "sources.json", daily_cap=3),
        feed_manager=FakeFeedManager([duplicate]),
        fallback_config=FallbackConfig(enabled=True, probability=1.0, daily_cap=2),
        fallback_provider=RewritingProvider("我在这儿，轻轻冒个泡。"),
    )

    await loop.tick()

    assert len(channel.sent) == 1
    assert channel.sent[0].content == "我在这儿，轻轻冒个泡。"
    tick = await store.get_last_proactive_tick()
    assert tick["action"] == "sent"


@pytest.mark.asyncio
async def test_drift_runs_but_non_shareable_result_is_not_sent(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    channel = FakeChannel()
    drift_result = type("Result", (), {"ran": True, "reason": None, "candidate": None})()
    loop = _loop(
        tmp_path,
        store,
        channel,
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
        drift_manager=FakeDriftManager(drift_result),
    )

    await loop.tick()
    tick = await store.get_last_proactive_tick()

    assert channel.sent == []
    assert tick["skip_reason"] == "no_candidate"


@pytest.mark.asyncio
async def test_promoted_drift_candidate_still_respects_budget(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    channel = FakeChannel()
    candidate = _candidate("drift", "drift-1", "可分享的 drift 结果", priority=0.8, confidence=0.9, novelty=0.6, user_fit=0.8)
    drift_result = type("Result", (), {"ran": True, "reason": None, "candidate": candidate})()
    loop = _loop(
        tmp_path,
        store,
        channel,
        budget=ProactiveBudgetConfig(daily_max=0, min_interval_minutes=0, quiet_hours_start="", quiet_hours_end=""),
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
        drift_manager=FakeDriftManager(drift_result),
    )

    await loop.tick()

    assert channel.sent == []


@pytest.mark.asyncio
async def test_fallback_only_sends_when_no_higher_ranked_candidate(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    channel = FakeChannel()
    feed_candidate = _candidate("feed", "feed-1", "更值得发的 feed", priority=0.8, confidence=0.95, novelty=0.95, user_fit=0.6)
    manager = FakeFeedManager([feed_candidate])
    loop = _loop(
        tmp_path,
        store,
        channel,
        feed_config=FeedConfig(enabled=True, sources_path=tmp_path / "sources.json", daily_cap=3),
        fallback_config=FallbackConfig(enabled=True, probability=1.0, daily_cap=2),
        fallback_provider=FakeProvider(),
        feed_manager=manager,
    )

    await loop.tick()

    assert len(channel.sent) == 1
    assert manager.acked == ["feed-1"]


@pytest.mark.asyncio
async def test_feed_candidate_can_be_rewritten_into_companion_share(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    channel = FakeChannel()
    candidate = _candidate("feed", "feed-1", "新鲜高分内容", priority=0.8, confidence=0.95, novelty=0.95, user_fit=0.7)
    candidate.summary = "这条内容提到一个刚发生的新变化。"
    candidate.url = "https://example.test/story"
    loop = _loop(
        tmp_path,
        store,
        channel,
        feed_config=FeedConfig(enabled=True, sources_path=tmp_path / "sources.json", daily_cap=3),
        fallback_provider=RewritingProvider("刚刷到一个你可能会想点开的新东西：新鲜高分内容\nhttps://example.test/story"),
        feed_manager=FakeFeedManager([candidate]),
    )

    await loop.tick()

    assert len(channel.sent) == 1
    assert channel.sent[0].content == "刚刷到一个你可能会想点开的新东西：新鲜高分内容\nhttps://example.test/story"


@pytest.mark.asyncio
async def test_feed_cover_image_is_preferred_over_auto_meme(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    file_workspace = tmp_path / "files"
    (file_workspace / "memes" / "hug").mkdir(parents=True)
    (file_workspace / "memes" / "hug" / "001.png").write_bytes(b"fake-image")
    (file_workspace / "memes" / "manifest.json").write_text(
        '{"version":1,"categories":{"hug":{"desc":"安慰抱抱","aliases":["抱抱","安慰"],"enabled":true,"files":["001.png"]}}}',
        encoding="utf-8",
    )
    channel = FakeChannel()
    candidate = _candidate("feed", "feed-1", "来看看这个更新", priority=0.8, confidence=0.95, novelty=0.95, user_fit=0.7)
    candidate.image_url = "https://example.test/cover.jpg"
    loop = ProactiveLoop(
        store=store,
        channel=channel,
        enabled=True,
        tick_interval_seconds=1,
        max_due_per_tick=50,
        target_chat_id="chat-1",
        budget=ProactiveBudgetConfig(daily_max=6, min_interval_minutes=0, quiet_hours_start="", quiet_hours_end=""),
        feed_config=FeedConfig(enabled=True, sources_path=tmp_path / "sources.json", daily_cap=3),
        feed_manager=FakeFeedManager([candidate]),
        meme_service=MemeService(file_workspace),
    )

    await loop.tick()

    assert len(channel.sent) == 1
    assert channel.sent[0].attachments
    assert channel.sent[0].attachments[0].url == "https://example.test/cover.jpg"
    assert channel.sent[0].attachments[0].local_path is None


@pytest.mark.asyncio
async def test_natural_drift_message_skips_optional_llm_rewrites(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    channel = FakeChannel()
    candidate = _candidate("drift", "drift-1", "刚看到一个小东西，感觉你可能会喜欢。", priority=0.8, confidence=0.9, novelty=0.7, user_fit=0.8)
    drift_result = type("Result", (), {"ran": True, "reason": None, "candidate": candidate})()
    provider = CountingProvider()
    loop = _loop(
        tmp_path,
        store,
        channel,
        fallback_config=FallbackConfig(enabled=True, probability=1.0, daily_cap=2),
        fallback_provider=provider,
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
        drift_manager=FakeDriftManager(drift_result),
    )

    await loop.tick()

    assert len(channel.sent) == 1
    assert channel.sent[0].content == "刚看到一个小东西，感觉你可能会喜欢。"
    assert provider.calls == 0


@pytest.mark.asyncio
async def test_feed_prefers_interest_matched_candidate(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    await store.update_user_profile("chat-1", {"preferences": ["鸣潮", "Wuthering Waves"]})
    await store.add_memory("chat-1", "喜欢玩鸣潮", tags=["preference"], memory_type="preference", source_kind="explicit", confidence=1.0)
    channel = FakeChannel()
    unrelated = _candidate("feed", "feed-python", "Python 版本更新", priority=0.8, confidence=0.95, novelty=0.95, user_fit=0.55)
    related = _candidate("feed", "feed-wuwa", "鸣潮 2.5 版本前瞻", priority=0.7, confidence=0.9, novelty=0.8, user_fit=0.55)
    loop = _loop(
        tmp_path,
        store,
        channel,
        feed_config=FeedConfig(enabled=True, sources_path=tmp_path / "sources.json", daily_cap=3),
        feed_manager=FakeFeedManager([unrelated, related]),
    )

    await loop.tick()

    assert len(channel.sent) == 1
    assert channel.sent[0].content == "我刚刷到一个你可能会感兴趣的小发现，顺手塞给你看看：鸣潮 2.5 版本前瞻"


@pytest.mark.asyncio
async def test_deferred_drift_candidate_is_ranked_when_drift_has_no_new_candidate(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    channel = FakeChannel()
    candidate = _candidate("drift", "drift-deferred", "saved search result", priority=0.8, confidence=0.9, novelty=0.7, user_fit=0.8)
    await store.add_proactive_candidate(
        chat_id="chat-1",
        candidate_id=candidate.candidate_id,
        source_type=candidate.source_type,
        title=candidate.title,
        body=candidate.body,
        url=candidate.url,
        confidence=candidate.confidence,
        novelty=candidate.novelty,
        user_fit=candidate.user_fit,
        priority=candidate.priority,
        shareable=candidate.shareable,
        dedupe_key=candidate.dedupe_key,
        artifact_path=candidate.artifact_path,
        created_at=candidate.created_at,
        expires_at=candidate.expires_at,
        score=0.7,
        status="dropped",
        drop_reason="budget",
    )
    drift_result = type("Result", (), {"ran": False, "reason": "daily_run_cap", "candidate": None})()
    loop = _loop(
        tmp_path,
        store,
        channel,
        drift_config=DriftConfig(
            enabled=True,
            tasks_path=tmp_path / "drift_tasks.json",
            output_dir=tmp_path / "runs",
            run_cooldown_minutes=0,
            daily_run_cap=0,
            promotion_enabled=True,
            daily_cap=2,
            skills_enabled=False,
            skills_workspace_dir=tmp_path / "drift_skills",
            skills_include_builtin=False,
        ),
        drift_manager=FakeDriftManager(drift_result),
    )

    await loop.tick()

    assert len(channel.sent) == 1
    assert channel.sent[0].content == "saved search result"


@pytest.mark.asyncio
async def test_deferred_drift_candidate_skips_seen_item(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    channel = FakeChannel()
    candidate = _candidate("drift", "drift-seen", "already sent elsewhere", priority=0.8, confidence=0.9, novelty=0.7, user_fit=0.8)
    await store.add_proactive_candidate(
        chat_id="chat-1",
        candidate_id=candidate.candidate_id,
        source_type=candidate.source_type,
        title=candidate.title,
        body=candidate.body,
        url=candidate.url,
        confidence=candidate.confidence,
        novelty=candidate.novelty,
        user_fit=candidate.user_fit,
        priority=candidate.priority,
        shareable=candidate.shareable,
        dedupe_key=candidate.dedupe_key,
        artifact_path=candidate.artifact_path,
        created_at=candidate.created_at,
        expires_at=candidate.expires_at,
        score=0.7,
        status="dropped",
        drop_reason="busy",
    )
    await store.mark_seen_item(candidate.dedupe_key, candidate.source_type, candidate.title, candidate.url)
    drift_result = type("Result", (), {"ran": False, "reason": "daily_run_cap", "candidate": None})()
    loop = _loop(
        tmp_path,
        store,
        channel,
        drift_config=DriftConfig(
            enabled=True,
            tasks_path=tmp_path / "drift_tasks.json",
            output_dir=tmp_path / "runs",
            run_cooldown_minutes=0,
            daily_run_cap=0,
            promotion_enabled=True,
            daily_cap=2,
            skills_enabled=False,
            skills_workspace_dir=tmp_path / "drift_skills",
            skills_include_builtin=False,
        ),
        drift_manager=FakeDriftManager(drift_result),
    )

    await loop.tick()
    tick = await store.get_last_proactive_tick()

    assert channel.sent == []
    assert tick["skip_reason"] == "no_candidate"


def test_feed_manager_extracts_rss_articles(tmp_path: Path) -> None:
    manager = ProactiveFeedManager(tmp_path / "missing.json", None)
    source = FeedSource(
        server="rss",
        channel="content",
        poll_tool=None,
        get_tool="get_content",
        ack_tool=None,
        poll_args={},
        get_args={},
    )

    events = manager._extract_events(
        source,
        {
            "articles": [
                {
                    "id": 7,
                    "title": "RSS Title",
                    "content": "body",
                    "link": "https://example.test/a",
                    "pubDate": "2026-04-22T12:00:00Z",
                    "feedTitle": "Example Feed",
                }
            ]
        },
    )

    assert events[0]["event_id"] == "7"
    assert events[0]["title"] == "RSS Title"
    assert events[0]["url"] == "https://example.test/a"
    assert events[0]["source"] == "Example Feed"


@pytest.mark.asyncio
async def test_feed_manager_skips_disconnected_source(tmp_path: Path) -> None:
    config = tmp_path / "sources.json"
    config.write_text(
        '{"sources":[{"server":"rss","channel":"content","get_tool":"get_content","enabled":true}]}',
        encoding="utf-8",
    )
    manager = ProactiveFeedManager(config, type("Registry", (), {"servers": {}})())

    events = await manager.poll()

    assert events == []
    assert manager.enabled_count() == 1
    assert manager.connected_count() == 0
