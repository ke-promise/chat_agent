from __future__ import annotations

from datetime import datetime, timezone

from chat_agent.scheduler import parse_after_reminder


def test_parse_after_reminder() -> None:
    now = datetime(2026, 4, 22, 1, 0, tzinfo=timezone.utc)

    parsed = parse_after_reminder("1分钟后提醒我喝水", now=now)

    assert parsed is not None
    assert parsed.content == "喝水"
    assert parsed.due_at.minute == 1
