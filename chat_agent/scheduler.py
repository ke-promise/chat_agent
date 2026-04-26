"""提醒时间解析与调度辅助。

当前 MVP 主要支持“X 分钟后提醒我做某事”这类 after 模式。
解析结果会由内置 create_reminder 工具写入 SQLite，之后 proactive loop 定时检查到期提醒。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


AFTER_PATTERNS = [
    re.compile(r"(?P<num>\d+)\s*(?P<unit>秒|分钟|分|小时|天)后\s*提醒我\s*(?P<text>.+)"),
    re.compile(r"提醒我\s*(?P<num>\d+)\s*(?P<unit>秒|分钟|分|小时|天)后\s*(?P<text>.+)"),
]


@dataclass(frozen=True)
class ParsedReminder:
    """自然语言提醒解析结果。

    字段:
        content: 到点后需要提醒用户的内容。
        due_at: 提醒到期时间，带时区 datetime。
    """

    content: str
    due_at: datetime


def parse_after_reminder(text: str, now: datetime | None = None) -> ParsedReminder | None:
    """解析“几分钟后提醒我...”这类 after 模式提醒。

    参数:
        text: 用户原始文本。
        now: 可选基准时间，测试时传入固定时间；为空时使用当前 UTC 时间。

    返回:
        成功时返回 ParsedReminder；不匹配、没有提醒内容或格式不支持时返回 None。

    说明:
        第一版只处理 after 模式，例如“1分钟后提醒我喝水”。“明天早上9点”这类绝对自然语言
        可以后续接 fast LLM 或更完整的时间解析器。
    """
    stripped = text.strip()
    match = None
    for pattern in AFTER_PATTERNS:
        match = pattern.search(stripped)
        if match:
            break
    if not match:
        return None

    amount = int(match.group("num"))
    unit = match.group("unit")
    content = match.group("text").strip(" ：:，,")
    if not content:
        return None

    if unit == "秒":
        delta = timedelta(seconds=amount)
    elif unit in {"分钟", "分"}:
        delta = timedelta(minutes=amount)
    elif unit == "小时":
        delta = timedelta(hours=amount)
    else:
        delta = timedelta(days=amount)

    base = now or datetime.now(timezone.utc)
    return ParsedReminder(content=content, due_at=base + delta)
