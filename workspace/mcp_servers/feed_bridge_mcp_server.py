"""把 RSS/Atom feed 包装成 MCP 工具的桥接服务。"""
from __future__ import annotations

import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from typing import Any

from _common import compact_text, http_get, serve

EVENTS: dict[str, dict[str, Any]] = {}
ACKED: set[str] = set()
LAST_POLL = 0.0


def configured_feeds() -> list[dict[str, str]]:
    """处理feed 数据源。

    返回:
        返回与本函数处理结果对应的数据。"""
    raw = os.environ.get("WEB_FEEDS", "[]")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    feeds: list[dict[str, str]] = []
    for item in data if isinstance(data, list) else []:
        if isinstance(item, dict) and item.get("url"):
            feeds.append({"name": str(item.get("name") or item["url"]), "url": str(item["url"])})
    return feeds


def poll_feeds() -> dict[str, Any]:
    """轮询feed 数据源。

    返回:
        返回与本函数处理结果对应的数据。"""
    global LAST_POLL
    LAST_POLL = time.time()
    count = 0
    for feed in configured_feeds():
        try:
            raw = http_get(feed["url"], limit=800_000)
            root = ET.fromstring(raw)
            items = root.findall(".//item") or root.findall(".//{http://www.w3.org/2005/Atom}entry")
            for item in items[:10]:
                event = parse_feed_item(feed, item)
                if event["event_id"] not in ACKED:
                    EVENTS[event["event_id"]] = event
                    count += 1
        except Exception as exc:
            print(f"feed poll failed name={feed['name']} error={exc}", file=sys.stderr, flush=True)
    return {"ok": True, "feeds": len(configured_feeds()), "events": count, "last_poll": LAST_POLL}


def parse_feed_item(feed: dict[str, str], item: ET.Element) -> dict[str, Any]:
    """解析feed 数据、`item`。

    参数:
        feed: 参与解析feed 数据、`item`的 `feed` 参数。
        item: 参与解析feed 数据、`item`的 `item` 参数。

    返回:
        返回与本函数处理结果对应的数据。
    """
    def find_text(*names: str) -> str:
        """查找文本。

        返回:
            返回与本函数处理结果对应的数据。"""
        for name in names:
            found = item.find(name)
            if found is not None and found.text:
                return found.text.strip()
            found = item.find(f"{{http://www.w3.org/2005/Atom}}{name}")
            if found is not None and found.text:
                return found.text.strip()
        return ""

    title = find_text("title") or feed["name"]
    link = find_text("link")
    atom_link = item.find("{http://www.w3.org/2005/Atom}link")
    if not link and atom_link is not None:
        link = atom_link.attrib.get("href", "")
    guid = find_text("guid", "id") or link or title
    content = find_text("description", "summary", "content")
    published = find_text("pubDate", "published", "updated")
    image_url = ""
    enclosure = item.find("enclosure")
    if enclosure is not None and str(enclosure.attrib.get("type", "")).startswith("image/"):
        image_url = enclosure.attrib.get("url", "")
    if not image_url:
        media_content = item.find("{http://search.yahoo.com/mrss/}content")
        if media_content is not None and str(media_content.attrib.get("type", "")).startswith("image/"):
            image_url = media_content.attrib.get("url", "")
    if not image_url:
        media_thumbnail = item.find("{http://search.yahoo.com/mrss/}thumbnail")
        if media_thumbnail is not None:
            image_url = media_thumbnail.attrib.get("url", "")
    if not image_url:
        image_url = find_text("image")
    if published:
        try:
            published = parsedate_to_datetime(published).isoformat()
        except Exception:
            pass
    return {
        "event_id": f"{feed['name']}:{guid}",
        "title": compact_text(title, 300),
        "url": link,
        "source": feed["name"],
        "content": compact_text(content, 2000),
        "published_at": published,
        "image_url": image_url,
    }


def get_proactive_events(channel: str = "content") -> dict[str, Any]:
    """处理`proactive`、事件列表。

    参数:
        channel: 参与处理`proactive`、事件列表的 `channel` 参数。

    返回:
        返回与本函数处理结果对应的数据。
    """
    events = [event for key, event in EVENTS.items() if key not in ACKED]
    return {"channel": channel, "events": events[:10]}


def ack_events(event_ids: list[str]) -> dict[str, Any]:
    """确认事件列表。

    参数:
        event_ids: 参与确认事件列表的 `event_ids` 参数。

    返回:
        返回与本函数处理结果对应的数据。
    """
    for event_id in event_ids:
        ACKED.add(str(event_id))
        EVENTS.pop(str(event_id), None)
    return {"acked": len(event_ids)}


TOOLS = [
    {
        "name": "poll_feeds",
        "description": "拉取配置的 RSS/Atom feed。",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_proactive_events",
        "description": "获取可主动推送的 feed 事件。",
        "inputSchema": {"type": "object", "properties": {"channel": {"type": "string"}}},
    },
    {
        "name": "ack_events",
        "description": "确认 feed 事件已处理。",
        "inputSchema": {"type": "object", "properties": {"event_ids": {"type": "array"}}},
    },
]


def main() -> None:
    """启动相关逻辑。

    返回:
        返回与本函数处理结果对应的数据。"""
    serve(
        "feed_bridge",
        "0.2.0",
        TOOLS,
        {
            "poll_feeds": lambda _: poll_feeds(),
            "get_proactive_events": lambda args: get_proactive_events(str(args.get("channel", "content"))),
            "ack_events": lambda args: ack_events(list(args.get("event_ids", []))),
        },
    )


if __name__ == "__main__":
    main()
