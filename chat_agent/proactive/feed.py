"""Proactive 外部 feed source 轮询。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

from chat_agent.mcp.registry import MCPRegistry
from chat_agent.memory.store import _from_iso, utc_now
from chat_agent.proactive.models import ProactiveCandidate

logger = logging.getLogger(__name__)


def _compact_summary(text: str, limit: int = 120) -> str:
    """压缩摘要。

    参数:
        text: 参与压缩摘要的 `text` 参数。
        limit: 参与压缩摘要的 `limit` 参数。

    返回:
        返回与本函数处理结果对应的数据。
    """
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "…"


@dataclass(slots=True)
class FeedSource:
    """一个主动 feed 数据源的配置项。

    字段:
        server: 提供该 feed 的 MCP server 名称。
        channel: feed 所属逻辑频道，用于筛选事件。
        poll_tool: 主动触发抓取的工具名；不需要轮询时可为空。
        get_tool: 读取 feed 事件列表的工具名。
        ack_tool: 确认已消费事件的工具名；不支持确认时可为空。
        poll_args: 调用 `poll_tool` 时附带的固定参数。
        get_args: 调用 `get_tool` 时附带的固定参数。
        enabled: 该 feed source 是否启用。
    """

    server: str
    channel: str
    poll_tool: str | None
    get_tool: str
    ack_tool: str | None
    poll_args: dict[str, Any]
    get_args: dict[str, Any]
    enabled: bool = True


class ProactiveFeedManager:
    """主动 feed 管理器。"""

    def __init__(self, sources_path: Path, mcp_registry: MCPRegistry | None) -> None:
        """初始化 `ProactiveFeedManager` 实例。

        参数:
            sources_path: 初始化 `ProactiveFeedManager` 时需要的 `sources_path` 参数。
            mcp_registry: 初始化 `ProactiveFeedManager` 时需要的 `mcp_registry` 参数。
        """
        self.sources_path = Path(sources_path)
        self.mcp_registry = mcp_registry
        self.sources: list[FeedSource] = []
        self._events_by_candidate_id: dict[str, dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        """从 JSON 配置文件加载 feed source 列表。"""
        self.sources = []
        if not self.sources_path.exists():
            logger.warning("Proactive sources config not found: %s", self.sources_path)
            return
        data = json.loads(self.sources_path.read_text(encoding="utf-8"))
        for item in data.get("sources", []):
            self.sources.append(
                FeedSource(
                    server=str(item["server"]),
                    channel=str(item.get("channel", "content")),
                    poll_tool=str(item["poll_tool"]) if item.get("poll_tool") else None,
                    get_tool=str(item["get_tool"]),
                    ack_tool=item.get("ack_tool"),
                    poll_args=dict(item.get("poll_args", {})),
                    get_args=dict(item.get("get_args", {})),
                    enabled=bool(item.get("enabled", True)),
                )
            )

    def enabled_count(self) -> int:
        """处理`count`。

        返回:
            返回与本函数处理结果对应的数据。"""
        return len([source for source in self.sources if source.enabled])

    def connected_count(self) -> int:
        """处理`count`。

        返回:
            返回与本函数处理结果对应的数据。"""
        return len([source for source in self.sources if source.enabled and self._server_connected(source.server)])

    async def poll(self) -> list[ProactiveCandidate]:
        """轮询所有启用且已连接的 feed source，并返回统一主动候选。"""
        if not self.mcp_registry:
            return []
        self._events_by_candidate_id = {}
        candidates: list[ProactiveCandidate] = []
        for source in self.sources:
            if not source.enabled:
                continue
            if not self._server_connected(source.server):
                logger.info("Feed source skipped: server_not_connected server=%s", source.server)
                continue
            try:
                if source.poll_tool:
                    await self.mcp_registry.call_tool(source.server, source.poll_tool, source.poll_args)
                get_args = {"channel": source.channel, **source.get_args}
                result = await self.mcp_registry.call_tool(source.server, source.get_tool, get_args)
                for event in self._extract_events(source, result):
                    candidate = self._build_candidate(source, event)
                    self._events_by_candidate_id[candidate.candidate_id] = event
                    candidates.append(candidate)
            except Exception:
                logger.warning("Feed source poll failed server=%s", source.server, exc_info=True)
        return candidates

    def _server_connected(self, server_name: str) -> bool:
        """处理`connected`。

        参数:
            server_name: 参与处理`connected`的 `server_name` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        return bool(self.mcp_registry and server_name in self.mcp_registry.servers)

    async def ack(self, candidate: ProactiveCandidate) -> None:
        """向 feed source 确认某个候选已处理。"""
        event = self._events_by_candidate_id.get(candidate.candidate_id)
        source = event.get("_feed_source") if isinstance(event, dict) else None
        if not self.mcp_registry or not isinstance(source, FeedSource) or not source.ack_tool:
            return
        event_id = event.get("event_id") or event.get("id") or event.get("url")
        try:
            await self.mcp_registry.call_tool(source.server, source.ack_tool, {"event_ids": [event_id]})
        except Exception:
            logger.warning("Feed source ack failed server=%s event=%s", source.server, event_id, exc_info=True)

    def _extract_events(self, source: FeedSource, result: Any) -> list[dict[str, Any]]:
        """把 MCP 工具返回值标准化为 feed event 列表。"""
        if isinstance(result, dict) and isinstance(result.get("events"), list):
            raw_items = result.get("events", [])
        elif isinstance(result, list):
            raw_items = result
        elif isinstance(result, dict):
            raw_items = result.get("articles") or result.get("items") or result.get("content") or []
        else:
            raw_items = []
        if isinstance(raw_items, dict):
            raw_items = raw_items.get("events") or raw_items.get("articles") or raw_items.get("items") or []

        events: list[dict[str, Any]] = []
        for item in raw_items if isinstance(raw_items, list) else []:
            if not isinstance(item, dict):
                continue
            title = item.get("title") or item.get("name") or item.get("headline") or "Untitled"
            url = item.get("url") or item.get("link") or item.get("href") or ""
            event_id = item.get("event_id") or item.get("id") or item.get("guid") or url or title
            events.append(
                {
                    "event_id": str(event_id),
                    "title": str(title),
                    "url": str(url),
                    "source": str(item.get("source") or item.get("feedTitle") or source.server),
                    "content": str(item.get("content") or item.get("description") or item.get("summary") or ""),
                    "published_at": str(
                        item.get("published_at") or item.get("pubDate") or item.get("published") or item.get("updated") or ""
                    ),
                    "image_url": str(
                        item.get("image_url")
                        or item.get("image")
                        or item.get("thumbnail")
                        or item.get("thumbnail_url")
                        or item.get("cover_image")
                        or ""
                    ),
                    "_raw": item,
                    "_feed_source": source,
                }
            )
        return events

    def _build_candidate(self, source: FeedSource, event: dict[str, Any]) -> ProactiveCandidate:
        """把单条 feed event 转成统一候选。"""
        now = utc_now()
        published_at = _from_iso(str(event.get("published_at") or "").replace("Z", "+00:00")) if event.get("published_at") else None
        recency_hours = 999.0
        if published_at:
            recency_hours = max((now - published_at).total_seconds() / 3600.0, 0.0)
        novelty = 1.0 if recency_hours <= 6 else 0.85 if recency_hours <= 24 else 0.65 if recency_hours <= 72 else 0.45
        expires_at = now + timedelta(hours=24 if novelty >= 0.85 else 72)
        title = str(event.get("title") or "有一条新内容")
        url = str(event.get("url") or "")
        summary = _compact_summary(str(event.get("content") or ""))
        lead = "我刚刷到一个你可能会感兴趣的小发现："
        body_lines = [f"{lead}{title}"]
        if summary and summary != title:
            body_lines.append(summary)
        if url:
            body_lines.append(url)
        body = "\n".join(body_lines)
        dedupe_key = "|".join(
            [
                str(event.get("event_id") or ""),
                url,
                title,
            ]
        ).strip("|")
        return ProactiveCandidate(
            candidate_id=f"feed:{source.server}:{event.get('event_id') or title}",
            source_type="feed",
            title=title,
            body=body,
            url=url,
            confidence=0.9 if title else 0.6,
            novelty=novelty,
            user_fit=0.55,
            priority=0.8,
            shareable=bool(title or url),
            created_at=now,
            expires_at=expires_at,
            dedupe_key=dedupe_key or f"feed:{source.server}:{title}",
            summary=summary,
            source_label=str(event.get("source") or source.server),
            image_url=str(event.get("image_url") or ""),
        )
