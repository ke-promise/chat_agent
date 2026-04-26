"""主动系统主循环。"""

from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime, time, timedelta

from chat_agent.agent.provider import LLMProvider
from chat_agent.config import DriftConfig, FallbackConfig, FeedConfig, ProactiveBudgetConfig
from chat_agent.memes import MemeService
from chat_agent.memory.interests import build_interest_watchlist, interest_match_score, parse_interest_watchlist_md
from chat_agent.memory.store import SQLiteStore, utc_now
from chat_agent.messages import OutboundAttachment, OutboundMessage
from chat_agent.presence import PresenceTracker
from chat_agent.proactive.drift import DriftManager
from chat_agent.proactive.feed import ProactiveFeedManager
from chat_agent.proactive.models import ProactiveCandidate

logger = logging.getLogger(__name__)


class ProactiveLoop:
    """后台主动循环。"""

    def __init__(
        self,
        store: SQLiteStore,
        channel,
        enabled: bool = True,
        tick_interval_seconds: int = 60,
        max_due_per_tick: int = 50,
        target_chat_id: str = "",
        budget: ProactiveBudgetConfig | None = None,
        fallback_config: FallbackConfig | None = None,
        feed_config: FeedConfig | None = None,
        drift_config: DriftConfig | None = None,
        presence: PresenceTracker | None = None,
        skip_when_busy: bool = True,
        fallback_provider: LLMProvider | None = None,
        feed_manager: ProactiveFeedManager | None = None,
        drift_manager: DriftManager | None = None,
        observe_store: SQLiteStore | None = None,
        meme_service: MemeService | None = None,
    ) -> None:
        """初始化 `ProactiveLoop` 实例。

        参数:
            store: 初始化 `ProactiveLoop` 时需要的 `store` 参数。
            channel: 初始化 `ProactiveLoop` 时需要的 `channel` 参数。
            enabled: 初始化 `ProactiveLoop` 时需要的 `enabled` 参数。
            tick_interval_seconds: 初始化 `ProactiveLoop` 时需要的 `tick_interval_seconds` 参数。
            max_due_per_tick: 初始化 `ProactiveLoop` 时需要的 `max_due_per_tick` 参数。
            target_chat_id: 初始化 `ProactiveLoop` 时需要的 `target_chat_id` 参数。
            budget: 初始化 `ProactiveLoop` 时需要的 `budget` 参数。
            fallback_config: 初始化 `ProactiveLoop` 时需要的 `fallback_config` 参数。
            feed_config: 初始化 `ProactiveLoop` 时需要的 `feed_config` 参数。
            drift_config: 初始化 `ProactiveLoop` 时需要的 `drift_config` 参数。
            presence: 初始化 `ProactiveLoop` 时需要的 `presence` 参数。
            skip_when_busy: 初始化 `ProactiveLoop` 时需要的 `skip_when_busy` 参数。
            fallback_provider: 初始化 `ProactiveLoop` 时需要的 `fallback_provider` 参数。
            feed_manager: 初始化 `ProactiveLoop` 时需要的 `feed_manager` 参数。
            drift_manager: 初始化 `ProactiveLoop` 时需要的 `drift_manager` 参数。
            observe_store: 初始化 `ProactiveLoop` 时需要的 `observe_store` 参数。
            meme_service: 初始化 `ProactiveLoop` 时需要的 `meme_service` 参数。
        """
        self.store = store
        self.channel = channel
        self.enabled = enabled
        self.tick_interval_seconds = tick_interval_seconds
        self.max_due_per_tick = max_due_per_tick
        self.target_chat_id = target_chat_id
        self.budget = budget or ProactiveBudgetConfig(daily_max=6, min_interval_minutes=90, quiet_hours_start="", quiet_hours_end="")
        self.fallback_config = fallback_config or FallbackConfig(enabled=False, probability=0.03, daily_cap=2)
        self.feed_config = feed_config or FeedConfig(enabled=False, sources_path=store.database_path.parent / "unused", daily_cap=3)
        self.drift_config = drift_config or DriftConfig(
            enabled=False,
            tasks_path=store.database_path.parent / "unused.json",
            output_dir=store.database_path.parent / "drift_runs",
            run_cooldown_minutes=180,
            daily_run_cap=3,
            promotion_enabled=True,
            daily_cap=2,
            skills_enabled=False,
            skills_workspace_dir=store.database_path.parent / "drift" / "skills",
            skills_include_builtin=True,
        )
        self.presence = presence
        self.skip_when_busy = skip_when_busy
        self.fallback_provider = fallback_provider
        self.feed_manager = feed_manager
        self.drift_manager = drift_manager
        self.observe_store = observe_store or store
        self.meme_service = meme_service
        self._stopped = asyncio.Event()

    async def run(self) -> None:
        """运行主动循环直到 stop() 被调用。"""
        if not self.enabled:
            logger.info("Proactive loop disabled")
            return
        logger.info("Proactive loop started")
        while not self._stopped.is_set():
            await self.tick()
            logger.info("Next proactive tick in %s seconds", self.tick_interval_seconds)
            try:
                await asyncio.wait_for(self._stopped.wait(), timeout=self.tick_interval_seconds)
            except TimeoutError:
                continue

    async def stop(self) -> None:
        """请求主动循环在下一次等待点停止。"""
        self._stopped.set()

    async def tick(self) -> None:
        """执行一次主动系统调度。"""
        try:
            due = await self.store.get_due_reminders(limit=self.max_due_per_tick)
            logger.info("Checked reminders due_count=%s", len(due))
            if due:
                await self._send_due_reminders(due)
                return

            if not self.target_chat_id:
                await self._add_tick_log("skip", "no_target", 0, None)
                return

            interest_hints = await self._load_interest_watchlist()
            feed_candidates = await self._collect_feed_candidates(interest_hints)
            drift_result = await self._run_drift()
            candidates = list(feed_candidates)
            if drift_result and drift_result.candidate:
                candidates.append(drift_result.candidate)
            if not candidates:
                fallback_candidate = await self._build_fallback_candidate()
                if fallback_candidate:
                    candidates.append(fallback_candidate)

            if not candidates:
                await self._add_tick_log("skip", "no_candidate", 0, None)
                return

            sent = await self._deliver_best_candidate(candidates)
            if not sent:
                await self._add_tick_log("skip", "no_deliverable_candidate", 0, None, content_count=len(candidates))
        except Exception as exc:
            logger.exception("Proactive tick failed")
            await self._add_tick_log("error", None, 0, None, error=str(exc))

    async def _send_due_reminders(self, reminders: list[dict]) -> None:
        """发送本轮查到的到期提醒。"""
        sent_any = False
        for reminder in reminders:
            if await self._skip_for_presence(reminder["chat_id"]):
                await self._add_tick_log("skip", "busy", len(reminders), None)
                continue
            text = f"叮咚，提醒时间到啦：{reminder['content']}"
            await self.channel.send(
                self._decorate_outbound(
                    OutboundMessage(channel="telegram", chat_id=reminder["chat_id"], content=text),
                    source="reminder",
                )
            )
            await self.store.mark_reminder_delivered(int(reminder["id"]))
            await self.store.add_proactive_delivery(reminder["chat_id"], text, source="reminder")
            sent_any = True
        await self._add_tick_log(
            "sent" if sent_any else "skip",
            None if sent_any else "busy",
            len(reminders),
            "due reminders" if sent_any else None,
            sent_count=1 if sent_any else 0,
        )

    async def _collect_feed_candidates(self, interest_hints: list[str]) -> list[ProactiveCandidate]:
        """处理feed 数据、候选集合。

        参数:
            interest_hints: 参与处理feed 数据、候选集合的 `interest_hints` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        if not self.feed_config.enabled or not self.feed_manager:
            return []
        if self.feed_manager.enabled_count() == 0:
            logger.info("Proactive feed skipped: no_source")
            return []
        if self.feed_manager.connected_count() == 0:
            logger.info("Proactive feed skipped: no_connected_source")
            return []
        candidates = await self.feed_manager.poll()
        for candidate in candidates:
            self._apply_interest_fit(candidate, interest_hints)
        return candidates

    async def _run_drift(self):
        """执行`drift`。"""
        if not self.drift_config.enabled or not self.drift_manager:
            return None
        return await self.drift_manager.run_once()

    async def _build_fallback_candidate(self) -> ProactiveCandidate | None:
        """构建`fallback`、候选项。

        返回:
            返回与本函数处理结果对应的数据。"""
        if not self.fallback_config.enabled:
            return None
        if random.random() > self.fallback_config.probability:
            return None
        text = await self._generate_check_in()
        now = utc_now()
        return ProactiveCandidate(
            candidate_id=f"fallback:{now.strftime('%Y%m%d%H%M%S')}",
            source_type="fallback",
            title="轻量问候",
            body=text,
            url="",
            confidence=0.75,
            novelty=0.4,
            user_fit=0.85,
            priority=0.3,
            shareable=True,
            created_at=now,
            expires_at=now + timedelta(hours=2),
            dedupe_key=f"fallback:{text[:50]}",
        )

    async def _load_interest_watchlist(self) -> list[str]:
        """读取当前 chat 的稳定兴趣 watchlist。"""
        interests_path = self.store.database_path.parent / "memory" / self.target_chat_id / "INTERESTS.md"
        if interests_path.exists():
            try:
                hints = parse_interest_watchlist_md(interests_path.read_text(encoding="utf-8"))
                if hints:
                    return hints
            except OSError:
                logger.warning("Failed to read interest watchlist: %s", interests_path)
        user_profile = await self.store.get_user_profile(self.target_chat_id)
        memories = await self.store.list_recent_memories(self.target_chat_id, limit=24)
        return build_interest_watchlist(user_profile, memories)

    def _apply_interest_fit(self, candidate: ProactiveCandidate, interest_hints: list[str]) -> None:
        """根据用户兴趣调整 feed 候选的 user_fit。"""
        if candidate.source_type != "feed":
            return
        combined = "\n".join(part for part in (candidate.title, candidate.body, candidate.url) if part)
        match_score, matched_terms = interest_match_score(combined, interest_hints)
        if match_score <= 0:
            candidate.user_fit = min(candidate.user_fit, 0.18)
            candidate.priority = min(candidate.priority, 0.72)
            return
        candidate.user_fit = max(candidate.user_fit, 0.6 + match_score * 0.35)
        candidate.priority = max(candidate.priority, min(1.0, candidate.priority + match_score * 0.12))

    async def _deliver_best_candidate(self, candidates: list[ProactiveCandidate]) -> bool:
        """过滤、排序并最多发送 1 条非 reminder 主动消息。"""
        now = utc_now()
        busy = await self._skip_for_presence(self.target_chat_id)
        quiet = self._in_quiet_hours(now)
        daily_count = await self.store.count_non_reminder_proactive_deliveries_since(self.target_chat_id, now.replace(hour=0, minute=0, second=0, microsecond=0))
        last_non_reminder = await self.store.last_non_reminder_proactive_delivery_at(self.target_chat_id)
        in_min_interval = bool(last_non_reminder and now - last_non_reminder < timedelta(minutes=self.budget.min_interval_minutes))

        ranked: list[tuple[float, ProactiveCandidate]] = []
        for candidate in candidates:
            drop_reason = await self._candidate_drop_reason(candidate, now, busy, quiet, daily_count, in_min_interval)
            score = self._score_candidate(candidate)
            if drop_reason:
                await self._audit_candidate(candidate, score, "dropped", drop_reason)
                continue
            ranked.append((score, candidate))

        if not ranked:
            return False

        ranked.sort(key=lambda item: (item[0], item[1].created_at), reverse=True)
        score, winner = ranked[0]
        tied = [candidate for candidate_score, candidate in ranked if abs(candidate_score - score) < 0.01]
        if len(tied) > 1:
            winner = await self._break_tie(tied, score)
            score = self._score_candidate(winner)

        text = await self._compose_candidate_message(winner)
        await self.channel.send(self._build_outbound_message(winner, text))
        await self.store.add_proactive_delivery(self.target_chat_id, text, source=winner.source_type)
        await self.store.mark_seen_item(winner.dedupe_key, winner.source_type, winner.title, winner.url)
        if winner.source_type == "feed" and self.feed_manager:
            await self.feed_manager.ack(winner)
        await self._audit_candidate(winner, score, "sent", None, sent_at=now)
        for other_score, candidate in ranked[1:]:
            await self._audit_candidate(candidate, other_score, "dropped", "low_score")
        await self._add_tick_log("sent", None, 0, text, content_count=len(candidates), sent_count=1)
        logger.info("Proactive sent source=%s chat_id=%s text=%r", winner.source_type, self.target_chat_id, text)
        return True

    async def _compose_candidate_message(self, candidate: ProactiveCandidate) -> str:
        """把候选内容整理成更像陪伴分享的最终消息。"""
        fallback = self._fallback_candidate_message(candidate)
        if not self._needs_message_rewrite(candidate) or not self.fallback_provider:
            return fallback
        messages = [
            {
                "role": "system",
                "content": (
                    "你负责把主动候选改写成一条自然的中文陪伴消息。"
                    "语气像顺手分享网上刷到的有趣事情，不要像汇报、简报、播报或客服通知。"
                    "只允许基于给定事实，不要编造细节，不要说自己用了搜索或工具。"
                    "输出 1 到 3 句，尽量控制在 90 字内；如果链接值得保留，可单独放最后一行。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"source_type={candidate.source_type}\n"
                    f"source_label={candidate.source_label or 'unknown'}\n"
                    f"title={candidate.title}\n"
                    f"summary={candidate.summary}\n"
                    f"body={candidate.body}\n"
                    f"url={candidate.url}\n\n"
                    "把它改成可以直接发给用户的话。"
                ),
            },
        ]
        try:
            result = await self.fallback_provider.chat(messages)
        except Exception:
            logger.exception("Candidate message rewrite failed source=%s", candidate.source_type)
            return fallback
        text = result.content.strip() if result.ok else ""
        return self._clean_candidate_message(text) if text else fallback

    def _needs_message_rewrite(self, candidate: ProactiveCandidate) -> bool:
        """只在明显需要时才额外调用 LLM 改写，避免主动流程被可选润色拖慢。"""
        if candidate.source_type == "feed":
            return True
        if candidate.source_type != "drift":
            return False
        body = (candidate.body or "").strip()
        if not body:
            return False
        if candidate.url.strip():
            return True
        if "\n" in body:
            return True
        report_like_markers = (
            "摘要",
            "简报",
            "汇总",
            "观察",
            "整理",
            "候选",
            "建议",
            "待跟进",
            "follow-up",
            "follow up",
        )
        return any(marker in body.lower() for marker in report_like_markers)

    def _fallback_candidate_message(self, candidate: ProactiveCandidate) -> str:
        """处理候选项、消息。

        参数:
            candidate: 参与处理候选项、消息的 `candidate` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        title = candidate.title.strip()
        body = candidate.body.strip()
        summary = candidate.summary.strip()
        url = candidate.url.strip()
        if candidate.source_type == "feed":
            headline = title or body or "我刷到一条新内容"
            lines = [f"我刚刷到一个你可能会感兴趣的小发现：{headline}"]
            if summary and summary not in {headline, body}:
                lines.append(summary)
            if url:
                lines.append(url)
            return "\n".join(lines)
        if candidate.source_type == "drift":
            return body or title or "我刚想到一个也许值得和你分享的小发现。"
        return body or title or "我在旁边待命呢，有事轻轻喊我就好。"

    def _clean_candidate_message(self, text: str) -> str:
        """处理候选项、消息。

        参数:
            text: 参与处理候选项、消息的 `text` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        cleaned = text.strip()
        if cleaned.startswith(("“", "\"", "'")) and cleaned.endswith(("”", "\"", "'")) and len(cleaned) >= 2:
            cleaned = cleaned[1:-1].strip()
        return cleaned or "我在旁边待命呢，有事轻轻喊我就好。"

    async def _candidate_drop_reason(
        self,
        candidate: ProactiveCandidate,
        now: datetime,
        busy: bool,
        quiet: bool,
        daily_count: int,
        in_min_interval: bool,
    ) -> str | None:
        """处理`drop`、`reason`。

        参数:
            candidate: 参与处理`drop`、`reason`的 `candidate` 参数。
            now: 参与处理`drop`、`reason`的 `now` 参数。
            busy: 参与处理`drop`、`reason`的 `busy` 参数。
            quiet: 参与处理`drop`、`reason`的 `quiet` 参数。
            daily_count: 参与处理`drop`、`reason`的 `daily_count` 参数。
            in_min_interval: 参与处理`drop`、`reason`的 `in_min_interval` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        if not candidate.shareable:
            return "not_shareable"
        if candidate.expires_at and candidate.expires_at <= now:
            return "expired"
        if await self.store.has_seen_item(candidate.dedupe_key):
            return "duplicate"
        if busy:
            return "busy"
        if quiet:
            return "quiet_hours"
        if daily_count >= self.budget.daily_max:
            return "budget"
        if in_min_interval:
            return "budget"
        source_cap = await self._source_cap_reached(candidate.source_type, now)
        if source_cap:
            return "source_cap"
        return None

    async def _source_cap_reached(self, source_type: str, now: datetime) -> bool:
        """处理`cap`、`reached`。

        参数:
            source_type: 参与处理`cap`、`reached`的 `source_type` 参数。
            now: 参与处理`cap`、`reached`的 `now` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        since = now.replace(hour=0, minute=0, second=0, microsecond=0)
        count = await self.store.count_proactive_deliveries_for_source_since(self.target_chat_id, source_type, since)
        cap_map = {
            "feed": self.feed_config.daily_cap,
            "drift": self.drift_config.daily_cap,
            "fallback": self.fallback_config.daily_cap,
        }
        cap = cap_map.get(source_type, self.budget.daily_max)
        return cap >= 0 and count >= cap

    def _score_candidate(self, candidate: ProactiveCandidate) -> float:
        """计算评分候选项。

        参数:
            candidate: 参与计算评分候选项的 `candidate` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        source_bonus = {
            "feed": 0.18,
            "drift": 0.08,
            "fallback": -0.12,
        }.get(candidate.source_type, 0.0)
        return (
            candidate.priority * 0.45
            + candidate.confidence * 0.25
            + candidate.novelty * 0.18
            + candidate.user_fit * 0.12
            + source_bonus
        )

    async def _break_tie(self, candidates: list[ProactiveCandidate], score: float) -> ProactiveCandidate:
        """在同分候选之间用 fast LLM 做轻量 tie-break。"""
        if not self.fallback_provider:
            return sorted(candidates, key=lambda item: (item.priority, item.novelty, item.created_at), reverse=True)[0]
        options = []
        for index, candidate in enumerate(candidates, start=1):
            options.append(
                f"{index}. source={candidate.source_type} title={candidate.title} confidence={candidate.confidence:.2f} "
                f"novelty={candidate.novelty:.2f} user_fit={candidate.user_fit:.2f} body={candidate.body[:100]}"
            )
        try:
            result = await self.fallback_provider.chat(
                [
                    {
                        "role": "system",
                        "content": "你只负责在同分主动候选里做 tie-break。请只回答一个数字，选择更适合现在发给用户的一条。",
                    },
                    {"role": "user", "content": f"分数相同({score:.3f})，请从下面选择更该发送的候选：\n" + "\n".join(options)},
                ]
            )
            if result.ok:
                text = result.content.strip()
                if text.isdigit():
                    idx = int(text) - 1
                    if 0 <= idx < len(candidates):
                        return candidates[idx]
        except Exception:
            logger.exception("Proactive tie-break failed")
        return sorted(candidates, key=lambda item: (item.priority, item.novelty, item.created_at), reverse=True)[0]

    async def _audit_candidate(
        self,
        candidate: ProactiveCandidate,
        score: float,
        status: str,
        drop_reason: str | None,
        sent_at: datetime | None = None,
    ) -> None:
        """处理候选项。

        参数:
            candidate: 参与处理候选项的 `candidate` 参数。
            score: 参与处理候选项的 `score` 参数。
            status: 参与处理候选项的 `status` 参数。
            drop_reason: 参与处理候选项的 `drop_reason` 参数。
            sent_at: 参与处理候选项的 `sent_at` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        await self.store.add_proactive_candidate(
            chat_id=self.target_chat_id,
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
            score=score,
            status=status,
            drop_reason=drop_reason,
            sent_at=sent_at,
        )

    async def _generate_check_in(self) -> str:
        """处理`check`、`in`。

        返回:
            返回与本函数处理结果对应的数据。"""
        fallback = "我在旁边待命呢，有事轻轻喊我就好。"
        if not self.fallback_provider:
            return fallback
        result = await self.fallback_provider.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "生成一句陪伴型中文 check-in，不超过 28 字。"
                        "要亲昵、俏皮、可爱一点，但不要要求用户必须回复，不要像客服话术。"
                    ),
                },
                {"role": "user", "content": "现在可以发什么？"},
            ]
        )
        return result.content.strip() if result.ok and result.content.strip() else fallback

    async def _skip_for_presence(self, chat_id: str) -> bool:
        """处理`for`、`presence`。

        参数:
            chat_id: 参与处理`for`、`presence`的 `chat_id` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        if not self.presence or not self.skip_when_busy:
            return False
        return self.presence.is_busy(chat_id)

    def _build_outbound_message(self, candidate: ProactiveCandidate, text: str) -> OutboundMessage:
        """构建出站消息、消息。

        参数:
            candidate: 参与构建出站消息、消息的 `candidate` 参数。
            text: 参与构建出站消息、消息的 `text` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        outbound = OutboundMessage(channel="telegram", chat_id=self.target_chat_id, content=text)
        image_url = candidate.image_url.strip()
        if image_url and candidate.source_type in {"feed", "drift"}:
            outbound = OutboundMessage(
                channel=outbound.channel,
                chat_id=outbound.chat_id,
                content=outbound.content,
                attachments=[OutboundAttachment(kind="photo", url=image_url)],
                metadata={**outbound.metadata, "proactive_image_url": image_url, "proactive_image_source": candidate.source_type},
            )
        return self._decorate_outbound(outbound, source=candidate.source_type)

    def _decorate_outbound(self, outbound: OutboundMessage, source: str) -> OutboundMessage:
        """补充出站消息。

        参数:
            outbound: 参与补充出站消息的 `outbound` 参数。
            source: 参与补充出站消息的 `source` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        if not self.meme_service:
            return outbound
        return self.meme_service.decorate_outbound(outbound, inbound_text=outbound.content, source=source)

    def _in_quiet_hours(self, now: datetime) -> bool:
        """处理`quiet`、`hours`。

        参数:
            now: 参与处理`quiet`、`hours`的 `now` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        start = self._parse_clock(self.budget.quiet_hours_start)
        end = self._parse_clock(self.budget.quiet_hours_end)
        if not start or not end:
            return False
        current = now.astimezone().time()
        if start <= end:
            return start <= current < end
        return current >= start or current < end

    def _parse_clock(self, value: str) -> time | None:
        """解析时刻值。

        参数:
            value: 参与解析时刻值的 `value` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        text = value.strip()
        if not text:
            return None
        try:
            hour, minute = text.split(":", 1)
            return time(hour=int(hour), minute=int(minute))
        except Exception:
            return None

    async def _add_tick_log(
        self,
        action: str,
        skip_reason: str | None,
        reminders_due: int,
        sent_message: str | None,
        error: str | None = None,
        content_count: int = 0,
        sent_count: int = 0,
    ) -> None:
        """添加`tick`、`log`。

        参数:
            action: 参与添加`tick`、`log`的 `action` 参数。
            skip_reason: 参与添加`tick`、`log`的 `skip_reason` 参数。
            reminders_due: 参与添加`tick`、`log`的 `reminders_due` 参数。
            sent_message: 参与添加`tick`、`log`的 `sent_message` 参数。
            error: 参与添加`tick`、`log`的 `error` 参数。
            content_count: 参与添加`tick`、`log`的 `content_count` 参数。
            sent_count: 参与添加`tick`、`log`的 `sent_count` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        await self.observe_store.add_proactive_tick_log(
            action,
            skip_reason,
            reminders_due,
            sent_message,
            error=error,
            content_count=content_count,
            sent_count=sent_count,
        )
        if self.observe_store is not self.store:
            await self.store.add_proactive_tick_log(
                action,
                skip_reason,
                reminders_due,
                sent_message,
                error=error,
                content_count=content_count,
                sent_count=sent_count,
            )
