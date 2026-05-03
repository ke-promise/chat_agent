"""被动消息主业务循环。"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

from chat_agent.context import ContextBuilder
from chat_agent.memes import MemeService
from chat_agent.memory.consolidation import ConsolidationService
from chat_agent.memory.embeddings import EmbeddingProvider
from chat_agent.memory.indexer import MemoryIndexer
from chat_agent.memory.retriever import MemoryRetriever
from chat_agent.memory.store import SQLiteStore, re_split_query
from chat_agent.memory.vector_store import VectorStore
from chat_agent.messages import Attachment, InboundMessage, OutboundMessage
from chat_agent.observe.trace import TraceRecorder
from chat_agent.presence import PresenceTracker
from chat_agent.reasoner import Reasoner
from chat_agent.reply_format import format_reply
from chat_agent.scheduler import parse_after_reminder

logger = logging.getLogger(__name__)


MEMORIZE_RE = re.compile(r"^\s*记住[：:]\s*(?P<content>.+)\s*$")
RECALL_RE = re.compile(r"(你记得什么|回忆一下|记得我|你记得)")
CONFIRM_RE = re.compile(r"^\s*(记住这个|对[，,\s]*就是这个)\s*$")
MEME_SAVE_RE = re.compile(r"(?:存成|记成|收录成|加入)(?:一张|个)?表情包[：: ]*(?P<category>[0-9A-Za-z_\-\u4e00-\u9fff]{1,32})")
RECENT_IMAGE_TTL_SECONDS = 300
AUTO_MEME_POSITIVE_MARKERS = ("表情包", "梗图", "贴纸", "meme", "emoji", "斗图")
AUTO_MEME_NEGATIVE_MARKERS = ("不是表情包", "不像表情包", "普通照片", "截图", "文档", "证件", "二维码")
AUTO_MEME_CATEGORY_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("鼓励", ("加油", "鼓励", "大拇指", "点赞", "没问题", "你很棒", "支持")),
    ("抱抱", ("抱抱", "安慰", "委屈", "难过", "心疼", "哭", "陪着你")),
    ("开心", ("开心", "高兴", "笑", "哈哈", "快乐", "太棒")),
    ("可爱", ("可爱", "萌", "贴贴", "乖", "软乎")),
    ("无语", ("无语", "尴尬", "沉默", "汗颜", "疑惑")),
    ("生气", ("生气", "愤怒", "气鼓鼓", "炸毛")),
    ("害羞", ("害羞", "脸红", "不好意思")),
    ("晚安", ("晚安", "睡觉", "困", "打哈欠")),
]


class AgentLoop:
    """单轮被动对话的业务编排器。"""

    def __init__(
        self,
        store: SQLiteStore,
        context_builder: ContextBuilder,
        reasoner: Reasoner,
        trace_recorder: TraceRecorder,
        presence: PresenceTracker,
        memory_enabled: bool = True,
        max_messages_per_chat: int = 500,
        scheduler_enabled: bool = True,
        summary_enabled: bool = True,
        summary_after_messages: int = 40,
        embedding_provider: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
        memory_indexer: MemoryIndexer | None = None,
        memory_retriever: MemoryRetriever | None = None,
        consolidation_service: ConsolidationService | None = None,
        meme_service: MemeService | None = None,
        model_main: str = "",
        model_fast: str = "",
    ) -> None:
        """初始化被动对话主循环。

        参数:
            store: 项目统一存储层，负责消息、记忆、提醒和审计落盘。
            context_builder: 用于把历史、记忆和工具信息拼装成模型上下文。
            reasoner: 负责驱动主模型推理与工具调用的协调器。
            trace_recorder: 记录消息处理链路 trace 的观测组件。
            presence: 用户活跃状态跟踪器，供主动系统共享。
            memory_enabled: 是否启用轻量记忆抽取、长期记忆写入和回忆能力。
            max_messages_per_chat: 单个会话保留的最大原始消息数，超过后会裁剪旧消息。
            scheduler_enabled: 是否允许解析并创建自然语言提醒。
            summary_enabled: 是否维护会话摘要。
            summary_after_messages: 每累计多少条消息后尝试更新一次摘要。
            embedding_provider: 可选 embedding provider，用于给长期记忆生成向量。
            vector_store: 可选向量存储实现，和 embedding provider 配套使用。
            consolidation_service: 可选后台整理器，用于把旧会话压缩成摘要与候选记忆。
            meme_service: 表情包服务，用于自动挂图和收录图片素材。
        """
        self.store = store
        self.context_builder = context_builder
        self.reasoner = reasoner
        self.trace_recorder = trace_recorder
        self.presence = presence
        self.memory_enabled = memory_enabled
        self.max_messages_per_chat = max_messages_per_chat
        self.scheduler_enabled = scheduler_enabled
        self.summary_enabled = summary_enabled
        self.summary_after_messages = summary_after_messages
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.memory_indexer = memory_indexer or MemoryIndexer(embedding_provider, vector_store)
        self.memory_retriever = memory_retriever or getattr(context_builder, "retriever", None)
        self.consolidation_service = consolidation_service
        self.meme_service = meme_service
        self.model_main = model_main
        self.model_fast = model_fast
        self._recent_image_attachments: dict[str, tuple[Attachment, float]] = {}

    async def handle_message(self, message: InboundMessage) -> OutboundMessage:
        """处理一条入站消息并生成最终出站回复。

        参数:
            message: 来自 Telegram 或内部通道的统一入站消息对象。

        返回:
            返回标准化的 `OutboundMessage`，供 channel 层发送给用户。
        """
        start = time.perf_counter()
        await self.store.record_chat(message.chat_id, message.username)
        self.presence.mark_busy(message.chat_id)
        reply = ""
        tools_used: list[str] = []
        mcp_tools_used: list[str] = []
        memory_hits: list[dict[str, Any]] = []
        error: str | None = None
        hyde_used = False

        try:
            text = message.content.strip()
            if not text and not message.attachments:
                reply = "我收到了一小团空白消息，还没读出你的意思呢。你再轻轻发我一次就好。"
            else:
                direct = await self._handle_direct_paths(message)
                if direct is not None:
                    reply = direct
                else:
                    bundle = await self.context_builder.build(message)
                    memory_hits = [_memory_hit_summary(item) for item in bundle.memory_hits]
                    result = await self.reasoner.run(bundle, message)
                    reply = format_reply(result.reply or "我刚刚没组织出合适的回答，稍等一下我们再来一次。")
                    tools_used = result.tools_used
                    mcp_tools_used = result.mcp_tools_used
                    error = result.error
                    hyde_used = bool(bundle.trace.get("hyde_used"))
                    auto_meme_note = self._auto_ingest_recognized_meme(message, reply)
                    if auto_meme_note:
                        reply = f"{reply}\n\n{auto_meme_note}"

            await self._commit(message, reply)
            latency_ms = int((time.perf_counter() - start) * 1000)
            await self.trace_recorder.record_message(
                chat_id=message.chat_id,
                user_message=message.content,
                assistant_reply=reply,
                tools_used=tools_used,
                memory_hits=memory_hits,
                latency_ms=latency_ms,
                error=error,
                model_main=self.model_main,
                model_fast=self.model_fast,
                mcp_tools_used=mcp_tools_used,
                hyde_used=hyde_used,
                attachments_count=len(message.attachments),
            )
            outbound = OutboundMessage(channel=message.channel, chat_id=message.chat_id, content=reply)
            outbound = self._decorate_outbound(message, outbound)
            logger.info(
                "Assistant reply chat_id=%s text=%r attachments=%s metadata=%s",
                message.chat_id,
                outbound.content,
                len(outbound.attachments),
                outbound.metadata if outbound.attachments else {},
            )
            return outbound
        except Exception as exc:
            logger.exception("Agent loop failed chat_id=%s", message.chat_id)
            reply = "刚刚处理时打了个小结，我已经记下日志啦，稍后我们再试一次。"
            latency_ms = int((time.perf_counter() - start) * 1000)
            await self.trace_recorder.record_message(
                message.chat_id,
                message.content,
                reply,
                tools_used,
                memory_hits,
                latency_ms,
                error=str(exc),
                model_main=self.model_main,
                model_fast=self.model_fast,
                mcp_tools_used=mcp_tools_used,
                attachments_count=len(message.attachments),
            )
            outbound = OutboundMessage(channel=message.channel, chat_id=message.chat_id, content=reply)
            return self._decorate_outbound(message, outbound)
        finally:
            self.presence.mark_idle(message.chat_id)

    async def _handle_direct_paths(self, message: InboundMessage) -> str | None:
        """优先处理无需进入主模型的直达分支。

        例如显式“记住”、提醒创建确认、表情包收录等规则命中的场景，会在这里提前返回。
        """
        text = message.content.strip()
        if message.attachments:
            self._remember_recent_image(message)
            saved = self._extract_meme_category(text)
            if saved and self.meme_service:
                return self._ingest_meme_attachment(message, message.attachments[0], saved)

        if not message.attachments:
            saved = self._extract_meme_category(text)
            if saved and self.meme_service:
                cached = self._pop_recent_image_attachment(message.chat_id)
                if cached:
                    return self._ingest_meme_attachment(message, cached, saved)
                return "我还没拿到可收录的本地图片呢。你可以先发一张图，再在 5 分钟内说“存成表情包：分类”，我就会乖乖收好。"

        if not text:
            return None

        memorize_match = MEMORIZE_RE.match(text)
        if memorize_match:
            if not self.memory_enabled:
                return "记忆小抽屉现在还没打开，暂时先记不进去。"
            content = memorize_match.group("content").strip()
            memory_type = _infer_memory_type(content)
            memory_id = await self.store.add_memory(
                message.chat_id,
                content,
                tags=[memory_type],
                memory_type=memory_type,
                importance=0.8,
                source_chat_id=message.chat_id,
                source_kind="explicit",
                confidence=1.0,
            )
            await self._embed_memory(message.chat_id, memory_id, content)
            logger.info("Created explicit memory id=%s chat_id=%s", memory_id, message.chat_id)
            return f"好呀，我认真记住啦：{content}"

        if self.memory_enabled:
            correction = await self._handle_natural_correction(message)
            if correction is not None:
                return correction

            if CONFIRM_RE.match(text):
                promoted = await self._confirm_latest_candidate(message.chat_id)
                if promoted:
                    return promoted

            if RECALL_RE.search(text):
                query = _normalize_recall_query(text)
                memories = (
                    await self.memory_retriever.retrieve(message.chat_id, query, top_k=5)
                    if self.memory_retriever
                    else await self.store.search_memories(message.chat_id, query, limit=5)
                )
                if memories:
                    lines = "\n".join(f"- #{item['id']} {item['content']}" for item in memories)
                    return f"我翻了翻记忆小本本，找到这些：\n{lines}"
                return "我翻了翻记忆小本本，暂时没找到相关内容。"

        if self.scheduler_enabled:
            parsed = parse_after_reminder(text)
            if parsed is not None:
                reminder_id = await self.store.add_reminder(
                    chat_id=message.chat_id,
                    user_id=message.sender,
                    content=parsed.content,
                    due_at=parsed.due_at,
                )
                local_hint = parsed.due_at.astimezone().strftime("%Y-%m-%d %H:%M:%S")
                logger.info("Created reminder id=%s chat_id=%s due_at=%s", reminder_id, message.chat_id, local_hint)
                return f"好呀，已经帮你系上提醒小铃铛：{local_hint} 提醒你 {parsed.content}"

        return None

    async def _commit(self, message: InboundMessage, reply: str) -> None:
        """把当前轮对话写入会话历史，并触发后续记忆维护。"""
        history_content = message.content
        if message.attachments:
            history_content = history_content or "请描述这张图片。"
            history_content += f"\n[附件数量: {len(message.attachments)}]"
        await self.store.add_session_message(message.chat_id, "user", history_content)
        await self.store.add_session_message(message.chat_id, "assistant", reply)
        await self._extract_light_memory(message)
        self._schedule_consolidation(message.chat_id)
        count = await self.store.count_session_messages(message.chat_id)
        if self.summary_enabled and count >= self.summary_after_messages:
            await self._update_summary(message.chat_id)
        deleted = await self.store.prune_session_messages(message.chat_id, keep=self.max_messages_per_chat)
        logger.info("Committed turn chat_id=%s message_count=%s pruned=%s", message.chat_id, count, deleted)

    async def _extract_light_memory(self, message: InboundMessage) -> None:
        """从当前用户消息中抽取高置信度轻量记忆并写入存储层。"""
        text = message.content.strip()
        if not text:
            return

        for content, memory_type, tags, importance in _high_confidence_inferred_memories(text):
            memory_id = await self.store.add_memory(
                message.chat_id,
                content,
                tags=tags,
                memory_type=memory_type,
                importance=importance,
                source_chat_id=message.chat_id,
                source_kind="inferred",
                confidence=0.7,
            )
            await self._embed_memory(message.chat_id, memory_id, content)

        for content, memory_type, tags, importance, confidence in _candidate_memories(text):
            await self.store.add_memory_candidate(
                message.chat_id,
                content,
                tags=tags,
                memory_type=memory_type,
                importance=importance,
                source_kind="candidate",
                confidence=confidence,
                source_ref=f"turn:{message.chat_id}:{message.message_id or message.created_at.isoformat()}",
            )

        updates = _extract_profile_updates(text)
        if updates:
            await self.store.update_user_profile(message.chat_id, updates)
        await self._promote_candidates(message.chat_id)

    async def _update_summary(self, chat_id: str) -> None:
        """按配置阈值刷新指定会话的摘要。"""
        history = await self.store.get_recent_session_messages(chat_id, limit=12)
        snippets = [f"{item['role']}: {item['content'][:80]}" for item in history[-8:]]
        summary = " | ".join(snippets)[-1200:]
        count = await self.store.count_session_messages(chat_id)
        await self.store.upsert_summary(chat_id, summary, count)

    def _schedule_consolidation(self, chat_id: str) -> None:
        """为指定会话启动一次后台记忆整理任务。

        参数:
            chat_id: 需要整理会话历史和候选记忆的会话 ID。

        说明:
            该任务通过 create_task 后台运行，失败时由回调记录日志，不阻塞当前用户回复。
        """
        if not self.memory_enabled or not self.consolidation_service:
            return
        task = asyncio.create_task(self.consolidation_service.run_once(chat_id), name=f"memory-consolidation-{chat_id}")
        task.add_done_callback(_log_background_task_error)

    async def _handle_natural_correction(self, message: InboundMessage) -> str | None:
        """识别用户的自然语言纠错，并把旧记忆替换成新版记忆。

        参数:
            message: 当前入站消息，内容里可能包含“不是……而是……”这类纠错表达。

        返回:
            命中纠错时返回要直接回复用户的文本；未命中时返回 None，让主模型继续处理。
        """
        parsed = _parse_correction(message.content)
        if not parsed:
            return None
        old_query, new_content = parsed
        memories = await self.store.search_memories(message.chat_id, old_query, limit=5)
        if not memories:
            return "我知道你在纠正我，但我暂时没定位到原来的那条记忆。你可以直接用“记住：...”告诉我新版内容。"
        scored = sorted(
            (
                (_correction_similarity(item["content"], old_query), item)
                for item in memories
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        best_score, best = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        if best_score < 0.70 or best_score - second_score < 0.15:
            lines = "\n".join(f"- #{item['id']} {item['content']}" for _, item in scored[:3])
            return f"我感觉你在纠正记忆，但还没法百分百确定是哪一条。你可以确认一下：\n{lines}"
        new_type = _infer_memory_type(new_content)
        new_id = await self.store.add_memory(
            message.chat_id,
            new_content,
            tags=[new_type, "corrected"],
            memory_type=new_type,
            importance=max(float(best.get("importance", 0.6)), 0.8),
            source_chat_id=message.chat_id,
            source_kind="explicit",
            confidence=1.0,
        )
        await self._embed_memory(message.chat_id, new_id, new_content)
        await self.store.supersede_memory(message.chat_id, int(best["id"]), new_id, "natural correction")
        return f"收到，我已经把旧记忆改成新版啦：{new_content}"

    async def _confirm_latest_candidate(self, chat_id: str) -> str | None:
        """把最近一条候选记忆正式晋升为长期记忆。

        参数:
            chat_id: 当前会话 ID。

        返回:
            晋升成功时返回确认文案；没有候选记忆时返回 None。
        """
        candidates = await self.store.get_memory_candidates(chat_id, limit=3)
        if not candidates:
            return None
        latest = candidates[0]
        memory_id = await self.store.add_memory(
            chat_id,
            latest["content"],
            tags=latest.get("tags"),
            memory_type=str(latest.get("type") or "fact"),
            importance=max(float(latest.get("importance", 0.55)), 0.7),
            source_chat_id=chat_id,
            source_ref=str(latest.get("source_ref") or f"candidate:{latest['id']}"),
            source_kind="promoted",
            confidence=max(float(latest.get("confidence", 0.5)), 0.8),
        )
        await self._embed_memory(chat_id, memory_id, str(latest["content"]))
        await self.store.archive_memory_candidate(chat_id, int(latest["id"]), status="promoted")
        return f"好呀，这条我已经正式记住啦：{latest['content']}"

    async def _promote_candidates(self, chat_id: str) -> None:
        """自动晋升证据足够的候选记忆。

        参数:
            chat_id: 当前会话 ID。

        说明:
            SQLiteStore 会筛出达到证据阈值的候选项；本函数负责写入长期记忆、补向量并归档候选。
        """
        candidates = await self.store.promote_ready_candidates(chat_id, min_evidence=2)
        for candidate in candidates:
            memory_id = await self.store.add_memory(
                chat_id,
                str(candidate["content"]),
                tags=candidate.get("tags"),
                memory_type=str(candidate.get("type") or "fact"),
                importance=max(float(candidate.get("importance", 0.55)), 0.6),
                source_chat_id=chat_id,
                source_ref=str(candidate.get("source_ref") or f"candidate:{candidate['id']}"),
                source_kind="promoted",
                confidence=max(float(candidate.get("confidence", 0.5)), 0.75),
            )
            await self._embed_memory(chat_id, memory_id, str(candidate["content"]))
            await self.store.archive_memory_candidate(chat_id, int(candidate["id"]), status="promoted")

    async def _embed_memory(self, chat_id: str, memory_id: int, content: str) -> None:
        """为一条长期记忆生成 embedding 并写入向量存储。

        参数:
            chat_id: 记忆所属会话 ID。
            memory_id: 已写入 SQLite 的长期记忆 ID。
            content: 需要向量化的记忆正文。
        """
        await self.memory_indexer.index_memory(chat_id, memory_id, content)

    def _decorate_outbound(self, message: InboundMessage, outbound: OutboundMessage) -> OutboundMessage:
        """根据表情包策略为被动回复补充出站附件。

        参数:
            message: 当前入站消息，用于判断用户语气和触发场景。
            outbound: 已生成的出站文本消息。

        返回:
            可能带有表情包附件的新出站消息。
        """
        if not self.meme_service:
            return outbound
        return self.meme_service.decorate_outbound(outbound, inbound_text=message.content, source="passive")

    def _extract_meme_category(self, text: str) -> str:
        """从用户文本中提取“存成表情包：分类”里的分类名。

        参数:
            text: 用户原始文本。

        返回:
            匹配到的分类名；未匹配时返回空字符串。
        """
        match = MEME_SAVE_RE.search(text.strip())
        return match.group("category").strip() if match else ""

    def _remember_recent_image(self, message: InboundMessage) -> None:
        """缓存最近一张图片附件，方便用户下一条文本命令把它收成表情包。"""
        for attachment in message.attachments:
            if attachment.kind == "image":
                self._recent_image_attachments[message.chat_id] = (attachment, time.time())
                return

    def _pop_recent_image_attachment(self, chat_id: str) -> Attachment | None:
        """取出并移除指定会话最近缓存的图片附件，超过有效期则丢弃。"""
        cached = self._recent_image_attachments.pop(chat_id, None)
        if not cached:
            return None
        attachment, created_at = cached
        if time.time() - created_at > RECENT_IMAGE_TTL_SECONDS:
            return None
        return attachment

    def _ingest_meme_attachment(self, message: InboundMessage, attachment: Attachment, category: str) -> str:
        """把用户发来的图片附件收录到本地表情包库。"""
        if not self.meme_service:
            return "表情包服务还没启用，暂时不能收图。"
        result = self.meme_service.ingest_attachment(attachment, category, description=f"{category} 表情包")
        if result.match:
            logger.info(
                "Saved meme chat_id=%s category=%s path=%s status=%s",
                message.chat_id,
                result.match.category,
                result.match.path,
                result.status,
            )
            if result.status == "duplicate":
                return f"这张已经在表情包库里啦：{result.match.category}/{result.match.path.name}"
            return f"好呀，这张我已经收进表情包库啦：{result.match.category}/{result.match.path.name}"
        if result.reason == "category_full":
            return f"{category} 这个分类已经装得太满啦，先清一清再继续收图比较稳妥。"
        if result.reason == "unsupported_attachment":
            return "我看见这张图啦，但它还没成功落到本地。QQ 需要先把 [qq].download_images 改成 true，才能收进表情包库。"
        return "我看见这张图啦，但它还没成功落到本地，暂时没法收进表情包库。"

    def _auto_ingest_recognized_meme(self, message: InboundMessage, reply: str) -> str:
        """当模型识别出图片是表情包时，按规则自动收录并返回附加说明。"""
        if not self.meme_service or not message.attachments or self._extract_meme_category(message.content):
            return ""
        category = self._infer_auto_meme_category(message.content, reply)
        if not category:
            return ""
        attachment = next((item for item in message.attachments if item.kind == "image"), None)
        if not attachment:
            return ""
        result = self.meme_service.ingest_attachment(attachment, category, description=f"{category} 表情包")
        if not result.match:
            logger.info("Auto meme ingest skipped chat_id=%s category=%s reason=%s", message.chat_id, category, result.reason)
            return ""
        logger.info(
            "Auto saved recognized meme chat_id=%s category=%s path=%s status=%s",
            message.chat_id,
            result.match.category,
            result.match.path,
            result.status,
        )
        if result.status == "duplicate":
            return ""
        return f"我也顺手把它收进表情包库啦：{result.match.category}/{result.match.path.name}"

    def _infer_auto_meme_category(self, text: str, reply: str) -> str:
        """根据用户文本和模型回复推断自动收录表情包的分类。"""
        haystack = f"{text}\n{reply}".lower()
        if any(marker.lower() in haystack for marker in AUTO_MEME_NEGATIVE_MARKERS):
            return ""
        if not any(marker.lower() in haystack for marker in AUTO_MEME_POSITIVE_MARKERS):
            return ""
        for category, tokens in AUTO_MEME_CATEGORY_RULES:
            if any(token.lower() in haystack for token in tokens):
                return category
        return "未分类"


def _infer_memory_type(content: str) -> str:
    """根据记忆正文粗略推断长期记忆类型。

    参数:
        content: 待分类的记忆正文。

    返回:
        preference、procedure、event 或 fact。
    """
    if any(token in content for token in ["喜欢", "偏好", "习惯"]):
        return "preference"
    if any(token in content for token in ["步骤", "流程", "方法"]):
        return "procedure"
    if any(token in content for token in ["今天", "明天", "昨天", "会议", "发生"]):
        return "event"
    return "fact"


def _normalize_recall_query(text: str) -> str:
    """把自然语言回忆问题压缩成适合记忆检索的关键词。

    参数:
        text: 用户原始问题。

    返回:
        去掉固定问法后的检索短语。
    """
    query = text
    for token in ["你记得什么", "回忆一下", "你记得", "记得我", "吗", "？", "?", "什么"]:
        query = query.replace(token, " ")
    return query.strip()


def _extract_profile_updates(text: str) -> dict[str, object]:
    """从用户文本中提取可直接写入用户画像的轻量字段。

    参数:
        text: 用户原始文本。

    返回:
        包含 name、preferences、dislikes 或 reply_style 的更新字典。
    """
    updates: dict[str, object] = {}
    nickname_match = re.search(r"我(?:叫|是)\s*([\u4e00-\u9fffA-Za-z0-9_-]{1,32})", text)
    if nickname_match:
        updates["name"] = nickname_match.group(1)
    preference_match = re.search(r"我喜欢\s*(.+)", text)
    if preference_match:
        updates["preferences"] = [preference_match.group(1).strip("。.!！ ")]
    dislike_match = re.search(r"我不喜欢\s*(.+)", text)
    if dislike_match:
        updates["dislikes"] = [dislike_match.group(1).strip("。.!！ ")]
    style_match = re.search(r"回答(?:风格|方式).*?(简洁|详细|直接|温柔|正式|口语化)", text)
    if style_match:
        updates["reply_style"] = style_match.group(1)
    return updates


def _high_confidence_inferred_memories(text: str) -> list[tuple[str, str, list[str], float]]:
    """从用户文本中抽取高置信度的隐式长期记忆。

    参数:
        text: 用户原始文本。

    返回:
        记忆正文、类型、标签和重要度组成的列表。
    """
    if MEMORIZE_RE.match(text):
        return []
    candidates: list[tuple[str, str, list[str], float]] = []
    for pattern, memory_type, tags, importance in [
        (r"我喜欢\s*(.+)", "preference", ["preference", "inferred"], 0.62),
        (r"我(?:叫|是)\s*([\u4e00-\u9fffA-Za-z0-9_-]{1,32})", "fact", ["profile", "inferred"], 0.68),
        (r"我住在\s*(.+)", "fact", ["profile", "location", "inferred"], 0.68),
        (r"我的习惯是\s*(.+)", "procedure", ["habit", "inferred"], 0.64),
    ]:
        match = re.search(pattern, text)
        if not match:
            continue
        content = match.group(0).strip("。.!！ ")
        if content:
            candidates.append((content, memory_type, tags, importance))
    return candidates[:3]


def _candidate_memories(text: str) -> list[tuple[str, str, list[str], float, float]]:
    """从用户文本中抽取需要后续证据确认的候选记忆。

    参数:
        text: 用户原始文本。

    返回:
        记忆正文、类型、标签、重要度和置信度组成的列表。
    """
    if MEMORIZE_RE.match(text):
        return []
    candidates: list[tuple[str, str, list[str], float, float]] = []
    for pattern, memory_type, tags in [
        (r"我不喜欢\s*(.+)", "preference", ["dislike", "candidate"]),
        (r"回答(?:风格|方式).*?(简洁|详细|直接|温柔|正式|口语化)", "preference", ["reply-style", "candidate"]),
    ]:
        match = re.search(pattern, text)
        if not match:
            continue
        content = match.group(0).strip("。.!！ ")
        if content:
            candidates.append((content, memory_type, tags, 0.56, 0.5))
    return candidates[:3]


def _parse_correction(text: str) -> tuple[str, str] | None:
    """解析“不是旧说法，新说法是……”这类自然语言记忆纠错。

    参数:
        text: 用户原始文本。

    返回:
        成功时返回旧记忆查询词和新记忆正文；无法解析时返回 None。
    """
    normalized = text.strip()
    if "不是" not in normalized:
        return None
    pieces = re.split(r"[，,。；;]", normalized)
    old_query = ""
    new_content = ""
    for piece in pieces:
        chunk = piece.strip()
        if not chunk:
            continue
        if "不是" in chunk and not old_query:
            old_query = chunk.split("不是", 1)[1].strip("：: ")
        if any(token in chunk for token in ["我喜欢", "我叫", "我住在", "我的习惯是"]) and not new_content:
            new_content = chunk
    if old_query and new_content:
        return old_query, new_content
    return None


def _correction_similarity(content: str, query: str) -> float:
    """计算候选旧记忆和纠错查询之间的简单词项相似度。

    参数:
        content: 候选旧记忆正文。
        query: 用户纠错中提到的旧内容线索。

    返回:
        0 到 1 之间的匹配分数。
    """
    content_terms = set(re_split_query(content.lower()))
    query_terms = set(re_split_query(query.lower()))
    if not content_terms or not query_terms:
        return 0.0
    overlap = len(content_terms & query_terms) / len(query_terms)
    if query.lower() in content.lower():
        overlap = max(overlap, 0.8)
    return overlap


def _memory_hit_summary(item: dict[str, Any]) -> dict[str, Any]:
    """把完整记忆命中压缩成 trace 中使用的摘要字段。

    参数:
        item: MemoryRetriever 返回的记忆命中字典。

    返回:
        只包含 id、分数、命中原因、来源和类型的轻量字典。
    """
    return {
        "id": int(item["id"]),
        "score": round(float(item.get("_match_score", 0.0)), 4),
        "reason": str(item.get("_match_reason") or "keyword_only"),
        "source_kind": str(item.get("source_kind") or "inferred"),
        "type": str(item.get("type") or "fact"),
    }


def _log_background_task_error(task: asyncio.Task) -> None:
    """记录后台记忆任务异常，避免 asyncio task 错误被吞掉。

    参数:
        task: 已完成的后台任务。
    """
    try:
        task.result()
    except asyncio.CancelledError:
        return
    except Exception:
        logger.exception("Background memory update failed")
