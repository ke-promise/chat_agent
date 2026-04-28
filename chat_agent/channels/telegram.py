"""Telegram-only 通道实现。

本模块是项目唯一直接接触 Telegram Bot API 的地方：负责 long polling、命令处理、
白名单鉴权、图片下载、把 Telegram Update 转换为 InboundMessage，以及把 OutboundMessage 发回 Telegram。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path

from telegram import Update
from telegram.constants import ChatAction
from telegram.error import NetworkError, TimedOut
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from chat_agent.config import TelegramConfig
from chat_agent.mcp.registry import MCPRegistry
from chat_agent.memory.store import SQLiteStore
from chat_agent.messages import Attachment, InboundMessage, OutboundAttachment, OutboundMessage
from chat_agent.skills import SkillsLoader

logger = logging.getLogger(__name__)


MessageHandlerFunc = Callable[[InboundMessage], Awaitable[OutboundMessage]]


def is_allowed_username(config: TelegramConfig, username: str | None) -> bool:
    """判断 Telegram 用户名是否在允许列表中。

    参数:
        config: Telegram 相关配置，主要读取其中的 allow_from 白名单。
        username: Telegram 传入的用户名，可能为空；函数内部会兼容带不带 @ 的写法。

    返回:
        True 表示允许继续处理本次消息；False 表示应拒绝或忽略该用户。

    说明:
        allow_from 为空时代表不启用白名单，任何用户都可以访问。这个函数只做纯判断，
        不发送消息、不写业务日志，方便单元测试单独覆盖权限逻辑。
    """
    if not config.allow_from:
        return True
    return username is not None and username.lstrip("@") in config.allow_from


class TelegramChannel:
    """Telegram 收发通道。

    这个类是项目与 Telegram Bot API 的唯一直接集成点，职责刻意保持很窄：
    - 启动和停止 python-telegram-bot Application。
    - 把 Telegram Update 转换成项目内部统一的 InboundMessage。
    - 把 AgentLoop 返回的 OutboundMessage 发送回 Telegram。
    - 处理 Telegram 命令、用户白名单、图片下载和基础状态查询。

    注意:
        这里不直接调用 LLM、不拼 prompt、不操作复杂业务流程。真正的对话编排由
        传入的 handler 负责，通常是 AgentLoop.handle_message。
    """

    def __init__(
        self,
        config: TelegramConfig,
        handler: MessageHandlerFunc,
        store: SQLiteStore | None = None,
        mcp_registry: MCPRegistry | None = None,
        skills_loader: SkillsLoader | None = None,
    ) -> None:
        """创建 Telegram 通道实例并注册命令/消息处理器。

        参数:
            config: Telegram 配置，包含 token、allow_from、图片下载开关和图片大小限制。
            handler: 业务处理入口。通道收到普通消息后会构造 InboundMessage 并 await 它。
            store: SQLiteStore，可选；用于 /status、/memory、/forget 等命令读取状态。
            mcp_registry: MCPRegistry，可选；用于 /mcp、/mcp_reload 展示或重载 MCP server。
            skills_loader: SkillsLoader，可选；用于 /skills 展示当前可用技能。

        说明:
            构造函数只完成本地对象初始化，不访问 Telegram 网络。真正联网发生在 start()。
        """
        self.config = config
        self.handler = handler
        self.store = store
        self.mcp_registry = mcp_registry
        self.skills_loader = skills_loader
        self.application = (
            Application.builder()
            .token(config.token)
            .connect_timeout(30)
            .read_timeout(30)
            .write_timeout(30)
            .pool_timeout(30)
            .build()
        )
        self.application.add_handler(CommandHandler("start", self._on_start))
        self.application.add_handler(CommandHandler("help", self._on_help))
        self.application.add_handler(CommandHandler("status", self._on_status))
        self.application.add_handler(CommandHandler("memory", self._on_memory))
        self.application.add_handler(CommandHandler("forget", self._on_forget))
        self.application.add_handler(CommandHandler("mcp", self._on_mcp))
        self.application.add_handler(CommandHandler("mcp_reload", self._on_mcp_reload))
        self.application.add_handler(CommandHandler("skills", self._on_skills))
        self.application.add_handler(CommandHandler("proactive_status", self._on_proactive_status))
        self.application.add_handler(MessageHandler((filters.TEXT | filters.PHOTO) & ~filters.COMMAND, self._on_message))
        self._stopped = asyncio.Event()
        self._initialized = False

    async def start(self) -> None:
        """启动 Telegram Bot 并开始 long polling。

        行为:
            - 初始化 Application。
            - 启动 application 和 updater polling。
            - 对常见网络错误做有限次数重试。

        异常:
            最终仍启动失败时会重新抛出异常，由 main.py 统一收尾。这样可以避免半启动状态
            残留后台任务或 MCP 子进程。
        """
        max_attempts = 5
        for attempt in range(1, max_attempts + 1):
            try:
                await self.application.initialize()
                self._initialized = True
                await self.application.start()
                if self.application.updater is None:
                    raise RuntimeError("Telegram application updater is unavailable")
                await self.application.updater.start_polling()
                logger.info("Telegram bot started")
                return
            except (TimedOut, NetworkError) as exc:
                await self._cleanup_failed_start()
                if attempt >= max_attempts:
                    logger.error("Failed to start Telegram bot after %s attempts: %s", attempt, exc)
                    raise
                delay = min(30, 2**attempt)
                logger.warning(
                    "Telegram start failed attempt=%s/%s network_error=%s; retrying in %s seconds",
                    attempt,
                    max_attempts,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
            except Exception:
                await self._cleanup_failed_start()
                logger.exception("Failed to start Telegram bot")
                raise
        logger.info("Telegram bot started")

    async def _cleanup_failed_start(self) -> None:
        """清理启动失败后可能残留的 Telegram Application 状态。

        start() 在 initialize/start/start_polling 任一阶段失败时都会调用该函数。函数会尽力
        停止 updater、application，并在需要时 shutdown；每一步都单独捕获异常，避免清理阶段
        的错误覆盖原始启动错误。
        """
        try:
            if self.application.updater and self.application.updater.running:
                await self.application.updater.stop()
        except Exception:
            logger.debug("Failed to cleanup Telegram updater after start failure", exc_info=True)
        try:
            if self.application.running:
                await self.application.stop()
        except Exception:
            logger.debug("Failed to cleanup Telegram application after start failure", exc_info=True)
        try:
            if self._initialized:
                await self.application.shutdown()
                self._initialized = False
        except Exception:
            logger.debug("Failed to shutdown Telegram application after start failure", exc_info=True)

    async def idle(self) -> None:
        """阻塞当前协程直到 stop() 被调用。

        main.py 使用它让进程保持运行。相比 busy wait，这里直接等待 asyncio.Event。
        """
        await self._stopped.wait()

    async def stop(self) -> None:
        """停止 Telegram 通道并释放 Application 资源。

        该方法可以重复调用；内部会检查 updater/application 当前状态。关闭顺序与
        python-telegram-bot 推荐流程一致：先停 polling，再停 application，最后 shutdown。
        """
        self._stopped.set()
        try:
            if self.application.updater and self.application.updater.running:
                await self.application.updater.stop()
        except Exception:
            logger.exception("Failed to stop Telegram updater")
        try:
            if self.application.running:
                await self.application.stop()
        except Exception:
            logger.exception("Failed to stop Telegram application")
        try:
            if self._initialized:
                await self.application.shutdown()
                self._initialized = False
        except Exception:
            logger.exception("Failed to shutdown Telegram application")

    async def send(self, message: OutboundMessage) -> None:
        """发送一条内部 OutboundMessage 到 Telegram。

        参数:
            message: 统一出站消息对象。channel 当前应为 telegram，chat_id 必须是 Telegram
                chat id 字符串，content 是要发送的正文。

        说明:
            Telegram 网络错误不会继续向外抛出，因为主动提醒和普通回复都不应因一次发送失败
            直接击穿主循环。失败细节会写入日志。
        """
        try:
            if message.attachments:
                await self._send_with_attachments(message)
            elif message.content.strip():
                await self.application.bot.send_message(
                    chat_id=message.chat_id,
                    text=message.content,
                    reply_to_message_id=int(message.reply_to_message_id) if message.reply_to_message_id else None,
                )
        except (TimedOut, NetworkError) as exc:
            logger.warning("Failed to send Telegram message chat_id=%s network_error=%s", message.chat_id, exc)
        except Exception:
            logger.exception("Failed to send Telegram message chat_id=%s", message.chat_id)

    async def _send_with_attachments(self, message: OutboundMessage) -> None:
        """按 Telegram 可接受的方式发送带附件的出站消息。

        第一版策略：
        - `photo` 使用 `send_photo`，并把正文放在第一张图片的 caption 中。
        - `sticker` 使用 `send_sticker`；若正文非空，则贴纸发完后再补一条普通文本。
        - 多个附件时按顺序逐个发送，正文只附着在第一张可带 caption 的媒体上。
        """
        remaining_text = message.content.strip()
        sent_caption = False
        reply_to = int(message.reply_to_message_id) if message.reply_to_message_id else None

        for index, attachment in enumerate(message.attachments):
            caption = None
            if attachment.kind == "photo" and remaining_text and not sent_caption:
                caption = remaining_text[:1024]
                sent_caption = True
                remaining_text = ""
            await self._send_attachment(
                chat_id=message.chat_id,
                attachment=attachment,
                reply_to_message_id=reply_to if index == 0 else None,
                caption=caption,
            )

        if remaining_text:
            await self.application.bot.send_message(
                chat_id=message.chat_id,
                text=remaining_text,
                reply_to_message_id=reply_to,
            )

    async def _send_attachment(
        self,
        chat_id: str,
        attachment: OutboundAttachment,
        reply_to_message_id: int | None = None,
        caption: str | None = None,
    ) -> None:
        """发送单个媒体附件。

        参数:
            chat_id: Telegram 目标会话 ID。
            attachment: 要发送的媒体附件描述。
            reply_to_message_id: 可选回复消息 ID。
            caption: 仅 `photo` 类型使用的 caption。
        """
        local_path = Path(attachment.local_path) if attachment.local_path else None
        if attachment.kind == "photo":
            if local_path:
                with local_path.open("rb") as file_obj:
                    await self.application.bot.send_photo(
                        chat_id=chat_id,
                        photo=file_obj,
                        caption=caption,
                        reply_to_message_id=reply_to_message_id,
                    )
            else:
                photo = attachment.file_id or attachment.url
                if not photo:
                    raise ValueError("Outbound photo attachment requires local_path, file_id or url")
                await self.application.bot.send_photo(
                    chat_id=chat_id,
                    photo=photo,
                    caption=caption,
                    reply_to_message_id=reply_to_message_id,
                )
            return

        if attachment.kind == "sticker":
            if local_path:
                with local_path.open("rb") as file_obj:
                    await self.application.bot.send_sticker(
                        chat_id=chat_id,
                        sticker=file_obj,
                        reply_to_message_id=reply_to_message_id,
                    )
            else:
                sticker = attachment.file_id or attachment.url
                if not sticker:
                    raise ValueError("Outbound sticker attachment requires local_path, file_id or url")
                await self.application.bot.send_sticker(
                    chat_id=chat_id,
                    sticker=sticker,
                    reply_to_message_id=reply_to_message_id,
                )
            return

        raise ValueError(f"Unsupported outbound attachment kind: {attachment.kind}")

    async def _on_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """处理 /start 命令。

        参数:
            update: python-telegram-bot 提供的原始更新对象。
            context: 当前 Bot 上下文，包含 bot 实例和命令参数等。

        说明:
            命令处理也会经过白名单校验，避免未授权用户通过命令探测 bot 状态。
        """
        if not await self._ensure_allowed(update, context):
            return
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="你好呀，我在这儿。文字、图片、提醒、记忆都可以交给我，轻轻喊一声就到。",
        )

    async def _on_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """处理 /help 命令，返回当前 MVP 支持的常用用法。"""
        if not await self._ensure_allowed(update, context):
            return
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=(
                "我会这些小把戏，想用哪个轻轻喊我就行：/start /help /status /memory /forget <id> /mcp /mcp_reload /skills /proactive_status\n"
                "记忆：记住：我喜欢简洁的回答\n"
                "回忆：你记得我喜欢什么？\n"
                "提醒：1分钟后提醒我喝水"
            ),
        )

    async def _on_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """处理 /status 命令，展示当前 chat 的运行状态。

        返回内容包括 chat_id、记忆数量、待提醒数量、最近 proactive tick、MCP 状态和
        skills 数量，方便用户把 chat_id 填到 proactive.loop.target_chat_id。
        """
        if not await self._ensure_allowed(update, context):
            return
        chat_id = str(update.effective_chat.id)
        memory_count = await self.store.count_memories(chat_id) if self.store else 0
        reminder_count = await self.store.count_pending_reminders(chat_id) if self.store else 0
        tick = await self.store.get_last_proactive_tick() if self.store else None
        tick_text = "暂时还没有" if not tick else f"{tick['tick_at']} action={tick['action']} reason={tick['skip_reason']}"
        mcp_text = self.mcp_registry.status() if self.mcp_registry else "MCP 还没启用。"
        skills_count = len(self.skills_loader.list_skills(filter_unavailable=False)) if self.skills_loader else 0
        await context.bot.send_message(
            chat_id=chat_id,
            text=(
                f"我在好好运行中。\nchat_id: {chat_id}\n"
                f"记忆数量: {memory_count}\n待提醒: {reminder_count}\n"
                f"最近 proactive tick: {tick_text}\nMCP: {mcp_text}\nskills: {skills_count}"
            ),
        )

    async def _on_memory(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """处理 /memory 命令，列出当前 chat 最近的长期记忆。"""
        if not await self._ensure_allowed(update, context):
            return
        chat_id = str(update.effective_chat.id)
        memories = await self.store.list_recent_memories(chat_id, limit=10) if self.store else []
        text = "当前还没有长期记忆，我的小本本这页还是空的。" if not memories else "\n".join(
            f"#{item['id']} [{item['type']}] {item['content']}" for item in memories
        )
        await context.bot.send_message(chat_id=chat_id, text=text)

    async def _on_forget(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """处理 /forget <id> 命令，删除当前 chat 下指定的长期记忆。"""
        if not await self._ensure_allowed(update, context):
            return
        chat_id = str(update.effective_chat.id)
        if not context.args:
            await context.bot.send_message(chat_id=chat_id, text="用法是 /forget <memory_id>，把编号给我就能轻轻擦掉那条记忆。")
            return
        try:
            memory_id = int(context.args[0])
        except ValueError:
            await context.bot.send_message(chat_id=chat_id, text="memory_id 要是数字哦，我才知道该擦掉哪一条。")
            return
        ok = await self.store.delete_memory(chat_id, memory_id) if self.store else False
        await context.bot.send_message(chat_id=chat_id, text=f"好呀，已轻轻删掉记忆 #{memory_id}。" if ok else "我翻了翻小本本，没有找到这条记忆。")

    async def _on_mcp(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """处理 /mcp 命令，展示已连接 MCP server 和工具数量。"""
        if not await self._ensure_allowed(update, context):
            return
        text = self.mcp_registry.status() if self.mcp_registry else "MCP 还没启用。"
        await context.bot.send_message(chat_id=update.effective_chat.id, text=text)

    async def _on_mcp_reload(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """处理 /mcp_reload 命令，重新读取 MCP 配置并重连 server。"""
        if not await self._ensure_allowed(update, context):
            return
        if not self.mcp_registry:
            await context.bot.send_message(chat_id=update.effective_chat.id, text="MCP 还没启用，暂时没法重载它。")
            return
        await self.mcp_registry.reload()
        await context.bot.send_message(chat_id=update.effective_chat.id, text="好啦，MCP 已重新加载。\n" + self.mcp_registry.status())

    async def _on_skills(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """处理 /skills 命令，列出内置与 workspace 中可见的技能说明书。"""
        if not await self._ensure_allowed(update, context):
            return
        if not self.skills_loader:
            await context.bot.send_message(chat_id=update.effective_chat.id, text="skills 还没启用，这个小抽屉暂时打不开。")
            return
        skills = self.skills_loader.list_skills(filter_unavailable=False)
        if not skills:
            await context.bot.send_message(chat_id=update.effective_chat.id, text="当前还没有 skill，小工具箱暂时空空的。")
            return
        lines = []
        for item in skills[:30]:
            state = "可用" if item["available"] else "不可用"
            missing = ",".join(item.get("missing_bins", []) + item.get("missing_env", []))
            suffix = f" 缺失：{missing}" if missing else ""
            lines.append(f"- {item['name']} [{item['source']}, {state}]{suffix}: {item['description']}")
        await context.bot.send_message(chat_id=update.effective_chat.id, text="\n".join(lines))

    async def _on_proactive_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """处理 /proactive_status 命令，展示最近一次主动循环和已见 feed 数量。"""
        if not await self._ensure_allowed(update, context):
            return
        tick = await self.store.get_last_proactive_tick() if self.store else None
        seen = await self.store.count_seen_items() if self.store else 0
        text = "暂时还没有 proactive tick，我的小巡逻还没留下记录。" if not tick else (
            f"最近 tick: {tick['tick_at']}\n"
            f"action={tick['action']} reason={tick['skip_reason']}\n"
            f"reminders_due={tick['reminders_due']} content_count={tick['content_count']} sent_count={tick['sent_count']}\n"
            f"seen_items={seen}"
        )
        await context.bot.send_message(chat_id=update.effective_chat.id, text=text)

    async def _on_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """处理 Telegram 文本或图片消息。

        参数:
            update: Telegram 原始更新，可能包含 text、caption、photo 等字段。
            context: Telegram handler 上下文，用于下载文件、发送 typing 状态等。

        流程:
            1. 校验 update 是否完整以及用户是否有权限。
            2. 将图片转换成 Attachment 列表，并提取正文或 caption。
            3. 构造 InboundMessage 交给业务 handler。
            4. 将 OutboundMessage 发送回 Telegram。

        注意:
            原始 update 只放进 metadata 供排查使用，业务层不应依赖 Telegram 专有对象。
        """
        if update.effective_chat is None or update.effective_user is None or update.message is None:
            return
        if not await self._ensure_allowed(update, context):
            return

        user = update.effective_user
        attachments = await self._build_attachments(update, context)
        content = update.message.text or update.message.caption or ""
        if update.message.photo and not content:
            content = "请描述这张图片。"

        inbound = InboundMessage(
            channel="telegram",
            chat_id=str(update.effective_chat.id),
            sender=str(user.id),
            content=content,
            attachments=attachments,
            message_id=str(update.message.message_id),
            metadata={"username": user.username, "caption": update.message.caption, "raw": update.to_dict()},
        )
        logger.info(
            "Received Telegram message chat_id=%s username=%s text=%r attachments=%s",
            inbound.chat_id,
            inbound.username,
            inbound.content,
            len(inbound.attachments),
        )

        if update.message.photo and not attachments:
            await context.bot.send_message(chat_id=update.effective_chat.id, text="这张图有点难拿到，可能太大或下载失败了。你换一张小一点的试试？")
            return

        try:
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
            outbound = await self.handler(inbound)
        except Exception:
            logger.exception("Telegram message handling failed")
            outbound = OutboundMessage(channel="telegram", chat_id=inbound.chat_id, content="刚刚处理这条消息时绊了一下，我缓缓再陪你试。")
        await self.send(outbound)

    async def _build_attachments(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> list[Attachment]:
        """从 Telegram 图片消息构造内部 Attachment。

        参数:
            update: 原始 Telegram Update，函数只读取 update.message.photo。
            context: Telegram 上下文，用于调用 get_file 和 download_to_drive。

        返回:
            Attachment 列表。当前只支持图片，因此最多返回一个 image attachment；如果没有图片、
            图片超出大小限制或下载失败，则返回空列表。
        """
        if update.message is None or not update.message.photo:
            return []
        photo = update.message.photo[-1]
        max_bytes = self.config.image_max_mb * 1024 * 1024
        if photo.file_size and photo.file_size > max_bytes:
            logger.warning("Telegram image too large size=%s max=%s", photo.file_size, max_bytes)
            return []
        try:
            tg_file = await context.bot.get_file(photo.file_id)
            url = tg_file.file_path if tg_file.file_path and str(tg_file.file_path).startswith("http") else None
            local_path = None
            if self.config.download_images:
                target_dir = Path("workspace") / "attachments"
                target_dir.mkdir(parents=True, exist_ok=True)
                target = target_dir / f"{photo.file_unique_id or photo.file_id}.jpg"
                await tg_file.download_to_drive(custom_path=target)
                local_path = str(target.resolve())
            return [
                Attachment(
                    kind="image",
                    file_id=photo.file_id,
                    mime_type="image/jpeg",
                    local_path=local_path,
                    url=url,
                    size=photo.file_size,
                )
            ]
        except Exception:
            logger.exception("Failed to download Telegram image")
            return []

    async def _ensure_allowed(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
        """统一执行 Telegram 用户白名单校验。

        参数:
            update: 原始 Telegram Update，用于读取 effective_user 和 effective_chat。
            context: Telegram handler 上下文，用于在配置允许时发送无权限提示。

        返回:
            True 表示后续 handler 可以继续；False 表示本次请求已经被拒绝。
        """
        if update.effective_user is None or update.effective_chat is None:
            return False
        username = update.effective_user.username
        if is_allowed_username(self.config, username):
            return True
        logger.warning("Rejected unauthorized Telegram user username=%s user_id=%s", username, update.effective_user.id)
        if self.config.unauthorized_reply:
            try:
                await context.bot.send_message(chat_id=update.effective_chat.id, text="这只小助手暂时还不认识你，先不能放你进来哦。")
            except Exception:
                logger.exception("Failed to send unauthorized reply")
        return False
