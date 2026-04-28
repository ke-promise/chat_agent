"""基于 Webhook OpenAPI 流程的 QQ 官方 Bot 通道。"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from chat_agent.config import QQBotConfig
from chat_agent.mcp.registry import MCPRegistry
from chat_agent.memory.store import SQLiteStore
from chat_agent.messages import Attachment, InboundMessage, OutboundAttachment, OutboundMessage
from chat_agent.skills import SkillsLoader

logger = logging.getLogger(__name__)

MessageHandlerFunc = Callable[[InboundMessage], Awaitable[OutboundMessage]]

QQ_DISPATCH = 0
QQ_HEARTBEAT = 1
QQ_HEARTBEAT_ACK = 7
QQ_HTTP_CALLBACK_ACK = 8


def is_allowed_qq_user(config: QQBotConfig, sender: str | None) -> bool:
    """根据 QQ openid 白名单判断当前发送者是否允许访问。"""
    allow_from = config.allow_from or []
    if not allow_from:
        return True
    return bool(sender and sender in allow_from)


class QQBotAPI:
    """QQ 官方 Bot REST 接口的轻量异步封装。"""

    def __init__(self, config: QQBotConfig) -> None:
        """保存 QQ API 配置，并延迟创建 HTTP 会话与 access token。"""
        self.config = config
        self._session: Any | None = None
        self._access_token = ""
        self._expires_at = 0.0

    async def start(self) -> None:
        """创建 aiohttp ClientSession，供后续 token 获取和消息发送复用。"""
        import aiohttp

        timeout = aiohttp.ClientTimeout(total=30)
        self._session = aiohttp.ClientSession(timeout=timeout)

    async def close(self) -> None:
        """关闭 HTTP 会话，释放连接资源。"""
        if self._session:
            await self._session.close()
            self._session = None

    @property
    def api_base_url(self) -> str:
        """返回当前应使用的 QQ OpenAPI 根地址，支持沙箱和自定义覆盖。"""
        if self.config.api_base_url:
            return self.config.api_base_url.rstrip("/")
        return "https://sandbox.api.sgroup.qq.com" if self.config.sandbox else "https://api.sgroup.qq.com"

    async def _token(self) -> str:
        """获取并缓存 QQ access token，在过期前自动复用。"""
        if self._access_token and time.time() < self._expires_at - 30:
            return self._access_token
        if self._session is None:
            await self.start()
        assert self._session is not None
        payload = {"appId": self.config.app_id, "clientSecret": self.config.app_secret}
        async with self._session.post(self.config.token_url, json=payload) as response:
            body = await response.text()
            if response.status >= 300:
                raise RuntimeError(f"QQ token request failed status={response.status} body={body[:300]}")
            data = json.loads(body)
        if int(data.get("code", 0) or 0) != 0:
            raise RuntimeError(f"QQ token request failed code={data.get('code')} message={data.get('message')}")
        self._access_token = str(data.get("access_token") or "")
        expires_in = int(data.get("expires_in") or 0)
        if not self._access_token:
            raise RuntimeError("QQ token response missing access_token")
        self._expires_at = time.time() + max(expires_in, 60)
        return self._access_token

    async def request(self, method: str, path: str, body: dict[str, Any]) -> dict[str, Any]:
        """向 QQ OpenAPI 发送一条 JSON 请求，并统一处理认证头与错误响应。"""
        if self._session is None:
            await self.start()
        assert self._session is not None
        token = await self._token()
        url = f"{self.api_base_url}{path}"
        headers = {"Authorization": f"QQBot {token}", "X-Union-Appid": self.config.app_id}
        async with self._session.request(method, url, json=body, headers=headers) as response:
            text = await response.text()
            if response.status >= 300:
                trace_id = response.headers.get("X-Tps-trace-ID", "")
                raise RuntimeError(f"QQ API failed status={response.status} trace_id={trace_id} body={text[:500]}")
            return json.loads(text) if text.strip() else {}

    async def send_text(self, chat_id: str, content: str, reply_to: str | None = None, event_id: str | None = None) -> None:
        """按 chat_id 场景发送 QQ 文本消息，可选择回复指定消息或事件。"""
        scene, target = split_qq_chat_id(chat_id)
        body: dict[str, Any] = {
            "content": content,
            "msg_type": 0,
            "msg_seq": _msg_seq(),
        }
        if reply_to:
            body["msg_id"] = reply_to
        if event_id:
            body["event_id"] = event_id

        if scene == "group":
            await self.request("POST", f"/v2/groups/{target}/messages", body)
        elif scene == "c2c":
            await self.request("POST", f"/v2/users/{target}/messages", body)
        elif scene == "dm":
            await self.request("POST", f"/dms/{target}/messages", body)
        else:
            await self.request("POST", f"/channels/{target}/messages", body)

    async def send_image_url(
        self,
        chat_id: str,
        url: str,
        caption: str = "",
        reply_to: str | None = None,
        event_id: str | None = None,
    ) -> None:
        """通过 URL 向 QQ 发送图片，group/c2c 与频道/私信使用不同接口。"""
        scene, target = split_qq_chat_id(chat_id)
        if scene in {"group", "c2c"}:
            body: dict[str, Any] = {
                "file_type": 1,
                "url": url,
                "srv_send_msg": True,
                "content": caption,
                "msg_seq": _msg_seq(),
            }
            if reply_to:
                body["msg_id"] = reply_to
            if event_id:
                body["event_id"] = event_id
            path = f"/v2/groups/{target}/files" if scene == "group" else f"/v2/users/{target}/files"
            await self.request("POST", path, body)
            return

        body = {"content": caption, "image": url, "msg_type": 0, "msg_seq": _msg_seq()}
        if reply_to:
            body["msg_id"] = reply_to
        if event_id:
            body["event_id"] = event_id
        path = f"/dms/{target}/messages" if scene == "dm" else f"/channels/{target}/messages"
        await self.request("POST", path, body)

    async def send_image_file(
        self,
        chat_id: str,
        file_path: str | Path,
        caption: str = "",
        reply_to: str | None = None,
        event_id: str | None = None,
    ) -> None:
        """把本地图片文件上传到 QQ group/c2c 会话并发送。"""
        scene, target = split_qq_chat_id(chat_id)
        if scene not in {"group", "c2c"}:
            raise RuntimeError(f"QQ local image upload is only supported for group/c2c chats, got {scene}")
        path_obj = Path(file_path)
        if not path_obj.exists() or not path_obj.is_file():
            raise RuntimeError(f"QQ local image not found: {path_obj}")

        body: dict[str, Any] = {
            "file_type": 1,
            "file_data": base64.b64encode(path_obj.read_bytes()).decode("ascii"),
            "srv_send_msg": True,
            "content": caption,
            "msg_seq": _msg_seq(),
        }
        if reply_to:
            body["msg_id"] = reply_to
        if event_id:
            body["event_id"] = event_id
        request_path = f"/v2/groups/{target}/files" if scene == "group" else f"/v2/users/{target}/files"
        await self.request("POST", request_path, body)


class QQBotChannel:
    """接收 QQ 官方 Bot Webhook，并通过 QQ OpenAPI 发送回复。"""

    name = "qq"

    def __init__(
        self,
        config: QQBotConfig,
        handler: MessageHandlerFunc,
        store: SQLiteStore | None = None,
        mcp_registry: MCPRegistry | None = None,
        skills_loader: SkillsLoader | None = None,
    ) -> None:
        """初始化 QQ Webhook 通道，保存业务处理器和可选状态组件。"""
        self.config = config
        self.handler = handler
        self.store = store
        self.mcp_registry = mcp_registry
        self.skills_loader = skills_loader
        self.api = QQBotAPI(config)
        self._stopped = asyncio.Event()
        self._runner: Any | None = None
        self._site: Any | None = None

    async def start(self) -> None:
        """启动本地 aiohttp Webhook 服务并注册 QQ 回调路由。"""
        try:
            from aiohttp import web
        except ModuleNotFoundError as exc:
            raise RuntimeError("QQ channel requires aiohttp. Install dependencies with pip install -e .") from exc

        await self.api.start()
        app = web.Application()
        app.router.add_post(_normalize_path(self.config.path), self._handle_webhook)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.config.host, self.config.port)
        await self._site.start()
        logger.info("QQ bot webhook started host=%s port=%s path=%s", self.config.host, self.config.port, self.config.path)

    async def idle(self) -> None:
        """阻塞当前协程，直到 stop() 请求停止通道。"""
        await self._stopped.wait()

    async def stop(self) -> None:
        """停止 Webhook 服务并关闭底层 QQ API 会话。"""
        self._stopped.set()
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        await self.api.close()

    async def send(self, message: OutboundMessage) -> None:
        """把统一出站消息转换成 QQ 文本或附件消息并发送。"""
        reply_to = message.reply_to_message_id
        event_id = str(message.metadata.get("qq_event_id") or "") or None
        try:
            if message.attachments:
                await self._send_with_attachments(message, reply_to=reply_to, event_id=event_id)
            elif message.content.strip():
                await self.api.send_text(message.chat_id, message.content[: self.config.max_text_chars], reply_to=reply_to, event_id=event_id)
        except Exception:
            logger.exception("Failed to send QQ message chat_id=%s", message.chat_id)

    async def _send_with_attachments(self, message: OutboundMessage, reply_to: str | None, event_id: str | None) -> None:
        """发送包含附件的 QQ 消息，先发文本，再依次发送图片附件。"""
        remaining_text = message.content.strip()
        if remaining_text:
            await self.api.send_text(message.chat_id, remaining_text[: self.config.max_text_chars], reply_to=reply_to, event_id=event_id)
            remaining_text = ""
            reply_to = None
            event_id = None
        sent_media = False
        for attachment in message.attachments:
            image_url = attachment.url or attachment.file_id
            if attachment.kind == "photo" and image_url and image_url.startswith(("http://", "https://")):
                await self.api.send_image_url(
                    message.chat_id,
                    image_url,
                    caption="",
                    reply_to=reply_to if not sent_media else None,
                    event_id=event_id if not sent_media else None,
                )
                sent_media = True
                continue
            if attachment.local_path:
                await self.api.send_image_file(
                    message.chat_id,
                    attachment.local_path,
                    caption="",
                    reply_to=reply_to if not sent_media else None,
                    event_id=event_id if not sent_media else None,
                )
                sent_media = True
                continue

    async def _handle_webhook(self, request: Any) -> Any:
        """处理 QQ Webhook HTTP 请求，完成签名校验、握手、心跳和事件分发。"""
        from aiohttp import web

        raw = await request.read()
        if self.config.verify_signature and not verify_qq_signature(self.config.app_secret, request.headers, raw):
            logger.warning("Rejected QQ webhook with invalid signature")
            return web.Response(status=401, text="invalid signature")

        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return web.Response(status=400, text="invalid json")

        op = _as_int(payload.get("op"), -1)
        data = payload.get("d") if isinstance(payload.get("d"), dict) else {}
        if data and "plain_token" in data and "event_ts" in data:
            signature = sign_qq_payload(self.config.app_secret, str(data["event_ts"]), str(data["plain_token"]).encode("utf-8"))
            return web.json_response({"plain_token": str(data["plain_token"]), "signature": signature})

        if op == QQ_HEARTBEAT:
            seq = _as_int(payload.get("s", payload.get("d")), 0)
            return web.json_response({"op": QQ_HEARTBEAT_ACK, "d": seq})

        if op == QQ_DISPATCH:
            task = asyncio.create_task(self._process_event(payload), name="qq-webhook-event")
            task.add_done_callback(_log_background_task_error)
            return web.json_response({"op": QQ_HTTP_CALLBACK_ACK, "d": 0})

        return web.json_response({})

    async def _process_event(self, payload: dict[str, Any]) -> None:
        """把 QQ dispatch 事件转换成入站消息，执行命令或交给 AgentLoop。"""
        inbound = await self._build_inbound(payload)
        if inbound is None:
            return
        if not is_allowed_qq_user(self.config, inbound.sender):
            logger.warning("Rejected unauthorized QQ sender=%s chat_id=%s", inbound.sender, inbound.chat_id)
            if self.config.unauthorized_reply:
                await self.api.send_text(inbound.chat_id, "这只小助手暂时还不认识你，先不能放你进来哦。", reply_to=inbound.message_id)
            return
        if await self._handle_command(inbound):
            return
        try:
            outbound = await self.handler(inbound)
        except Exception:
            logger.exception("QQ message handling failed")
            outbound = OutboundMessage(channel="qq", chat_id=inbound.chat_id, content="刚刚处理这条消息时绊了一下，我缓缓再陪你试。")
        if outbound.reply_to_message_id is None:
            outbound.reply_to_message_id = inbound.message_id
        outbound.metadata = {**outbound.metadata, "qq_event_id": inbound.metadata.get("event_id", "")}
        await self.send(outbound)

    async def _build_inbound(self, payload: dict[str, Any]) -> InboundMessage | None:
        """从 QQ 事件载荷中提取会话、发送者、文本和附件，构造统一入站消息。"""
        event_type = str(payload.get("t") or "")
        data = payload.get("d")
        if not isinstance(data, dict):
            return None

        scene = "channel"
        target_id = str(data.get("channel_id") or "")
        if event_type == "C2C_MESSAGE_CREATE":
            scene = "c2c"
            target_id = _first_text(data, ("user_openid", "openid", "user_id")) or _first_text(data.get("author"), ("id", "user_openid", "openid"))
        elif event_type == "GROUP_AT_MESSAGE_CREATE":
            scene = "group"
            target_id = _first_text(data, ("group_openid", "group_id"))
        elif event_type == "DIRECT_MESSAGE_CREATE":
            scene = "dm"
            target_id = _first_text(data, ("guild_id", "channel_id"))
        elif not target_id:
            logger.debug("QQ event ignored without target event_type=%s data=%s", event_type, data)
            return None

        sender = _first_text(data.get("author"), ("id", "user_openid", "member_openid", "openid")) or _first_text(
            data, ("op_member_openid", "member_openid", "user_openid", "openid", "user_id")
        )
        content = clean_qq_content(str(data.get("content") or ""))
        attachments = await self._build_attachments(data)
        if attachments and not content:
            content = "请描述这张图片。"
        inbound = InboundMessage(
            channel="qq",
            chat_id=f"{scene}:{target_id}",
            sender=sender or "unknown",
            content=content,
            attachments=attachments,
            message_id=str(data.get("id") or ""),
            metadata={
                "scene": scene,
                "event_type": event_type,
                "event_id": str(data.get("event_id") or payload.get("id") or ""),
                "raw": payload,
            },
        )
        logger.info("Received QQ message event=%s chat_id=%s sender=%s text=%r", event_type, inbound.chat_id, inbound.sender, inbound.content)
        return inbound

    async def _build_attachments(self, data: dict[str, Any]) -> list[Attachment]:
        """从 QQ 消息数据中提取图片附件，并按配置决定是否下载到本地。"""
        items = data.get("attachments")
        if not isinstance(items, list):
            return []
        attachments: list[Attachment] = []
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "")
            content_type = str(item.get("content_type") or item.get("mime_type") or "")
            filename = str(item.get("filename") or item.get("file_name") or "")
            if not _looks_like_image(content_type, filename, url):
                continue
            local_path = await self._download_attachment(url, filename or f"qq-{index}.jpg") if self.config.download_images and url else None
            attachments.append(
                Attachment(
                    kind="image",
                    file_id=str(item.get("id") or url or filename),
                    mime_type=content_type or "image/jpeg",
                    local_path=local_path,
                    url=url or None,
                    size=_as_int(item.get("size"), 0) or None,
                )
            )
        return attachments

    async def _download_attachment(self, url: str, filename: str) -> str | None:
        """下载 QQ 图片附件到 workspace/attachments，并执行大小限制。"""
        if self.api._session is None:
            await self.api.start()
        assert self.api._session is not None
        target_dir = Path("workspace") / "attachments"
        target_dir.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^0-9A-Za-z._-]+", "_", filename)[:80] or f"qq-{int(time.time())}.jpg"
        target = target_dir / safe_name
        try:
            async with self.api._session.get(url) as response:
                response.raise_for_status()
                max_bytes = self.config.image_max_mb * 1024 * 1024
                data = await response.content.read(max_bytes + 1)
                if len(data) > max_bytes:
                    logger.warning("QQ image too large url=%s", url)
                    return None
                target.write_bytes(data)
                return str(target.resolve())
        except Exception:
            logger.exception("Failed to download QQ attachment url=%s", url)
            return None

    async def _handle_command(self, message: InboundMessage) -> bool:
        """处理 QQ 文本命令；命中命令时直接回复并返回 True。"""
        text = message.content.strip()
        if not text.startswith("/"):
            return False
        command, _, arg = text.partition(" ")
        command = command.lower()
        chat_id = message.chat_id
        if command in {"/start", "/help"}:
            await self.api.send_text(chat_id, "你好呀，我在这儿。文字、图片、提醒、记忆都可以交给我，轻轻喊一声就到。", reply_to=message.message_id)
            return True
        if command == "/status":
            memory_count = await self.store.count_memories(chat_id) if self.store else 0
            reminder_count = await self.store.count_pending_reminders(chat_id) if self.store else 0
            skills_count = len(self.skills_loader.list_skills(filter_unavailable=False)) if self.skills_loader else 0
            text = f"我在好好运行中。\nchat_id: {chat_id}\n记忆数量: {memory_count}\n待提醒: {reminder_count}\nskills: {skills_count}"
            await self.api.send_text(chat_id, text, reply_to=message.message_id)
            return True
        if command == "/memory":
            memories = await self.store.list_recent_memories(chat_id, limit=10) if self.store else []
            text = "当前还没有长期记忆，我的小本本这页还是空的。" if not memories else "\n".join(f"#{item['id']} [{item['type']}] {item['content']}" for item in memories)
            await self.api.send_text(chat_id, text, reply_to=message.message_id)
            return True
        if command == "/forget":
            try:
                memory_id = int(arg.strip())
            except ValueError:
                await self.api.send_text(chat_id, "用法是 /forget <memory_id>，把编号给我就能轻轻擦掉那条记忆。", reply_to=message.message_id)
                return True
            ok = await self.store.delete_memory(chat_id, memory_id) if self.store else False
            await self.api.send_text(chat_id, f"好呀，已轻轻删掉记忆 #{memory_id}。" if ok else "我翻了翻小本本，没有找到这条记忆。", reply_to=message.message_id)
            return True
        if command == "/mcp":
            await self.api.send_text(chat_id, self.mcp_registry.status() if self.mcp_registry else "MCP 还没启用。", reply_to=message.message_id)
            return True
        if command == "/mcp_reload":
            if not self.mcp_registry:
                await self.api.send_text(chat_id, "MCP 还没启用，暂时没法重载它。", reply_to=message.message_id)
            else:
                await self.mcp_registry.reload()
                await self.api.send_text(chat_id, "好啦，MCP 已重新加载。\n" + self.mcp_registry.status(), reply_to=message.message_id)
            return True
        if command == "/skills":
            if not self.skills_loader:
                await self.api.send_text(chat_id, "skills 还没启用，这个小抽屉暂时打不开。", reply_to=message.message_id)
            else:
                skills = self.skills_loader.list_skills(filter_unavailable=False)
                text = "当前还没有 skill，小工具箱暂时空空的。" if not skills else "\n".join(
                    f"- {item['name']} [{item['source']}]: {item['description']}" for item in skills[:20]
                )
                await self.api.send_text(chat_id, text, reply_to=message.message_id)
            return True
        if command == "/proactive_status":
            tick = await self.store.get_last_proactive_tick() if self.store else None
            text = "暂时还没有 proactive tick，我的小巡逻还没留下记录。" if not tick else f"最近 tick: {tick['tick_at']}\naction={tick['action']} reason={tick['skip_reason']}"
            await self.api.send_text(chat_id, text, reply_to=message.message_id)
            return True
        return False


def split_qq_chat_id(chat_id: str) -> tuple[str, str]:
    """把内部 chat_id 拆成 QQ 场景和目标 id。"""
    if ":" not in chat_id:
        return "channel", chat_id
    scene, target = chat_id.split(":", 1)
    scene = scene.strip().lower() or "channel"
    if scene not in {"channel", "group", "c2c", "dm"}:
        scene = "channel"
    return scene, target.strip()


def clean_qq_content(content: str) -> str:
    """清理 QQ 消息中的表情标记和 at 机器人片段。"""
    content = re.sub(r"<faceType=[^>]*>", "", content)
    content = re.sub(r"<@!?\d+>", "", content)
    content = re.sub(r"<@!?[0-9A-Za-z_-]+>", "", content)
    return content.strip()


def verify_qq_signature(secret: str, headers: Any, body: bytes) -> bool:
    """校验 QQ Webhook 请求的 Ed25519 签名。"""
    signature = headers.get("X-Signature-Ed25519", "")
    timestamp = headers.get("X-Signature-Timestamp", "")
    if not signature or not timestamp:
        return False
    try:
        from nacl.signing import SigningKey

        signing_key = SigningKey(_secret_seed(secret))
        verify_key = signing_key.verify_key
        verify_key.verify(timestamp.encode("utf-8") + body, binascii.unhexlify(signature))
        return True
    except ModuleNotFoundError as exc:
        raise RuntimeError("QQ webhook signature verification requires PyNaCl. Install dependencies with pip install -e .") from exc
    except Exception:
        return False


def sign_qq_payload(secret: str, timestamp: str, body: bytes) -> str:
    """使用 app_secret 生成 QQ 回调握手需要的 Ed25519 签名。"""
    try:
        from nacl.signing import SigningKey
    except ModuleNotFoundError as exc:
        raise RuntimeError("QQ webhook validation requires PyNaCl. Install dependencies with pip install -e .") from exc
    signing_key = SigningKey(_secret_seed(secret))
    signed = signing_key.sign(timestamp.encode("utf-8") + body)
    return signed.signature.hex()


def _secret_seed(secret: str) -> bytes:
    """把 QQ app_secret 扩展或裁剪成 PyNaCl SigningKey 需要的 32 字节种子。"""
    raw = secret.encode("utf-8")
    if not raw:
        raise ValueError("secret is empty")
    while len(raw) < 32:
        raw += raw
    return raw[:32]


def _normalize_path(path: str) -> str:
    """规范化 Webhook path，保证以斜杠开头。"""
    path = path.strip() or "/qqbot"
    return path if path.startswith("/") else f"/{path}"


def _msg_seq() -> int:
    """生成 QQ 消息发送需要的递增近似序号。"""
    return int(time.time() * 1000) % 2_147_483_647


def _as_int(value: Any, default: int) -> int:
    """把任意值安全转换为 int，失败时返回默认值。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _first_text(data: Any, keys: tuple[str, ...]) -> str:
    """按优先级从字典中取出第一个非空文本字段。"""
    if not isinstance(data, dict):
        return ""
    for key in keys:
        value = data.get(key)
        if value is not None and str(value):
            return str(value)
    return ""


def _looks_like_image(content_type: str, filename: str, url: str) -> bool:
    """根据 MIME、文件名和 URL 粗略判断附件是否是图片。"""
    lowered = " ".join([content_type, filename, url]).lower()
    return content_type.startswith("image/") or any(lowered.endswith(ext) or ext in lowered for ext in (".jpg", ".jpeg", ".png", ".webp", ".gif"))


def _log_background_task_error(task: asyncio.Task) -> None:
    """记录 QQ 后台任务异常，避免 create_task 的错误静默丢失。"""
    try:
        task.result()
    except asyncio.CancelledError:
        return
    except Exception:
        logger.exception("QQ background task failed")
