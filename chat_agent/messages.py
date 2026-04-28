"""Agent 内部统一消息模型。

Telegram Update、QQ Webhook 事件和后台主动任务都会先在 channel 或调度层转换成这些 dataclass。
AgentLoop、ContextBuilder、Reasoner 和 ToolRegistry 只依赖统一消息结构，
这样后续扩展附件、trace、工具上下文时不会把具体通道 SDK 或原始回调载荷泄漏到业务层。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


@dataclass(slots=True)
class Attachment:
    """用户消息中的附件描述。

    这个对象是 channel 层和 agent 层之间传递附件的统一格式，避免业务逻辑直接依赖
    Telegram PhotoSize、QQ 附件字典等原始通道对象。

    字段:
        kind: 附件类型。当前只区分 image 和 file；图片会进入多模态上下文，
            file 先作为未来扩展保留。
        file_id: 通道侧文件标识或 URL 兜底标识，用于日志、排查和后续重新下载。
        mime_type: 文件 MIME 类型，例如 image/jpeg。模型构造 data URL 时会用到。
        local_path: 如果 channel 已把文件下载到本地，这里保存绝对路径或工作区路径。
        url: 如果 channel 能提供可访问 URL，这里保存 URL。优先级低于 local_path。
        size: 文件大小，单位字节。用于限制过大图片和记录 trace。
    """

    kind: Literal["image", "file"]
    file_id: str
    mime_type: str | None = None
    local_path: str | None = None
    url: str | None = None
    size: int | None = None


@dataclass(slots=True)
class OutboundAttachment:
    """出站媒体附件描述。

    这是被动回复和主动推送共享的统一媒体结构。Telegram、QQ 等 channel 会根据自身能力
    解释 kind、local_path、file_id 和 url，例如发送图片、复用已上传文件或降级为文本链接。

    字段:
        kind: 出站媒体类型，目前支持 `photo` 和 `sticker`。
        local_path: 本地文件路径。优先级最高，适合工作区内的表情包图片。
        file_id: 通道侧已有的文件标识；支持复用时可避免重复上传。
        url: 可直接发送或下载后转发的远程 URL。
        mime_type: 可选 MIME 类型，仅用于日志和后续扩展。
    """

    kind: Literal["photo", "sticker"]
    local_path: str | None = None
    file_id: str | None = None
    url: str | None = None
    mime_type: str | None = None


@dataclass(slots=True)
class InboundMessage:
    """进入 AgentLoop 的统一入站消息模型。

    任意通道消息进入业务层之前都要转换成这个结构。这样 AgentLoop、ContextBuilder、
    Reasoner、MemoryStore 都不需要知道 Telegram SDK、QQ Webhook 或后台调度的细节。

    字段:
        channel: 来源通道名，例如 "telegram"、"qq" 或后台任务使用的 "proactive"。
        chat_id: 会话标识。Telegram 使用 chat.id 字符串；QQ 会按场景使用
            group:<openid>、c2c:<openid> 等形式，用于隔离历史、记忆、提醒和主动推送目标。
        sender: 发送者标识。Telegram 通常是 user.id，QQ 通常是用户 openid 或成员 openid。
        content: 用户文本内容。图片无 caption 时会填入默认描述请求。
        attachments: 附件列表。图片理解、附件摘要和历史保存都依赖这里。
        message_id: 原始消息 ID。用于回复引用或排查问题，可以为空。
        created_at: 消息创建时间。为空时自动填入当前 UTC 时间。
        metadata: channel 私有补充信息，例如 username、caption、QQ event_id、scene、原始载荷。
            业务层可以读取，但不应依赖某个 channel 的复杂对象。
    """

    channel: str
    chat_id: str
    sender: str
    content: str
    attachments: list[Attachment] = field(default_factory=list)
    message_id: str | None = None
    created_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """补齐默认创建时间。

        dataclass 初始化完成后执行。统一使用 timezone-aware UTC 时间，避免 SQLite 中
        写入 naive datetime 后在跨时区逻辑里出错。
        """
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)

    @property
    def text(self) -> str:
        """兼容旧代码的文本别名。"""
        return self.content

    @property
    def user_id(self) -> str:
        """兼容旧代码的用户 ID 别名。"""
        return self.sender

    @property
    def username(self) -> str | None:
        """从 metadata 中读取 username。

        返回:
            如果 channel 填入了 username，则返回字符串；否则返回 None。当前主要由 Telegram 填入，
            QQ 通常没有这个字段。
        """
        value = self.metadata.get("username")
        return str(value) if value is not None else None


@dataclass(slots=True)
class OutboundMessage:
    """AgentLoop 输出给 channel 层的统一出站消息模型。

    channel 层只负责把这个对象发送出去，不再关心 LLM、记忆、提醒等业务流程。

    字段:
        channel: 目标通道名，例如 "telegram" 或 "qq"。
        chat_id: 目标会话 ID。
        content: 要发送的文本内容。纯文本消息直接发送；媒体消息则作为 caption 或后续补充文本。
        attachments: 可选出站媒体列表，用于 Telegram 图片/贴纸、QQ 图片等媒体回复。
        reply_to_message_id: 可选的被回复消息 ID，例如 Telegram reply_to_message_id 或 QQ msg_id。
        metadata: 预留给通道扩展，例如 parse_mode、QQ event_id、静默发送、按钮等。
    """

    channel: str
    chat_id: str
    content: str = ""
    attachments: list[OutboundAttachment] = field(default_factory=list)
    reply_to_message_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """兼容旧代码的文本别名。"""
        return self.content

    @property
    def has_attachments(self) -> bool:
        """判断本条出站消息是否包含媒体附件。"""
        return bool(self.attachments)
