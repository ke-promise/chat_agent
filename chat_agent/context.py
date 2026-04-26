"""LLM 上下文构建器。"""

from __future__ import annotations

import base64
import logging
import mimetypes
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from chat_agent.agent.provider import make_multimodal_user_content
from chat_agent.memory.files import MemoryFiles
from chat_agent.memory.retriever import MemoryRetriever
from chat_agent.memory.store import SQLiteStore
from chat_agent.messages import Attachment, InboundMessage
from chat_agent.skills import SkillsLoader
from chat_agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ContextBundle:
    """上下文构建结果。

    Attributes:
        messages: 最终传给 LLM 的消息列表，已包含 system prompt、历史消息和当前输入。
        memory_hits: 记忆检索命中结果，便于后续 trace、调试和审计。
        trace: 本次上下文构建的统计信息，例如历史条数、字符数和是否注入摘要。
    """
    messages: list[dict[str, Any]]
    memory_hits: list[dict[str, Any]]
    trace: dict[str, Any]


class ContextBuilder:
    """负责把数据库、工具、skills 和当前消息组装成 LLM 上下文。"""

    def __init__(
        self,
        store: SQLiteStore,
        retriever: MemoryRetriever,
        tools: ToolRegistry,
        history_window: int = 20,
        memory_top_k: int = 5,
        summary_enabled: bool = True,
        max_prompt_chars: int = 12000,
        vision_enabled: bool = False,
        skills_loader: SkillsLoader | None = None,
        inject_skills_catalog: bool = True,
        memory_files: MemoryFiles | None = None,
    ) -> None:
        """初始化上下文构建器。

        参数:
            store: 会话历史、摘要和用户画像的持久化存储。
            retriever: 长期记忆检索器，用于根据当前消息召回相关记忆。
            tools: 当前轮允许模型看到的工具注册表。
            history_window: 构建 prompt 时最多带入多少条近期历史消息。
            memory_top_k: 最多注入多少条长期记忆命中。
            summary_enabled: 是否把会话摘要加入系统上下文。
            max_prompt_chars: prompt 的近似字符预算，超出后会裁剪历史。
            vision_enabled: 是否允许把图片附件转成多模态输入。
            skills_loader: 可选的 skills 装载器，用于按触发词注入技能说明。
            inject_skills_catalog: 是否把技能目录摘要注入系统上下文。
            memory_files: 旧版记忆导出对象，当前仅为兼容保留。
        """
        self.store = store
        self.retriever = retriever
        self.tools = tools
        self.history_window = history_window
        self.memory_top_k = memory_top_k
        self.summary_enabled = summary_enabled
        self.max_prompt_chars = max_prompt_chars
        self.vision_enabled = vision_enabled
        self.skills_loader = skills_loader
        self.inject_skills_catalog = inject_skills_catalog
        self.memory_files = memory_files  # kept for backward compatibility; no longer used in online prompt

    async def build(self, inbound: InboundMessage) -> ContextBundle:
        """根据当前入站消息组装一份完整的 LLM 上下文。

        参数:
            inbound: 当前用户输入，包含文本、附件、发送者和元数据。

        返回:
            返回 `ContextBundle`，其中包含发送给模型的消息、命中的长期记忆以及构建 trace。
        """
        history = await self.store.get_recent_session_messages(inbound.chat_id, limit=self.history_window)
        memory_hits = await self.retriever.retrieve(inbound.chat_id, inbound.content, self.memory_top_k)
        summary = await self.store.get_summary(inbound.chat_id) if self.summary_enabled else None
        user_profile = await self.store.get_user_profile(inbound.chat_id)

        identity = (
            "你是一个长期运行在 Telegram 中的陪伴型个人智能体，像可靠又有点俏皮的贴身小伙伴。"
            "你要温柔、亲近、可爱、简洁、可靠地帮助用户。可以自然使用轻量语气词和少量可爱表达，"
            "例如“好呀”“收到啦”“我来看看”“交给我”，但不要油腻、撒娇过度或影响信息密度。"
            "你可以使用 OpenAI-compatible tool calling，也可以理解简化文本工具标签。"
            "最终回复必须是普通文本，不要把工具调用标签发给用户。"
        )
        behavior = (
            "行为规则：保护隐私；不要泄漏 token 或 API key；工具失败时简要说明；"
            "如果图片模型不可用，不要假装看到了图片。"
            "回答应像陪伴型助手：先接住用户，再给清楚答案；可以有一点俏皮和温度，但事实、代码、配置和风险提示必须准确。"
            "不要频繁使用感叹号；不要每句都卖萌；用户明显严肃、排错或赶时间时，语气收敛，优先解决问题。"
            "当用户询问今天、最新、实时、新闻、天气、价格、网页信息等可能变化的内容时，"
            "必须优先调用可见的搜索/MCP 工具获取信息，不要直接说没有联网能力。"
            "如果页面抓取工具返回 403 或超时，但搜索工具已经返回标题、摘要或链接，"
            "就基于这些搜索结果作答并说明来源有限，不要放弃回答。"
        )
        now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        tool_hint = self.tools.list_descriptions(only_visible=True)
        attachment_summary = self._attachment_summary(inbound.attachments)
        skills_catalog = ""
        active_skills = ""
        active_skill_names: list[str] = []
        if self.skills_loader:
            active_skill_names = self.skills_loader.get_always_skills()
            active_skill_names.extend(self.skills_loader.extract_triggered_skill_names(inbound.content))
            active_skill_names = sorted(set(active_skill_names))
            if self.inject_skills_catalog:
                skills_catalog = self.skills_loader.build_skills_summary()
            active_skills = self.skills_loader.load_skills_for_context(active_skill_names)

        context_lines = [f"当前时间：{now}"]
        if summary:
            context_lines.append(f"近期上下文摘要：{summary}")
        if user_profile:
            context_lines.append("用户画像：\n" + _format_user_profile(user_profile))
        if memory_hits:
            memory_text = "\n".join(
                f"- #{item['id']} [{item['type']}] {item['content']} "
                f"(命中: {item['_match_reason']}, 分数: {float(item['_match_score']):.2f})"
                for item in memory_hits
            )
            context_lines.append(f"可参考的长期记忆：\n{memory_text}")
        if attachment_summary:
            context_lines.append(f"附件摘要：{attachment_summary}")
        context_lines.append(f"当前默认可见工具（可用 tool_search 发现更多工具）：\n{tool_hint}")
        if skills_catalog:
            context_lines.append(
                "Skills catalog：下面是可用技能说明书摘要。需要完整说明时，可用 read_skill 工具读取；"
                "本轮显式触发或 always 技能会在后面完整注入。\n"
                f"{skills_catalog}"
            )
        if active_skills:
            context_lines.append(f"Active skills：\n{active_skills}")

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": identity},
            {"role": "system", "content": behavior + "\n\n" + "\n\n".join(context_lines)},
        ]
        for item in history:
            if item["role"] in {"user", "assistant"}:
                messages.append({"role": item["role"], "content": item["content"]})

        image_urls = self._image_urls(inbound.attachments) if self.vision_enabled else []
        messages.append({"role": "user", "content": make_multimodal_user_content(inbound.content, image_urls)})

        messages = self._trim_messages(messages)
        identity_chars = len(identity) + len(behavior)
        history_chars = sum(len(str(item.get("content", ""))) for item in history)
        memory_chars = sum(len(str(item.get("content", ""))) for item in memory_hits)
        total_chars = sum(len(str(item.get("content", ""))) for item in messages)
        retriever_trace = getattr(self.retriever, "last_trace", {}) or {}
        trace = {
            "history_count": len(history),
            "memory_count": len(memory_hits),
            "profile_included": bool(user_profile),
            "total_chars": total_chars,
            "summary_included": bool(summary),
            "attachments_count": len(inbound.attachments),
            "hyde_used": bool(retriever_trace.get("hyde_used")),
            "skills_catalog_chars": len(skills_catalog),
            "active_skills": active_skill_names,
            "active_skills_chars": len(active_skills),
            "memory_file_chars": 0,
            "candidates_considered": int(retriever_trace.get("candidates_considered", 0)),
        }
        logger.info(
            "Prompt breakdown chat_id=%s identity_chars=%s history_chars=%s memory_chars=%s tools=%s attachments=%s skills_catalog=%s active_skills=%s total_chars=%s",
            inbound.chat_id,
            identity_chars,
            history_chars,
            memory_chars,
            self.tools.visible_count(),
            len(inbound.attachments),
            len(skills_catalog),
            len(active_skills),
            total_chars,
        )
        return ContextBundle(messages=messages, memory_hits=memory_hits, trace=trace)

    def _trim_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """在超过字符预算时裁剪历史消息。

        参数:
            messages: 已按 system/history/current-user 顺序排列的消息列表。

        返回:
            返回裁剪后的消息列表，优先保留 system 指令和最新输入。
        """
        result = list(messages)
        while len(result) > 3 and _message_chars(result) > self.max_prompt_chars:
            del result[2]
        return result

    def _attachment_summary(self, attachments: list[Attachment]) -> str:
        """把附件列表压缩成简短文字摘要。

        参数:
            attachments: 当前消息携带的附件列表。

        返回:
            返回适合写入系统提示词的附件摘要；没有附件时返回空字符串。
        """
        if not attachments:
            return ""
        parts: list[str] = []
        for attachment in attachments:
            label = attachment.kind
            if attachment.mime_type:
                label += f"({attachment.mime_type})"
            parts.append(label)
        return "、".join(parts)

    def _image_urls(self, attachments: list[Attachment]) -> list[str]:
        """提取可供视觉模型读取的图片地址。

        参数:
            attachments: 当前消息携带的附件列表。

        返回:
            返回图片 `url` 或本地图片转换出的 `data:` URL 列表。
        """
        urls: list[str] = []
        for attachment in attachments:
            if attachment.kind != "image":
                continue
            if attachment.local_path:
                data_url = _local_image_to_data_url(Path(attachment.local_path), attachment.mime_type)
                if data_url:
                    urls.append(data_url)
                    continue
            if attachment.url:
                urls.append(attachment.url)
        return urls


def _message_chars(messages: list[dict[str, Any]]) -> int:
    """统计消息列表中 `content` 字段的大致字符数。"""
    return sum(len(str(item.get("content", ""))) for item in messages)


def _local_image_to_data_url(path: Path, mime_type: str | None = None) -> str | None:
    """把本地图片文件编码为 `data:` URL。

    参数:
        path: 本地图片路径。
        mime_type: 可选 MIME 类型；为空时会根据文件名猜测。

    返回:
        成功时返回 data URL，读取失败时返回 `None`。
    """
    try:
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        logger.warning("Failed to read local image for multimodal context: %s", path)
        return None
    guessed = mime_type or mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{guessed};base64,{payload}"


def _format_user_profile(profile: dict[str, Any]) -> str:
    """把用户画像字典格式化为多行文本。

    参数:
        profile: 存储层中的用户画像对象。

    返回:
        返回适合直接注入 prompt 的多行 Markdown 样式文本。
    """
    lines: list[str] = []
    for key, value in profile.items():
        if isinstance(value, list):
            joined = "、".join(str(item) for item in value if str(item).strip())
            if joined:
                lines.append(f"- {key}: {joined}")
        elif isinstance(value, dict):
            pairs = "，".join(f"{inner_key}={inner_value}" for inner_key, inner_value in value.items())
            if pairs:
                lines.append(f"- {key}: {pairs}")
        elif str(value).strip():
            lines.append(f"- {key}: {value}")
    return "\n".join(lines)
