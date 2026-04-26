"""Builtin tool registration."""

from __future__ import annotations

import asyncio
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from chat_agent.memes import MemeCatalog
from chat_agent.memory.store import SQLiteStore
from chat_agent.messages import OutboundAttachment, OutboundMessage
from chat_agent.scheduler import parse_after_reminder
from chat_agent.skills import SkillsLoader, is_valid_skill_name
from chat_agent.tools.registry import Tool, ToolContext, ToolRegistry
from chat_agent.url_safety import URLSafetyError, ensure_public_http_url


def build_default_registry(
    store: SQLiteStore,
    fetch_timeout: int = 10,
    tool_search_enabled: bool = True,
    file_workspace: Path | str = Path("workspace/files"),
    skills_loader: SkillsLoader | None = None,
    extra_model_tools: list[str] | None = None,
) -> ToolRegistry:
    """构建`default`、`registry`。

    参数:
        store: 参与构建`default`、`registry`的 `store` 参数。
        fetch_timeout: 参与构建`default`、`registry`的 `fetch_timeout` 参数。
        tool_search_enabled: 参与构建`default`、`registry`的 `tool_search_enabled` 参数。
        file_workspace: 参与构建`default`、`registry`的 `file_workspace` 参数。
        skills_loader: 参与构建`default`、`registry`的 `skills_loader` 参数。
        extra_model_tools: 参与构建`default`、`registry`的 `extra_model_tools` 参数。

    返回:
        返回与本函数处理结果对应的数据。
    """
    registry = ToolRegistry(store=store, extra_model_tools=extra_model_tools)
    file_root = Path(file_workspace).resolve()
    file_root.mkdir(parents=True, exist_ok=True)

    async def memorize(context: ToolContext, args: dict[str, Any]) -> str:
        """处理相关逻辑。

        参数:
            context: 参与处理相关逻辑的 `context` 参数。
            args: 参与处理相关逻辑的 `args` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        content = str(args.get("content", "")).strip()
        tags = args.get("tags", [])
        memory_type = str(args.get("type", "fact")).strip() or "fact"
        importance = float(args.get("importance", 0.6))
        if not content:
            return "memorize 需要 content 参数。"
        memory_id = await store.add_memory(
            context.message.chat_id,
            content,
            tags=tags,
            memory_type=memory_type,
            importance=importance,
            source_chat_id=context.message.chat_id,
            source_kind="explicit",
            confidence=1.0,
        )
        return f"已保存长期记忆 #{memory_id}。"

    async def recall_memory(context: ToolContext, args: dict[str, Any]) -> str:
        """处理记忆。

        参数:
            context: 参与处理记忆的 `context` 参数。
            args: 参与处理记忆的 `args` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        query = str(args.get("query", "")).strip()
        limit = int(args.get("limit", 5))
        memories = await store.search_memories(context.message.chat_id, query, limit=limit)
        if not memories:
            return "没有找到相关长期记忆。"
        return "\n".join(f"- #{item['id']} {item['content']}" for item in memories)

    async def create_reminder(context: ToolContext, args: dict[str, Any]) -> str:
        """创建提醒。

        参数:
            context: 参与创建提醒的 `context` 参数。
            args: 参与创建提醒的 `args` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        content = str(args.get("content", "")).strip()
        delay_seconds = int(args.get("delay_seconds", 0) or 0)
        due_at_raw = str(args.get("due_at", "")).strip()
        natural_text = str(args.get("text", "")).strip()
        if natural_text and not content:
            parsed = parse_after_reminder(natural_text)
            if parsed:
                content = parsed.content
                due_at = parsed.due_at
            else:
                return "没有识别出提醒时间。"
        elif delay_seconds > 0:
            due_at = datetime.now(timezone.utc) + timedelta(seconds=delay_seconds)
        elif due_at_raw:
            try:
                due_at = datetime.fromisoformat(due_at_raw)
                if due_at.tzinfo is None:
                    due_at = due_at.replace(tzinfo=timezone.utc)
            except ValueError:
                return "due_at 需要 ISO 8601 时间格式。"
        else:
            return "create_reminder 需要 delay_seconds、due_at 或 text。"
        if not content:
            return "create_reminder 需要 content 参数。"
        reminder_id = await store.add_reminder(context.message.chat_id, context.message.sender, content, due_at)
        return f"已创建提醒 #{reminder_id}：{content}"

    async def list_reminders(context: ToolContext, args: dict[str, Any]) -> str:
        """列出`reminders`。

        参数:
            context: 参与列出`reminders`的 `context` 参数。
            args: 参与列出`reminders`的 `args` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        limit = int(args.get("limit", 10))
        reminders = await store.list_pending_reminders(context.message.chat_id, limit=limit)
        if not reminders:
            return "当前没有待发送提醒。"
        return "\n".join(
            f"- #{item['id']} {item['due_at'].astimezone().strftime('%Y-%m-%d %H:%M:%S')} {item['content']}"
            for item in reminders
        )

    async def cancel_reminder(context: ToolContext, args: dict[str, Any]) -> str:
        """处理提醒。

        参数:
            context: 参与处理提醒的 `context` 参数。
            args: 参与处理提醒的 `args` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        reminder_id = int(args.get("id", args.get("reminder_id", 0)) or 0)
        if reminder_id <= 0:
            return "cancel_reminder 需要 id 参数。"
        ok = await store.cancel_reminder(context.message.chat_id, reminder_id)
        return f"已取消提醒 #{reminder_id}。" if ok else f"没有找到可取消的提醒 #{reminder_id}。"

    async def web_fetch(_: ToolContext, args: dict[str, Any]) -> str:
        """处理抓取结果。

        参数:
            _: 参与处理抓取结果的 `_` 参数。
            args: 参与处理抓取结果的 `args` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        url = str(args.get("url", "")).strip()
        try:
            safe_url = ensure_public_http_url(url)
        except URLSafetyError as exc:
            return f"web_fetch URL 不安全或不合法：{exc}"

        def run() -> str:
            """执行相关逻辑。

            返回:
                返回与本函数处理结果对应的数据。"""
            req = urllib.request.Request(safe_url, headers={"User-Agent": "telegram-personal-agent/0.3"})
            with urllib.request.urlopen(req, timeout=fetch_timeout) as response:
                body = response.read(200_000)
                charset = response.headers.get_content_charset() or "utf-8"
                return body.decode(charset, errors="replace")

        try:
            content = await asyncio.to_thread(run)
        except Exception:
            return "获取网页失败，可能是网络、权限或页面格式问题。"
        return content[:8000]

    async def list_files(_: ToolContext, args: dict[str, Any]) -> str:
        """列出文件列表。

        参数:
            _: 参与列出文件列表的 `_` 参数。
            args: 参与列出文件列表的 `args` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        base = _safe_file_path(file_root, str(args.get("path", ".") or "."))
        if not base.exists():
            return "目录不存在。"
        if not base.is_dir():
            return f"{base.relative_to(file_root)} 不是目录。"
        limit = int(args.get("limit", 50) or 50)
        items = []
        for path in sorted(base.iterdir())[:limit]:
            suffix = "/" if path.is_dir() else ""
            items.append(f"- {_relative_display(path, file_root)}{suffix}")
        return "\n".join(items) if items else "目录为空。"

    async def read_file(_: ToolContext, args: dict[str, Any]) -> str:
        """读取文件。

        参数:
            _: 参与读取文件的 `_` 参数。
            args: 参与读取文件的 `args` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        path = _safe_file_path(file_root, str(args.get("path", "")).strip())
        if not path.exists() or not path.is_file():
            return "文件不存在。"
        max_chars = int(args.get("max_chars", 12000) or 12000)
        try:
            return path.read_text(encoding="utf-8", errors="replace")[:max_chars]
        except Exception:
            return "读取文件失败。"

    async def write_file(_: ToolContext, args: dict[str, Any]) -> str:
        """写入文件。

        参数:
            _: 参与写入文件的 `_` 参数。
            args: 参与写入文件的 `args` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        path = _safe_file_path(file_root, str(args.get("path", "")).strip())
        content = str(args.get("content", ""))
        append = _as_bool(args.get("append", False))
        if not path.name:
            return "write_file 需要 path 参数。"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if append:
                with path.open("a", encoding="utf-8") as file_obj:
                    file_obj.write(content)
            else:
                path.write_text(content, encoding="utf-8")
        except Exception:
            return "写入文件失败。"
        return f"已写入文件：{_relative_display(path, file_root)}"

    async def tool_search(_: ToolContext, args: dict[str, Any]) -> str:
        """处理`search`。

        参数:
            _: 参与处理`search`的 `_` 参数。
            args: 参与处理`search`的 `args` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        query = str(args.get("query", "")).strip()
        matches = registry.search(query, exposures={"discoverable"})
        if not matches:
            return "没有找到匹配工具。"
        return "本轮可发现工具：\n" + "\n".join(tool.description_line() for tool in matches)

    async def list_skills(_: ToolContext, args: dict[str, Any]) -> str:
        """列出技能集合。

        参数:
            _: 参与列出技能集合的 `_` 参数。
            args: 参与列出技能集合的 `args` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        if not skills_loader:
            return "skills 未启用。"
        include_unavailable = _as_bool(args.get("include_unavailable", False))
        skills = skills_loader.list_skills(filter_unavailable=not include_unavailable)
        if not skills:
            return "当前没有可用 skill。"
        lines = []
        for item in skills:
            state = "available" if item["available"] else "unavailable"
            missing = ",".join(item.get("missing_bins", []) + item.get("missing_env", []))
            suffix = f" missing={missing}" if missing else ""
            lines.append(f"- {item['name']} [{item['source']}, {state}]{suffix}: {item['description']}")
        return "\n".join(lines)

    async def read_skill(_: ToolContext, args: dict[str, Any]) -> str:
        """读取技能。

        参数:
            _: 参与读取技能的 `_` 参数。
            args: 参与读取技能的 `args` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        if not skills_loader:
            return "skills 未启用。"
        name = str(args.get("name", "")).strip()
        body = skills_loader.load_skill(name)
        return body if body else f"没有找到可用 skill：{name}"

    async def create_skill(_: ToolContext, args: dict[str, Any]) -> str:
        """创建技能。

        参数:
            _: 参与创建技能的 `_` 参数。
            args: 参与创建技能的 `args` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        if not skills_loader:
            return "skills 未启用。"
        name = str(args.get("name", "")).strip()
        description = str(args.get("description", "")).strip()
        body = str(args.get("body", "")).strip()
        always = _as_bool(args.get("always", False))
        if not is_valid_skill_name(name):
            return "skill name 只允许小写字母、数字和连字符。"
        if not description or not body:
            return "create_skill 需要 description 和 body。"
        path = skills_loader.write_workspace_skill(name, description, body, always=always)
        return f"已创建 workspace skill：{name} ({path})"

    async def update_skill(_: ToolContext, args: dict[str, Any]) -> str:
        """处理技能。

        参数:
            _: 参与处理技能的 `_` 参数。
            args: 参与处理技能的 `args` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        if not skills_loader:
            return "skills 未启用。"
        name = str(args.get("name", "")).strip()
        body = str(args.get("body", "")).strip()
        if not is_valid_skill_name(name):
            return "skill name 只允许小写字母、数字和连字符。"
        if not body:
            return "update_skill 需要 body。"
        try:
            path = skills_loader.update_workspace_skill(name, body)
        except FileNotFoundError:
            return f"只能更新 workspace 中已存在的 skill：{name}"
        return f"已更新 workspace skill：{name} ({path})"

    registry.register(
        Tool("memorize", "保存长期事实或偏好。", _schema({"content": "string", "tags": "array", "type": "string"}), memorize, exposure="always", risk="read")
    )
    registry.register(
        Tool("recall_memory", "检索长期记忆。", _schema({"query": "string", "limit": "integer"}), recall_memory, exposure="always", risk="read")
    )
    registry.register(
        Tool("create_reminder", "创建未来提醒。", _schema({"content": "string", "delay_seconds": "integer", "due_at": "string", "text": "string"}), create_reminder, exposure="always", risk="read")
    )
    registry.register(
        Tool("list_reminders", "列出未完成提醒。", _schema({"limit": "integer"}), list_reminders, exposure="always", risk="read")
    )
    registry.register(
        Tool("cancel_reminder", "取消未完成提醒。", _schema({"id": "integer"}), cancel_reminder, exposure="discoverable", risk="read")
    )
    registry.register(
        Tool("web_fetch", "按 URL 获取网页文本。", _schema({"url": "string"}), web_fetch, exposure="discoverable", risk="read")
    )
    registry.register(
        Tool("list_files", "列出 workspace 文件目录中的文件。", _schema({"path": "string", "limit": "integer"}), list_files, exposure="discoverable", risk="read")
    )
    registry.register(
        Tool("read_file", "读取 workspace 文件目录中的文本文件。", _schema({"path": "string", "max_chars": "integer"}), read_file, exposure="discoverable", risk="read")
    )
    registry.register(
        Tool("write_file", "写入 workspace 文件目录中的文本文件。", _schema({"path": "string", "content": "string", "append": "string"}), write_file, exposure="hidden", risk="write")
    )
    if tool_search_enabled:
        registry.register(
            Tool("tool_search", "搜索并发现更多可用工具。", _schema({"query": "string"}), tool_search, exposure="always", risk="read")
        )
    if skills_loader:
        registry.register(
            Tool("list_skills", "列出可用 skill 说明书。", _schema({"include_unavailable": "string"}), list_skills, exposure="always", risk="read")
        )
        registry.register(
            Tool("read_skill", "读取指定 SKILL.md 的完整内容。", _schema({"name": "string"}), read_skill, exposure="always", risk="read")
        )
        registry.register(
            Tool(
                "create_skill",
                "创建 workspace skill，不会修改内置 skill。",
                _schema({"name": "string", "description": "string", "body": "string", "always": "string"}),
                create_skill,
                exposure="hidden",
                risk="write",
            )
        )
        registry.register(
            Tool("update_skill", "更新 workspace 中已有 skill 的正文。", _schema({"name": "string", "body": "string"}), update_skill, exposure="hidden", risk="write")
        )
    return registry


def register_message_push_tool(
    registry: ToolRegistry,
    channel: Any,
    default_chat_id: str = "",
    file_workspace: Path | str = Path("workspace/files"),
) -> None:
    """注册消息、`push`、工具。

    参数:
        registry: 参与注册消息、`push`、工具的 `registry` 参数。
        channel: 参与注册消息、`push`、工具的 `channel` 参数。
        default_chat_id: 参与注册消息、`push`、工具的 `default_chat_id` 参数。
        file_workspace: 参与注册消息、`push`、工具的 `file_workspace` 参数。

    返回:
        返回与本函数处理结果对应的数据。
    """
    meme_catalog = MemeCatalog(file_workspace)

    def resolve_chat_id(context: ToolContext, raw_chat_id: Any) -> tuple[str, set[str]]:
        """解析`chat`、`id`。

        参数:
            context: 参与解析`chat`、`id`的 `context` 参数。
            raw_chat_id: 参与解析`chat`、`id`的 `raw_chat_id` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        chat_id = str(raw_chat_id or default_chat_id or context.message.chat_id).strip()
        allowed = {context.message.chat_id}
        if default_chat_id:
            allowed.add(default_chat_id)
        return chat_id, allowed

    async def send_message(context: ToolContext, args: dict[str, Any]) -> str:
        """发送消息。

        参数:
            context: 参与发送消息的 `context` 参数。
            args: 参与发送消息的 `args` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        content = str(args.get("content", "")).strip()
        emoji = str(args.get("emoji", "")).strip()
        chat_id, allowed = resolve_chat_id(context, args.get("chat_id", ""))
        if chat_id not in allowed:
            return "send_message 只能发送到当前 chat 或配置的 proactive target。"
        content = _compose_emoji_message(content, emoji)
        if not content:
            return "send_message 需要 content 或 emoji 参数。"
        await channel.send(OutboundMessage(channel="telegram", chat_id=chat_id, content=content[:3500]))
        return f"已发送 Telegram 消息到 chat_id={chat_id}。"

    async def send_emoji(context: ToolContext, args: dict[str, Any]) -> str:
        """发送`emoji`。

        参数:
            context: 参与发送`emoji`的 `context` 参数。
            args: 参与发送`emoji`的 `args` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        emoji = str(args.get("emoji", "")).strip()
        text = str(args.get("text", "")).strip()
        repeat = max(1, min(5, int(args.get("repeat", 1) or 1)))
        if emoji:
            emoji = emoji * repeat
        return await send_message(context, {"chat_id": args.get("chat_id", ""), "content": text, "emoji": emoji})

    async def list_memes(_: ToolContext, args: dict[str, Any]) -> str:
        """列出`memes`。

        参数:
            _: 参与列出`memes`的 `_` 参数。
            args: 参与列出`memes`的 `args` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        categories = meme_catalog.list_categories()
        if not categories:
            return "当前还没有可用表情包。请把图片放到 workspace/files/memes/，或准备 memes/manifest.json。"
        limit = max(1, min(20, int(args.get("limit", 12) or 12)))
        lines = []
        for item in categories[:limit]:
            aliases = f" aliases={','.join(item['aliases'])}" if item.get("aliases") else ""
            desc = f" {item['desc']}" if item.get("desc") else ""
            moods = f" moods={','.join(item['mood_tags'])}" if item.get("mood_tags") else ""
            sources = f" sources={','.join(item['source_allowlist'])}" if item.get("source_allowlist") else ""
            lines.append(f"- {item['name']} ({item['count']} 张){aliases}{moods}{sources}{desc}")
        return "\n".join(lines)

    async def send_meme(context: ToolContext, args: dict[str, Any]) -> str:
        """发送`meme`。

        参数:
            context: 参与发送`meme`的 `context` 参数。
            args: 参与发送`meme`的 `args` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        chat_id, allowed = resolve_chat_id(context, args.get("chat_id", ""))
        if chat_id not in allowed:
            return "send_meme 只能发送到当前 chat 或配置的 proactive target。"
        query = str(args.get("query", "")).strip()
        category = str(args.get("category", "")).strip()
        caption = _compose_emoji_message(str(args.get("caption", "")).strip(), str(args.get("emoji", "")).strip())
        match = meme_catalog.pick(query=query, category=category)
        if not match:
            return "没有找到合适的表情包。可以先调用 list_memes 看看当前有哪些分类。"
        await channel.send(
            OutboundMessage(
                channel="telegram",
                chat_id=chat_id,
                content=caption[:1024],
                attachments=[OutboundAttachment(kind=match.kind, local_path=str(match.path))],
                metadata={"meme_category": match.category},
            )
        )
        return f"已发送表情包：{match.category} / {match.path.name}"

    if not registry.get_tool("send_message"):
        registry.register(
            Tool(
                "send_message",
                "向当前 chat 或配置的 proactive target 主动发送一条 Telegram 消息。",
                _schema({"chat_id": "string", "content": "string", "emoji": "string"}),
                send_message,
                exposure="hidden",
                risk="side_effect",
            )
        )
    if not registry.get_tool("send_emoji"):
        registry.register(
            Tool(
                "send_emoji",
                "向当前 chat 或 proactive target 发送带 emoji 的消息，适合可爱提醒和轻量主动问候。",
                _schema({"chat_id": "string", "emoji": "string", "text": "string", "repeat": "integer"}),
                send_emoji,
                exposure="hidden",
                risk="side_effect",
            )
        )
    if not registry.get_tool("list_memes"):
        registry.register(
            Tool(
                "list_memes",
                "列出本地可用表情包分类和数量，帮助选择 send_meme 的 query 或 category。",
                _schema({"limit": "integer"}),
                list_memes,
                exposure="always",
                risk="read",
            )
        )
    if not registry.get_tool("send_meme"):
        registry.register(
            Tool(
                "send_meme",
                "从 workspace/files/memes/ 中挑选并发送一张表情包图片或贴纸，可按分类或关键词选择。",
                _schema({"chat_id": "string", "query": "string", "category": "string", "caption": "string", "emoji": "string"}),
                send_meme,
                exposure="discoverable",
                risk="side_effect",
            )
        )


def _schema(fields: dict[str, str]) -> dict[str, Any]:
    """处理相关逻辑。

    参数:
        fields: 参与处理相关逻辑的 `fields` 参数。

    返回:
        返回与本函数处理结果对应的数据。
    """
    type_map = {"string": "string", "integer": "integer", "array": "array", "number": "number"}
    return {"type": "object", "properties": {name: {"type": type_map.get(kind, "string")} for name, kind in fields.items()}}


def _safe_file_path(root: Path, relative_path: str) -> Path:
    """安全处理文件、路径。

    参数:
        root: 参与安全处理文件、路径的 `root` 参数。
        relative_path: 参与安全处理文件、路径的 `relative_path` 参数。

    返回:
        返回与本函数处理结果对应的数据。
    """
    if not relative_path:
        raise ValueError("path is required")
    raw = Path(relative_path)
    if raw.is_absolute():
        raise ValueError("absolute paths are not allowed")
    path = (root / raw).resolve()
    if root != path and root not in path.parents:
        raise ValueError("path escapes file workspace")
    return path


def _relative_display(path: Path, root: Path) -> str:
    """处理`display`。

    参数:
        path: 参与处理`display`的 `path` 参数。
        root: 参与处理`display`的 `root` 参数。

    返回:
        返回与本函数处理结果对应的数据。
    """
    return path.relative_to(root).as_posix()


def _as_bool(value: Any) -> bool:
    """处理布尔值。

    参数:
        value: 参与处理布尔值的 `value` 参数。

    返回:
        返回与本函数处理结果对应的数据。
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _compose_emoji_message(text: str, emoji: str) -> str:
    """拼装`emoji`、消息。

    参数:
        text: 参与拼装`emoji`、消息的 `text` 参数。
        emoji: 参与拼装`emoji`、消息的 `emoji` 参数。

    返回:
        返回与本函数处理结果对应的数据。
    """
    text = text.strip()
    emoji = emoji.strip()
    if emoji and text:
        return f"{emoji} {text}"
    return emoji or text
