"""工具注册、发现与执行中心。"""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal

from chat_agent.messages import InboundMessage

logger = logging.getLogger(__name__)

ToolExposure = Literal["always", "discoverable", "hidden"]
ToolRisk = Literal["read", "write", "side_effect"]


@dataclass(slots=True)
class ToolContext:
    """传入工具实现函数的运行时上下文。

    字段:
        message: 触发本次工具调用的入站消息。
        store: 供工具读写业务状态的存储层对象。
    """

    message: InboundMessage
    store: Any


ToolFunc = Callable[[ToolContext, dict[str, Any]], str | Awaitable[str]]


@dataclass(slots=True)
class Tool:
    """统一的工具定义。

    字段:
        name: 工具名，必须在整个注册表中唯一。
        description: 提供给模型和调试日志的工具说明。
        parameters: JSON Schema 形式的参数定义。
        func: 实际执行工具逻辑的同步或异步函数。
        exposure: 工具暴露级别，决定默认是否可见。
        risk: 工具风险等级，用于日志、检索和治理。
        source: 工具来源，例如 builtin 或具体 MCP server。
    """

    name: str
    description: str
    parameters: dict[str, Any]
    func: ToolFunc
    exposure: ToolExposure = "discoverable"
    risk: ToolRisk = "read"
    source: str = "builtin"

    async def execute(self, context: ToolContext, **kwargs: Any) -> str:
        """执行当前工具定义对应的实现函数。

        参数:
            context: 本次工具调用的运行时上下文，包含触发消息和存储层引用。
            **kwargs: 传给工具函数的具名参数，会被打包成字典交给 `func`。

        返回:
            返回工具执行结果的字符串表示，供 Reasoner 回填到后续上下文。
        """
        result = self.func(context, kwargs)
        if inspect.isawaitable(result):
            result = await result
        return str(result)

    def schema(self) -> dict[str, Any]:
        """生成提供给模型的 OpenAI-compatible function schema。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def description_line(self) -> str:
        """生成一行便于展示和检索的工具摘要文本。"""
        params = ", ".join(self.parameters.get("properties", {}).keys())
        return f"- {self.name} [{self.source}, {self.exposure}, {self.risk}]: {self.description} 参数: {params or '无'}"


class ToolRegistry:
    """维护工具目录并负责工具发现与调用。"""

    def __init__(self, store: Any | None = None, extra_model_tools: Iterable[str] | None = None) -> None:
        """初始化工具注册表。

        参数:
            store: 默认注入到 `ToolContext` 的存储层对象。
            extra_model_tools: 即使不是 `always` 暴露级别，也强制向模型公开的工具名集合。
        """
        self.store = store
        self._tools: dict[str, Tool] = {}
        self._extra_model_tools = {str(name).strip() for name in (extra_model_tools or []) if str(name).strip()}

    def register(
        self,
        tool: Tool,
        source: str | None = None,
        exposure: ToolExposure | None = None,
        risk: ToolRisk | None = None,
    ) -> None:
        """注册或覆盖一个工具定义。

        参数:
            tool: 待注册的工具对象。
            source: 可选工具来源；传入时会覆盖 `tool.source`。
            exposure: 可选暴露级别；传入时会覆盖 `tool.exposure`。
            risk: 可选风险等级；传入时会覆盖 `tool.risk`。
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        if source is not None:
            tool.source = source
        if exposure is not None:
            tool.exposure = exposure
        if risk is not None:
            tool.risk = risk
        self._tools[tool.name] = tool

    def unregister_source(self, source: str) -> None:
        """批量移除某个来源注册的所有工具。"""
        for name in [name for name, tool in self._tools.items() if tool.source == source]:
            del self._tools[name]

    def get_tool(self, name: str) -> Tool | None:
        """按名称读取工具定义，不存在时返回 `None`。"""
        return self._tools.get(name)

    def get(self, name: str) -> Tool | None:
        """`get_tool` 的简短别名。"""
        return self.get_tool(name)

    def search(
        self,
        query: str,
        limit: int = 8,
        exposures: set[ToolExposure] | None = None,
    ) -> list[Tool]:
        """按名称、描述和来源模糊搜索工具。"""
        needle = query.lower().strip()
        allowed_exposures = exposures or {"discoverable"}
        scored: list[Tool] = []
        for tool in self._tools.values():
            if tool.exposure not in allowed_exposures:
                continue
            haystack = f"{tool.name} {tool.description} {tool.source}".lower()
            if not needle or needle in haystack or any(part in haystack for part in needle.split()):
                scored.append(tool)
        return scored[:limit]

    def default_visible_names(self) -> list[str]:
        """返回默认就应暴露给模型的工具名列表。"""
        visible: list[str] = []
        for tool in self._tools.values():
            if tool.exposure == "always" or tool.name in self._extra_model_tools:
                visible.append(tool.name)
        return visible

    def resolve_visible_names(self, session_tool_names: Iterable[str] | None = None) -> list[str]:
        """合并默认工具与本轮会话临时开放的工具名。"""
        visible = set(self.default_visible_names())
        if session_tool_names:
            visible.update(str(name) for name in session_tool_names if str(name) in self._tools)
        return [tool.name for tool in self._tools.values() if tool.name in visible]

    def get_schema(self, names: list[str] | None = None) -> list[dict[str, Any]]:
        """按名称列表导出工具 schema；为空时导出默认可见工具。"""
        selected_names = names if names is not None else self.default_visible_names()
        return [self._tools[name].schema() for name in selected_names if name in self._tools]

    def list_descriptions(self, only_visible: bool = False, names: list[str] | None = None) -> str:
        """返回适合拼进 prompt 的多行工具说明文本。"""
        if names is not None:
            tools = [self._tools[name] for name in names if name in self._tools]
        elif only_visible:
            visible = set(self.default_visible_names())
            tools = [tool for tool in self._tools.values() if tool.name in visible]
        else:
            tools = list(self._tools.values())
        if not tools:
            return "- 当前没有可用工具"
        return "\n".join(tool.description_line() for tool in tools)

    def visible_count(self) -> int:
        """统计当前默认对模型可见的工具数量。"""
        return len(self.default_visible_names())

    def visible_names(self) -> list[str]:
        """返回当前默认可见工具名列表。"""
        return self.default_visible_names()

    async def execute(self, name: str, args: dict[str, Any], message: InboundMessage) -> str:
        """执行指定工具并返回字符串结果。

        参数:
            name: 要执行的工具名。
            args: 传给工具实现的参数字典。
            message: 触发本次执行的入站消息。
        """
        tool = self.get_tool(name)
        if tool is None:
            return f"工具 {name} 不存在。"
        try:
            return await tool.execute(ToolContext(message=message, store=self.store), **args)
        except Exception:
            logger.exception("Tool execution failed: %s args=%s", name, json.dumps(args, ensure_ascii=False))
            return f"调用工具 {name} 时失败了。"

    async def call(self, name: str, arguments: dict[str, Any], message: InboundMessage) -> str:
        """兼容旧调用路径的薄封装，内部直接转发到 `execute`。"""
        return await self.execute(name, arguments, message)
