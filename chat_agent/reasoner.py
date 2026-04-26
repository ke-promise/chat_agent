"""推理器与工具循环。

封装模型调用、工具解析、工具执行回灌以及最终回复清洗逻辑。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from chat_agent.agent.provider import LLMProvider
from chat_agent.context import ContextBundle
from chat_agent.messages import InboundMessage
from chat_agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


TOOL_CALL_RE = re.compile(r"<tool_call\s+name=[\"'](?P<name>[^\"']+)[\"']\s*>(?P<body>.*?)</tool_call>", re.DOTALL)


@dataclass(slots=True)
class ReasonerResult:
    """一次推理执行的标准化输出。

    Attributes:
        reply: 最终要返回给用户的文本回复。
        tools_used: 本轮实际调用过的内置工具名称列表。
        mcp_tools_used: 本轮实际调用过的 MCP 工具名称列表。
        error: 若推理失败或降级，记录对应的错误信息。
    """
    reply: str
    tools_used: list[str] = field(default_factory=list)
    mcp_tools_used: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass(slots=True)
class ParsedToolCall:
    """从模型输出中解析出的单条工具调用。

    Attributes:
        name: 工具名称。
        args: 解析后的 JSON 参数字典。
        raw: 原始 `<tool_call>` 文本片段。
        parse_error: 解析失败时的人类可读错误说明。
    """
    name: str
    args: dict[str, Any]
    raw: str
    parse_error: str | None = None


class Reasoner:
    """负责驱动模型推理和多轮工具调用的协调器。

    它会把 `ContextBuilder` 产出的消息发送给主模型，必要时执行多轮工具调用，直到得到
    最终自然语言回复。
    """
    def __init__(
        self,
        provider: LLMProvider,
        tools: ToolRegistry,
        max_iterations: int = 5,
        tool_loop_enabled: bool = True,
    ) -> None:
        """初始化 `Reasoner` 实例。

        参数:
            provider: 初始化 `Reasoner` 时需要的 `provider` 参数。
            tools: 初始化 `Reasoner` 时需要的 `tools` 参数。
            max_iterations: 初始化 `Reasoner` 时需要的 `max_iterations` 参数。
            tool_loop_enabled: 初始化 `Reasoner` 时需要的 `tool_loop_enabled` 参数。
        """
        self.provider = provider
        self.tools = tools
        self.max_iterations = max_iterations
        self.tool_loop_enabled = tool_loop_enabled

    async def run(self, bundle: ContextBundle, inbound: InboundMessage) -> ReasonerResult:
        """执行相关逻辑。

        参数:
            bundle: 参与执行相关逻辑的 `bundle` 参数。
            inbound: 参与执行相关逻辑的 `inbound` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        if inbound.attachments and not self.provider.config.enable_vision:
            return ReasonerResult(reply="当前主模型不支持图片理解，请换用 enable_vision=true 的视觉模型。")

        messages = list(bundle.messages)
        tools_used: list[str] = []
        mcp_tools_used: list[str] = []
        repeated: dict[tuple[str, str], int] = {}
        last_content = ""
        session_tool_names: set[str] = set()

        for _ in range(max(1, self.max_iterations)):
            visible_names = self.tools.resolve_visible_names(session_tool_names)
            schemas = self.tools.get_schema(visible_names) if self.tool_loop_enabled else None
            if schemas:
                logger.info("Visible tools for LLM: %s", ", ".join(visible_names))
            result = await self.provider.chat(messages, tools=schemas)
            if not result.ok:
                return ReasonerResult(reply=result.content, tools_used=tools_used, mcp_tools_used=mcp_tools_used, error=result.content)

            content = result.content.strip()
            last_content = content
            openai_calls = [ParsedToolCall(call["name"], call.get("arguments", {}), call.get("id", "")) for call in result.tool_calls]
            text_calls = parse_tool_calls(content)
            calls = openai_calls or text_calls

            if not self.tool_loop_enabled or not calls:
                return ReasonerResult(reply=strip_tool_calls(content), tools_used=tools_used, mcp_tools_used=mcp_tools_used)

            if result.tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": content or None,
                        "tool_calls": [
                            {
                                "id": call["id"],
                                "type": "function",
                                "function": {
                                    "name": call["name"],
                                    "arguments": call.get("raw_arguments", json.dumps(call.get("arguments", {}), ensure_ascii=False)),
                                },
                            }
                            for call in result.tool_calls
                        ],
                    }
                )
            else:
                messages.append({"role": "assistant", "content": content})

            for call in calls:
                key = (call.name, json.dumps(call.args, sort_keys=True, ensure_ascii=False))
                repeated[key] = repeated.get(key, 0) + 1
                if repeated[key] > 2:
                    logger.warning("Repeated tool call terminated: %s", call.name)
                    return ReasonerResult(reply="工具调用重复过多，我先停止这次操作。", tools_used=tools_used, mcp_tools_used=mcp_tools_used)

                if call.parse_error:
                    tool_result = f"工具参数 JSON 解析失败：{call.parse_error}。请修正后重试。"
                else:
                    if call.name == "tool_search":
                        query = str(call.args.get("query", "")).strip()
                        session_tool_names.update(tool.name for tool in self.tools.search(query, exposures={"discoverable"}))
                    tool_result = await self.tools.execute(call.name, call.args, inbound)
                    tools_used.append(call.name)
                    tool = self.tools.get_tool(call.name)
                    if tool and tool.source.startswith("mcp:"):
                        mcp_tools_used.append(call.name)

                if result.tool_calls and call.raw:
                    messages.append({"role": "tool", "tool_call_id": call.raw, "content": tool_result})
                else:
                    messages.append({"role": "user", "content": f"工具 {call.name} 返回：\n{tool_result}\n请继续给出最终回复。"})

        final_reply = await self._finalize_without_tools(messages, last_content)
        return ReasonerResult(reply=final_reply, tools_used=tools_used, mcp_tools_used=mcp_tools_used)

    async def _finalize_without_tools(self, messages: list[dict[str, Any]], last_content: str) -> str:
        """处理`without`、工具集合。

        参数:
            messages: 参与处理`without`、工具集合的 `messages` 参数。
            last_content: 参与处理`without`、工具集合的 `last_content` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        messages = list(messages)
        messages.append(
            {
                "role": "user",
                "content": (
                    "工具循环已达到最大次数。请立刻停止调用工具，基于上面的工具结果生成最终回复。"
                    "如果工具结果包含搜索标题、摘要或链接，就用这些信息回答并说明来源有限。"
                    "不要回答“没有联网能力”。"
                ),
            }
        )
        result = await self.provider.chat(messages, tools=None)
        if result.ok and result.content.strip():
            return strip_tool_calls(result.content.strip())
        return strip_tool_calls(last_content) or "我已经努力翻找了一圈，但还没整理出足够可靠的答案。你换个更具体的问题，我再陪你查一次。"


def parse_tool_calls(text: str) -> list[ParsedToolCall]:
    """解析工具、`calls`。

    参数:
        text: 参与解析工具、`calls`的 `text` 参数。

    返回:
        返回与本函数处理结果对应的数据。
    """
    calls: list[ParsedToolCall] = []
    for match in TOOL_CALL_RE.finditer(text):
        name = match.group("name").strip()
        body = match.group("body").strip()
        try:
            args = json.loads(body) if body else {}
            if not isinstance(args, dict):
                raise ValueError("tool args must be a JSON object")
            calls.append(ParsedToolCall(name=name, args=args, raw=match.group(0)))
        except Exception as exc:
            calls.append(ParsedToolCall(name=name, args={}, raw=match.group(0), parse_error=str(exc)))
    return calls


def strip_tool_calls(text: str) -> str:
    """清理工具、`calls`。

    参数:
        text: 参与清理工具、`calls`的 `text` 参数。

    返回:
        返回与本函数处理结果对应的数据。
    """
    return TOOL_CALL_RE.sub("", text).strip()
