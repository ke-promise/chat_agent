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

    字段:
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

    字段:
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
        """初始化推理器。

        参数:
            provider: 主模型调用封装。
            tools: 可供模型发现和执行的工具注册表。
            max_iterations: 单轮推理最多允许多少次模型/工具往返。
            tool_loop_enabled: 是否启用工具循环；关闭时只做一次普通模型回复。
        """
        self.provider = provider
        self.tools = tools
        self.max_iterations = max_iterations
        self.tool_loop_enabled = tool_loop_enabled

    async def run(self, bundle: ContextBundle, inbound: InboundMessage) -> ReasonerResult:
        """执行一轮模型推理，并在需要时驱动多轮工具调用。

        参数:
            bundle: ContextBuilder 已拼装好的模型上下文。
            inbound: 当前入站消息，工具执行时会作为 ToolContext 的来源消息。

        返回:
            标准化 ReasonerResult，包含最终回复、工具使用记录和可能的错误信息。
        """
        if inbound.attachments and not self.provider.config.enable_vision:
            return ReasonerResult(reply="我看见你发了图片，但当前主模型还没开图片理解。把 enable_vision=true 的视觉模型换上后，我就能认真看图啦。")

        messages = list(bundle.messages)
        tools_used: list[str] = []
        mcp_tools_used: list[str] = []
        repeated: dict[tuple[str, str], int] = {}
        last_content = ""
        session_tool_names: set[str] = set()
        disabled_tool_names: set[str] = set()

        for _ in range(max(1, self.max_iterations)):
            visible_names = [
                name
                for name in self.tools.resolve_visible_names(session_tool_names)
                if name not in disabled_tool_names
            ]
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
                assistant_message = {
                    "role": "assistant",
                    "content": content,
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
                if result.reasoning_content is not None:
                    assistant_message["reasoning_content"] = result.reasoning_content
                messages.append(assistant_message)
            else:
                assistant_message = {"role": "assistant", "content": content}
                if result.reasoning_content is not None:
                    assistant_message["reasoning_content"] = result.reasoning_content
                messages.append(assistant_message)

            for call in calls:
                key = (call.name, json.dumps(call.args, sort_keys=True, ensure_ascii=False))
                repeated[key] = repeated.get(key, 0) + 1
                if repeated[key] > 2:
                    logger.warning("Repeated tool call terminated: %s", call.name)
                    return ReasonerResult(reply="这个工具我绕了太多圈，先停一下，免得把你也绕晕。我们换个问法或稍后再试。", tools_used=tools_used, mcp_tools_used=mcp_tools_used)

                if call.parse_error:
                    tool_result = f"工具参数 JSON 解析失败：{call.parse_error}。请修正后重试。"
                else:
                    if call.name in disabled_tool_names:
                        tool_result = (
                            f"工具 {call.name} 本轮已因临时故障停用。"
                            "请不要继续调用它；如果还有其他可见工具可以尝试，否则请基于已有信息给出有限回复。"
                        )
                    elif call.name == "tool_search":
                        query = str(call.args.get("query", "")).strip()
                        session_tool_names.update(tool.name for tool in self.tools.search(query, exposures={"discoverable"}))
                        tool_result = await self.tools.execute(call.name, call.args, inbound)
                    else:
                        tool_result = await self.tools.execute(call.name, call.args, inbound)

                    tools_used.append(call.name)
                    tool = self.tools.get_tool(call.name)
                    if tool and tool.source.startswith("mcp:"):
                        mcp_tools_used.append(call.name)
                    if _is_degraded_search_result(tool_result):
                        disabled_tool_names.add(call.name)
                        tool_result = (
                            f"{tool_result}\n\n"
                            f"注意：工具 {call.name} 的搜索后端本轮临时不可用，已经停用。"
                            "请不要换关键词重复调用它；如果没有其他搜索工具，请直接说明实时搜索结果暂时不可用。"
                        )

                if result.tool_calls and call.raw:
                    messages.append({"role": "tool", "tool_call_id": call.raw, "content": tool_result})
                else:
                    messages.append({"role": "user", "content": f"工具 {call.name} 返回：\n{tool_result}\n请继续给出最终回复。"})

        final_reply = await self._finalize_without_tools(messages, last_content)
        return ReasonerResult(reply=final_reply, tools_used=tools_used, mcp_tools_used=mcp_tools_used)

    async def _finalize_without_tools(self, messages: list[dict[str, Any]], last_content: str) -> str:
        """在工具循环达到上限后，要求模型停止调用工具并生成最终回复。

        参数:
            messages: 已包含历史工具结果的消息列表。
            last_content: 最后一次模型文本输出，用于最终兜底。

        返回:
            清理过工具标签的最终自然语言回复。
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
    """从文本协议中解析 `<tool_call>` 工具调用。

    参数:
        text: 模型返回的原始文本。

    返回:
        解析出的工具调用列表；JSON 参数错误会保留在 parse_error 中。
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
    """移除文本回复中残留的 `<tool_call>` 标签。

    参数:
        text: 原始模型回复。

    返回:
        适合展示给用户的纯文本。
    """
    return TOOL_CALL_RE.sub("", text).strip()


def _is_degraded_search_result(text: str) -> bool:
    """判断工具结果是否表示搜索后端已降级为临时空结果。"""
    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and payload.get("degraded") is True and isinstance(payload.get("results"), list)
