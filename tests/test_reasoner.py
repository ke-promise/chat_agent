from __future__ import annotations

import pytest

from chat_agent.agent.provider import LLMResult
from chat_agent.context import ContextBundle
from chat_agent.messages import InboundMessage
from chat_agent.reasoner import Reasoner, parse_tool_calls
from chat_agent.tools.registry import Tool, ToolContext, ToolRegistry


class SequenceProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, messages, tools=None):
        self.calls += 1
        if self.calls == 1:
            return LLMResult('<tool_call name="echo">{"value":"ok"}</tool_call>', [])
        return LLMResult("final reply", [])


class OpenAIToolProvider:
    def __init__(self) -> None:
        self.calls = 0
        self.config = type("Config", (), {"enable_vision": False})()

    async def chat(self, messages, tools=None):
        self.calls += 1
        if self.calls == 1:
            return LLMResult("", [{"id": "call-1", "name": "echo", "arguments": {"value": "ok"}, "raw_arguments": '{"value":"ok"}'}])
        return LLMResult("final reply", [])


class NeverStopsToolProvider:
    def __init__(self) -> None:
        self.calls = 0
        self.config = type("Config", (), {"enable_vision": False})()

    async def chat(self, messages, tools=None):
        self.calls += 1
        if tools:
            return LLMResult("", [{"id": f"call-{self.calls}", "name": "echo", "arguments": {"value": "loop"}, "raw_arguments": '{"value":"loop"}'}])
        return LLMResult("fallback final reply", [])


class DiscoveryProvider:
    def __init__(self) -> None:
        self.calls = 0
        self.seen_tools: list[list[str]] = []
        self.config = type("Config", (), {"enable_vision": False})()

    async def chat(self, messages, tools=None):
        self.calls += 1
        self.seen_tools.append([tool["function"]["name"] for tool in (tools or [])])
        if self.calls == 1:
            return LLMResult("", [{"id": "call-1", "name": "tool_search", "arguments": {"query": "discover"}, "raw_arguments": '{"query":"discover"}'}])
        if self.calls == 2:
            return LLMResult("", [{"id": "call-2", "name": "discover_tool", "arguments": {"value": "ok"}, "raw_arguments": '{"value":"ok"}'}])
        return LLMResult("final", [])


@pytest.mark.asyncio
async def test_reasoner_parses_and_executes_tool_call() -> None:
    async def echo(context: ToolContext, args: dict) -> str:
        return args["value"]

    registry = ToolRegistry()
    registry.register(Tool("echo", "echo tool", {"type": "object", "properties": {"value": {"type": "string"}}}, echo))
    reasoner = Reasoner(SequenceProvider(), registry, max_iterations=3)
    bundle = ContextBundle(messages=[{"role": "user", "content": "run"}], memory_hits=[], trace={})
    message = InboundMessage(channel="telegram", chat_id="chat-1", sender="user-1", content="run")

    result = await reasoner.run(bundle, message)

    assert result.reply == "final reply"
    assert result.tools_used == ["echo"]


@pytest.mark.asyncio
async def test_reasoner_finalizes_when_tool_loop_hits_limit() -> None:
    async def echo(context: ToolContext, args: dict) -> str:
        return args["value"]

    registry = ToolRegistry()
    registry.register(Tool("echo", "echo tool", {"type": "object", "properties": {"value": {"type": "string"}}}, echo, exposure="always"))
    provider = NeverStopsToolProvider()
    reasoner = Reasoner(provider, registry, max_iterations=2)
    bundle = ContextBundle(messages=[{"role": "user", "content": "run"}], memory_hits=[], trace={})
    message = InboundMessage(channel="telegram", chat_id="chat-1", sender="user-1", content="run")

    result = await reasoner.run(bundle, message)

    assert result.reply == "fallback final reply"
    assert provider.calls == 3


def test_parse_tool_calls_bad_json_gets_error() -> None:
    calls = parse_tool_calls('<tool_call name="x">{bad}</tool_call>')

    assert calls[0].name == "x"
    assert calls[0].parse_error


@pytest.mark.asyncio
async def test_reasoner_executes_openai_tool_calls() -> None:
    async def echo(context: ToolContext, args: dict) -> str:
        return args["value"]

    registry = ToolRegistry()
    registry.register(Tool("echo", "echo tool", {"type": "object", "properties": {"value": {"type": "string"}}}, echo, exposure="always"))
    reasoner = Reasoner(OpenAIToolProvider(), registry, max_iterations=3)
    bundle = ContextBundle(messages=[{"role": "user", "content": "run"}], memory_hits=[], trace={})
    message = InboundMessage(channel="telegram", chat_id="chat-1", sender="user-1", content="run")

    result = await reasoner.run(bundle, message)

    assert result.reply == "final reply"
    assert result.tools_used == ["echo"]


@pytest.mark.asyncio
async def test_reasoner_tool_search_only_unlocks_tools_within_single_run() -> None:
    async def tool_search(context: ToolContext, args: dict) -> str:
        return "search complete"

    async def discover_tool(context: ToolContext, args: dict) -> str:
        return args["value"]

    registry = ToolRegistry()
    registry.register(
        Tool("tool_search", "discover tool helper", {"type": "object", "properties": {"query": {"type": "string"}}}, tool_search, exposure="always")
    )
    registry.register(
        Tool("discover_tool", "discover target", {"type": "object", "properties": {"value": {"type": "string"}}}, discover_tool, exposure="discoverable")
    )

    provider = DiscoveryProvider()
    reasoner = Reasoner(provider, registry, max_iterations=3)
    bundle = ContextBundle(messages=[{"role": "user", "content": "run"}], memory_hits=[], trace={})
    message = InboundMessage(channel="telegram", chat_id="chat-1", sender="user-1", content="run")

    result = await reasoner.run(bundle, message)

    assert result.reply == "final"
    assert provider.seen_tools[0] == ["tool_search"]
    assert set(provider.seen_tools[1]) == {"tool_search", "discover_tool"}
    assert result.tools_used == ["tool_search", "discover_tool"]

    second_provider = DiscoveryProvider()
    second_reasoner = Reasoner(second_provider, registry, max_iterations=1)
    await second_reasoner.run(bundle, message)

    assert second_provider.seen_tools[0] == ["tool_search"]
