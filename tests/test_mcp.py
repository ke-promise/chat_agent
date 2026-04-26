from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from chat_agent.mcp.registry import MCPRegistry
from chat_agent.memory.store import SQLiteStore
from chat_agent.messages import InboundMessage
from chat_agent.tools.registry import ToolRegistry


def _python_exe() -> str:
    return sys.executable


def _write_config(path: Path, servers: dict[str, dict]) -> None:
    path.write_text(json.dumps({"servers": servers}), encoding="utf-8")


@pytest.mark.asyncio
async def test_mcp_fake_server_tools_list_and_call(tmp_path: Path) -> None:
    server = tmp_path / "fake_mcp.py"
    server.write_text(
        """
import json, sys
for line in sys.stdin:
    req = json.loads(line)
    method = req.get("method")
    if method == "initialize":
        result = {"serverInfo": {"name": "fake"}}
    elif method == "tools/list":
        result = {"tools": [{"name": "hello", "description": "say hello", "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}}}}]}
    elif method == "tools/call":
        result = {"content": "hello " + req["params"]["arguments"].get("name", "world")}
    else:
        result = {}
    print(json.dumps({"jsonrpc": "2.0", "id": req.get("id"), "result": result}), flush=True)
""",
        encoding="utf-8",
    )
    config = tmp_path / "mcp_servers.json"
    _write_config(config, {"fake": {"command": [_python_exe(), str(server)], "env": {}}})
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    tools = ToolRegistry(store)
    registry = MCPRegistry(config, tools, store)

    await registry.load()
    result = await tools.execute(
        "fake_hello",
        {"name": "alice"},
        InboundMessage(channel="telegram", chat_id="chat-1", sender="user-1", content="hi"),
    )
    await registry.shutdown()

    assert "hello alice" in result


@pytest.mark.asyncio
async def test_mcp_search_tool_is_visible_by_default(tmp_path: Path) -> None:
    server = tmp_path / "fake_search_mcp.py"
    server.write_text(
        """
import json, sys
for line in sys.stdin:
    req = json.loads(line)
    method = req.get("method")
    if method == "initialize":
        result = {"serverInfo": {"name": "fake-search"}}
    elif method == "tools/list":
        result = {"tools": [{"name": "web_search", "description": "search the web", "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}}}]}
    elif method == "tools/call":
        result = {"content": "result"}
    else:
        result = {}
    print(json.dumps({"jsonrpc": "2.0", "id": req.get("id"), "result": result}), flush=True)
""",
        encoding="utf-8",
    )
    config = tmp_path / "mcp_servers.json"
    _write_config(config, {"duckduckgo": {"command": [_python_exe(), str(server)], "env": {}}})
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    tools = ToolRegistry(store)
    registry = MCPRegistry(config, tools, store)

    await registry.load()

    assert "duckduckgo_web_search" in tools.visible_names()

    await registry.shutdown()


@pytest.mark.asyncio
async def test_non_search_mcp_tool_is_hidden_by_default(tmp_path: Path) -> None:
    server = tmp_path / "fake_hidden_mcp.py"
    server.write_text(
        """
import json, sys
for line in sys.stdin:
    req = json.loads(line)
    method = req.get("method")
    if method == "initialize":
        result = {"serverInfo": {"name": "fake"}}
    elif method == "tools/list":
        result = {"tools": [{"name": "hello", "description": "say hello", "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}}}}]}
    elif method == "tools/call":
        result = {"content": "hello " + req["params"]["arguments"].get("name", "world")}
    else:
        result = {}
    print(json.dumps({"jsonrpc": "2.0", "id": req.get("id"), "result": result}), flush=True)
""",
        encoding="utf-8",
    )
    config = tmp_path / "mcp_servers.json"
    _write_config(config, {"fake": {"command": [_python_exe(), str(server)], "env": {}}})
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    tools = ToolRegistry(store)
    registry = MCPRegistry(config, tools, store)

    await registry.load()

    assert "fake_hello" not in tools.visible_names()
    assert tools.get_tool("fake_hello") is not None

    await registry.shutdown()


@pytest.mark.asyncio
async def test_mcp_skips_non_json_stdout_and_sanitizes_tool_name(tmp_path: Path) -> None:
    server = tmp_path / "noisy_mcp.py"
    server.write_text(
        """
import json, sys
for line in sys.stdin:
    req = json.loads(line)
    method = req.get("method")
    print("server banner on stdout", flush=True)
    if method == "initialize":
        result = {"serverInfo": {"name": "noisy"}}
    elif method == "tools/list":
        result = {"tools": [{"name": "web-search", "description": "search the web", "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}}}]}
    elif method == "tools/call":
        result = {"content": "searched " + req["params"]["arguments"].get("query", "")}
    else:
        result = {}
    print(json.dumps({"jsonrpc": "2.0", "id": req.get("id"), "result": result}), flush=True)
""",
        encoding="utf-8",
    )
    config = tmp_path / "mcp_servers.json"
    _write_config(config, {"duckduckgo": {"command": [_python_exe(), str(server)], "env": {}}})
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    tools = ToolRegistry(store)
    registry = MCPRegistry(config, tools, store)

    await registry.load()
    result = await tools.execute(
        "duckduckgo_web_search",
        {"query": "news"},
        InboundMessage(channel="telegram", chat_id="chat-1", sender="user-1", content="hi"),
    )
    await registry.shutdown()

    assert "searched news" in result


@pytest.mark.asyncio
async def test_mcp_missing_env_placeholder_skips_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEARCH_API_KEY", raising=False)
    config = tmp_path / "mcp_servers.json"
    _write_config(
        config,
        {"search": {"command": ["missing-command"], "env": {"SEARCH_API_KEY": "${SEARCH_API_KEY}"}}},
    )
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    tools = ToolRegistry(store)
    registry = MCPRegistry(config, tools, store)

    await registry.load()

    assert registry.servers == {}


@pytest.mark.asyncio
async def test_mcp_allowed_tools_filters_registration_and_calls(tmp_path: Path) -> None:
    server = tmp_path / "fake_allow_mcp.py"
    server.write_text(
        """
import json, sys
for line in sys.stdin:
    req = json.loads(line)
    method = req.get("method")
    if method == "initialize":
        result = {"serverInfo": {"name": "fake"}}
    elif method == "tools/list":
        result = {"tools": [
            {"name": "hello", "description": "say hello", "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}}}},
            {"name": "web_search", "description": "search", "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}}}
        ]}
    elif method == "tools/call":
        result = {"content": req["params"]["name"]}
    else:
        result = {}
    print(json.dumps({"jsonrpc": "2.0", "id": req.get("id"), "result": result}), flush=True)
""",
        encoding="utf-8",
    )
    config = tmp_path / "mcp_servers.json"
    _write_config(config, {"fake": {"command": [_python_exe(), str(server)], "env": {}}})
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    tools = ToolRegistry(store)
    registry = MCPRegistry(config, tools, store, allowed_tools=["fake:web_search"])

    await registry.load()

    assert tools.get_tool("fake_web_search") is not None
    assert tools.get_tool("fake_hello") is None
    with pytest.raises(RuntimeError, match="not allowed"):
        await registry.call_tool("fake", "hello", {"name": "alice"})

    await registry.shutdown()


@pytest.mark.asyncio
async def test_mcp_concurrent_requests_are_routed_to_the_right_callers(tmp_path: Path) -> None:
    server = tmp_path / "fake_concurrent_mcp.py"
    server.write_text(
        """
import json, sys, time
for line in sys.stdin:
    req = json.loads(line)
    method = req.get("method")
    if method == "initialize":
        result = {"serverInfo": {"name": "fake"}}
    elif method == "tools/list":
        result = {"tools": [{"name": "hello", "description": "say hello", "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}, "delay_ms": {"type": "integer"}}}}]}
    elif method == "tools/call":
        args = req["params"]["arguments"]
        time.sleep((args.get("delay_ms", 0) or 0) / 1000)
        result = {"content": "hello " + args.get("name", "world")}
    else:
        result = {}
    print(json.dumps({"jsonrpc": "2.0", "id": req.get("id"), "result": result}), flush=True)
""",
        encoding="utf-8",
    )
    config = tmp_path / "mcp_servers.json"
    _write_config(config, {"fake": {"command": [_python_exe(), str(server)], "env": {}}})
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    tools = ToolRegistry(store)
    registry = MCPRegistry(config, tools, store)

    await registry.load()
    results = await asyncio.gather(
        registry.call_tool("fake", "hello", {"name": "slow", "delay_ms": 120}),
        registry.call_tool("fake", "hello", {"name": "fast", "delay_ms": 0}),
    )

    assert results[0]["content"] == "hello slow"
    assert results[1]["content"] == "hello fast"

    await registry.shutdown()


@pytest.mark.asyncio
async def test_mcp_stderr_output_does_not_block_calls(tmp_path: Path) -> None:
    server = tmp_path / "fake_stderr_mcp.py"
    server.write_text(
        """
import json, sys
for line in sys.stdin:
    req = json.loads(line)
    method = req.get("method")
    print("server log", file=sys.stderr, flush=True)
    if method == "initialize":
        result = {"serverInfo": {"name": "fake"}}
    elif method == "tools/list":
        result = {"tools": [{"name": "hello", "description": "say hello", "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}}}}]}
    elif method == "tools/call":
        result = {"content": "hello " + req["params"]["arguments"].get("name", "world")}
    else:
        result = {}
    print(json.dumps({"jsonrpc": "2.0", "id": req.get("id"), "result": result}), flush=True)
""",
        encoding="utf-8",
    )
    config = tmp_path / "mcp_servers.json"
    _write_config(config, {"fake": {"command": [_python_exe(), str(server)], "env": {}}})
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    tools = ToolRegistry(store)
    registry = MCPRegistry(config, tools, store)

    await registry.load()
    result = await registry.call_tool("fake", "hello", {"name": "alice"})

    assert result["content"] == "hello alice"

    await registry.shutdown()


@pytest.mark.asyncio
async def test_duckduckgo_search_results_are_normalized(tmp_path: Path) -> None:
    server = tmp_path / "fake_duckduckgo_mcp.py"
    server.write_text(
        """
import json, sys
for line in sys.stdin:
    req = json.loads(line)
    method = req.get("method")
    if method == "initialize":
        result = {"serverInfo": {"name": "duckduckgo"}}
    elif method == "tools/list":
        result = {"tools": [{"name": "web-search", "description": "search the web", "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}}}]}
    elif method == "tools/call":
        result = {"content": [{"type": "text", "text": "1. **OpenAI**\\nURL: https://openai.com/\\nSnippet: AI research lab\\n\\n2. **Example**\\nURL: https://example.com/\\nSnippet: Example site"}]}
    else:
        result = {}
    print(json.dumps({"jsonrpc": "2.0", "id": req.get("id"), "result": result}), flush=True)
""",
        encoding="utf-8",
    )
    config = tmp_path / "mcp_servers.json"
    _write_config(config, {"duckduckgo": {"command": [_python_exe(), str(server)], "env": {}}})
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    tools = ToolRegistry(store)
    registry = MCPRegistry(config, tools, store)

    await registry.load()
    result = await registry.call_tool("duckduckgo", "web-search", {"query": "openai"})

    assert result["query"] == "openai"
    assert result["provider"] == "duckduckgo"
    assert result["results"][0]["title"] == "OpenAI"
    assert result["results"][0]["url"] == "https://openai.com/"
    assert result["results"][0]["snippet"] == "AI research lab"

    await registry.shutdown()


@pytest.mark.asyncio
async def test_duckduckgo_search_schema_and_args_are_sanitized(tmp_path: Path) -> None:
    server = tmp_path / "fake_duckduckgo_schema_mcp.py"
    server.write_text(
        """
import json, sys
for line in sys.stdin:
    req = json.loads(line)
    method = req.get("method")
    if method == "initialize":
        result = {"serverInfo": {"name": "duckduckgo"}}
    elif method == "tools/list":
        result = {"tools": [{"name": "web-search", "description": "search the web", "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}, "numResults": {"type": "integer"}, "mode": {"type": "string"}}}}]}
    elif method == "tools/call":
        result = {"content": json.dumps(req["params"]["arguments"], ensure_ascii=False)}
    else:
        result = {}
    print(json.dumps({"jsonrpc": "2.0", "id": req.get("id"), "result": result}), flush=True)
""",
        encoding="utf-8",
    )
    config = tmp_path / "mcp_servers.json"
    _write_config(config, {"duckduckgo": {"command": [_python_exe(), str(server)], "env": {}}})
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    tools = ToolRegistry(store)
    registry = MCPRegistry(config, tools, store)

    await registry.load()
    tool = tools.get_tool("duckduckgo_web_search")
    assert tool is not None
    mode_schema = tool.parameters["properties"]["mode"]
    assert mode_schema["enum"] == ["short"]
    assert tool.parameters["properties"]["numResults"]["maximum"] == 5

    result = await registry.call_tool("duckduckgo", "web-search", {"query": "news", "numResults": 10, "mode": "detailed"})
    assert result["raw_text"] == '{"query": "news", "numResults": 5, "mode": "short"}'

    await registry.shutdown()


@pytest.mark.asyncio
async def test_duckduckgo_transient_search_failure_degrades_to_empty_results(tmp_path: Path) -> None:
    server = tmp_path / "fake_duckduckgo_error_mcp.py"
    server.write_text(
        """
import json, sys
for line in sys.stdin:
    req = json.loads(line)
    method = req.get("method")
    if method == "initialize":
        result = {"serverInfo": {"name": "duckduckgo"}}
        print(json.dumps({"jsonrpc": "2.0", "id": req.get("id"), "result": result}), flush=True)
    elif method == "tools/list":
        result = {"tools": [{"name": "web-search", "description": "search the web", "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}}}]}
        print(json.dumps({"jsonrpc": "2.0", "id": req.get("id"), "result": result}), flush=True)
    elif method == "tools/call":
        print(json.dumps({"jsonrpc": "2.0", "id": req.get("id"), "error": "HTTP 202: Failed to fetch search results"}), flush=True)
    else:
        print(json.dumps({"jsonrpc": "2.0", "id": req.get("id"), "result": {}}), flush=True)
""",
        encoding="utf-8",
    )
    config = tmp_path / "mcp_servers.json"
    _write_config(config, {"duckduckgo": {"command": [_python_exe(), str(server)], "env": {}}})
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    tools = ToolRegistry(store)
    registry = MCPRegistry(config, tools, store)

    await registry.load()
    result = await tools.execute(
        "duckduckgo_web_search",
        {"query": "鸣潮 同人视频推荐"},
        InboundMessage(channel="telegram", chat_id="chat-1", sender="user-1", content="hi"),
    )
    await registry.shutdown()

    payload = json.loads(result)
    assert payload["query"] == "鸣潮 同人视频推荐"
    assert payload["results"] == []
    assert payload["provider"] == "duckduckgo"
    assert payload["degraded"] is True


@pytest.mark.asyncio
async def test_web_content_fetch_tool_is_hidden_and_normalized(tmp_path: Path) -> None:
    server = tmp_path / "fake_web_content_mcp.py"
    server.write_text(
        """
import json, sys
for line in sys.stdin:
    req = json.loads(line)
    method = req.get("method")
    if method == "initialize":
        result = {"serverInfo": {"name": "web-content"}}
    elif method == "tools/list":
        result = {"tools": [{"name": "fetch_page", "description": "fetch page", "inputSchema": {"type": "object", "properties": {"url": {"type": "string"}}}}]}
    elif method == "tools/call":
        result = {"url": req["params"]["arguments"]["url"], "title": "Title", "content": "Body"}
    else:
        result = {}
    print(json.dumps({"jsonrpc": "2.0", "id": req.get("id"), "result": result}), flush=True)
""",
        encoding="utf-8",
    )
    config = tmp_path / "mcp_servers.json"
    _write_config(config, {"web_content": {"command": [_python_exe(), str(server)], "env": {}}})
    store = SQLiteStore(tmp_path / "agent.sqlite3")
    tools = ToolRegistry(store)
    registry = MCPRegistry(config, tools, store)

    await registry.load()
    result = await registry.call_tool("web_content", "fetch_page", {"url": "https://example.com"})

    assert "web_content_fetch_page" not in tools.visible_names()
    assert result == {
        "url": "https://example.com",
        "title": "Title",
        "content": "Body",
        "provider": "web_content",
    }

    await registry.shutdown()
