"""MCP 标准输入输出注册表。

负责启动外部 MCP server、发现其工具并把它们注册到运行时工具表中，同时提供
安全的请求路由、结果归一化和故障降级能力。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chat_agent.memory.store import SQLiteStore
from chat_agent.tools.registry import Tool, ToolContext, ToolRegistry

logger = logging.getLogger(__name__)
ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
TOOL_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_]")


@dataclass(slots=True)
class MCPServer:
    """单个 MCP 服务器的运行时状态。

    字段:
        name: 服务器名称，对应配置文件里的 key。
        command: 启动该 MCP server 的命令行数组。
        env: 启动进程时附加的环境变量。
        process: 已启动的子进程对象；未启动时为 `None`。
        request_id: 当前 server 已分配到的 JSON-RPC 请求序号。
        tools: 该 server 暴露的原始工具描述列表。
        pending_requests: 等待返回结果的请求 Future，键为请求 id。
        write_lock: 串行写入 stdin 的锁，避免并发请求互相覆盖。
        stdout_reader_task: 持续读取标准输出的后台任务。
        stderr_reader_task: 持续读取标准错误的后台任务。
    """
    name: str
    command: list[str]
    env: dict[str, str] = field(default_factory=dict)
    process: asyncio.subprocess.Process | None = None
    request_id: int = 0
    tools: list[dict[str, Any]] = field(default_factory=list)
    pending_requests: dict[int, asyncio.Future[dict[str, Any]]] = field(default_factory=dict)
    write_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    stdout_reader_task: asyncio.Task[None] | None = None
    stderr_reader_task: asyncio.Task[None] | None = None


class MCPRegistry:
    """MCP 工具注册与请求分发中心。

    该类负责读取 MCP 配置、管理 server 生命周期、把 MCP 工具注册成内部 `Tool`，
    并在工具执行时把参数路由到对应的 MCP server。
    """
    def __init__(
        self,
        config_path: Path,
        tool_registry: ToolRegistry,
        store: SQLiteStore,
        allowed_servers: list[str] | None = None,
        allowed_tools: list[str] | None = None,
    ) -> None:
        """初始化 MCP 注册表。

        参数:
            config_path: MCP server 配置文件路径。
            tool_registry: 需要写入 MCP 工具的运行时工具注册表。
            store: 用于写入 MCP 调用日志的存储层。
            allowed_servers: 可选 server 白名单；为空表示按配置加载全部启用 server。
            allowed_tools: 可选原始工具白名单，格式为 `<server>:<tool>`。
        """
        self.config_path = Path(config_path)
        self.tool_registry = tool_registry
        self.store = store
        self.servers: dict[str, MCPServer] = {}
        self.allowed_servers = {item.strip() for item in (allowed_servers or []) if str(item).strip()}
        self.allowed_tools = {item.strip() for item in (allowed_tools or []) if str(item).strip()}

    async def load(self) -> None:
        """读取 MCP 配置、启动 server，并把发现的工具注册到 ToolRegistry。"""
        await self.shutdown()
        if not self.config_path.exists():
            logger.warning("MCP config not found: %s", self.config_path)
            return
        data = json.loads(self.config_path.read_text(encoding="utf-8"))
        for name, raw in data.get("servers", {}).items():
            if raw.get("enabled") is False:
                logger.info("MCP server disabled name=%s", name)
                continue
            if self.allowed_servers and name not in self.allowed_servers:
                logger.info("MCP server skipped by allowlist name=%s", name)
                continue
            env = _expand_env(dict(raw.get("env", {})))
            if env is None:
                logger.warning("MCP server %s skipped because required environment variables are missing", name)
                continue
            server = MCPServer(name=name, command=list(raw.get("command", [])), env=env)
            if not server.command:
                logger.warning("MCP server %s has empty command", name)
                continue
            try:
                await self._start_server(server)
                self.servers[name] = server
            except Exception as exc:
                await self._stop_server_process(server)
                logger.warning("Failed to start MCP server %s: %s", name, exc)

    async def reload(self) -> None:
        """重新加载 MCP 配置并重建所有 server 连接。"""
        await self.load()

    async def shutdown(self) -> None:
        """停止所有 MCP server，并移除它们注册过的工具。"""
        old_names = list(self.servers.keys())
        for server in self.servers.values():
            await self._stop_server_process(server)
        self.servers.clear()
        for name in old_names:
            self.tool_registry.unregister_source(f"mcp:{name}")

    async def _stop_server_process(self, server: MCPServer) -> None:
        """停止单个 MCP server 进程并清理读写后台任务。

        参数:
            server: 要停止的 server 状态对象。
        """
        self._fail_pending(server, RuntimeError(f"MCP server {server.name} stopped"))
        tasks = [task for task in (server.stdout_reader_task, server.stderr_reader_task) if task]
        if server.process and server.process.returncode is None:
            server.process.terminate()
            try:
                await asyncio.wait_for(server.process.wait(), timeout=3)
            except TimeoutError:
                server.process.kill()
                with contextlib.suppress(Exception):
                    await server.process.wait()
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        server.stdout_reader_task = None
        server.stderr_reader_task = None

    def status(self) -> str:
        """返回当前 MCP server 连接状态的文本摘要。"""
        if not self.servers:
            return "未连接 MCP server。"
        return "\n".join(f"- {name}: tools={len(server.tools)}" for name, server in self.servers.items())

    async def _start_server(self, server: MCPServer) -> None:
        """启动单个 MCP server，初始化协议并注册其工具。

        参数:
            server: 待启动的 server 状态对象。
        """
        server.process = await asyncio.create_subprocess_exec(
            *server.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, **server.env},
        )
        server.stdout_reader_task = asyncio.create_task(self._stdout_reader(server), name=f"mcp-stdout-{server.name}")
        server.stderr_reader_task = asyncio.create_task(self._stderr_reader(server), name=f"mcp-stderr-{server.name}")
        await self._request(
            server,
            "initialize",
            {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "telegram-agent", "version": "0.1"}},
        )
        tools_result = await self._request(server, "tools/list", {})
        server.tools = tools_result.get("tools", [])
        for tool_def in server.tools:
            self._register_mcp_tool(server.name, tool_def)
        logger.info("MCP server connected name=%s tools=%s", server.name, len(server.tools))

    def _register_mcp_tool(self, server_name: str, tool_def: dict[str, Any]) -> None:
        """注册`mcp`、工具。

        参数:
            server_name: 参与注册`mcp`、工具的 `server_name` 参数。
            tool_def: 参与注册`mcp`、工具的 `tool_def` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        name = str(tool_def["name"])
        original_name = f"{server_name}:{name}"
        if self.allowed_tools and original_name not in self.allowed_tools:
            logger.info("Skipping MCP tool by allowlist server=%s tool=%s", server_name, name)
            return

        registered_name = _safe_tool_name(f"{server_name}_{name}")
        description = str(tool_def.get("description", f"MCP tool {name}"))
        schema = _normalize_mcp_schema(server_name, name, tool_def.get("inputSchema") or {"type": "object", "properties": {}})

        async def call_mcp(_: ToolContext, args: dict[str, Any], *, _server=server_name, _tool=name) -> str:
            """处理`mcp`。

            参数:
                _: 参与处理`mcp`的 `_` 参数。
                args: 参与处理`mcp`的 `args` 参数。

            返回:
                返回与本函数处理结果对应的数据。
            """
            started = time.perf_counter()
            error = None
            result_text = ""
            try:
                result = await self.call_tool(_server, _tool, args)
                result_text = json.dumps(result, ensure_ascii=False)
                return result_text
            except Exception as exc:
                error = str(exc)
                if _is_transient_search_failure(_server, _tool, error):
                    degraded = _build_degraded_search_result(_server, args, error)
                    result_text = json.dumps(degraded, ensure_ascii=False)
                    logger.info("MCP search degraded to empty results server=%s tool=%s error=%s", _server, _tool, error)
                    return result_text
                logger.warning("MCP tool failed server=%s tool=%s error=%s", _server, _tool, error)
                return (
                    f"MCP 工具 {_server}/{_tool} 调用失败：{error}。\n"
                    "如果这是搜索、抓取或实时信息任务，请换用其他可见搜索工具继续尝试，"
                    "例如 duckduckgo_web_search；不要直接声称没有联网能力。"
                )
            finally:
                await self.store.add_mcp_tool_log(
                    _server,
                    _tool,
                    json.dumps(args, ensure_ascii=False),
                    result_text,
                    int((time.perf_counter() - started) * 1000),
                    error,
                )

        exposure = "always" if _is_search_tool(server_name, name, description) else "hidden"
        risk = _infer_tool_risk(name, description)
        self.tool_registry.register(
            Tool(registered_name, description, schema, call_mcp, exposure=exposure, risk=risk, source=f"mcp:{server_name}")
        )
        logger.info(
            "Registered MCP tool name=%s original=%s source=mcp:%s exposure=%s risk=%s",
            registered_name,
            name,
            server_name,
            exposure,
            risk,
        )

    async def call_tool(self, server_name: str, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """处理工具。

        参数:
            server_name: 参与处理工具的 `server_name` 参数。
            tool_name: 参与处理工具的 `tool_name` 参数。
            args: 参与处理工具的 `args` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        server = self.servers[server_name]
        if self.allowed_tools and f"{server_name}:{tool_name}" not in self.allowed_tools:
            raise RuntimeError(f"MCP tool not allowed: {server_name}:{tool_name}")
        sanitized_args = _sanitize_mcp_args(server_name, tool_name, args)
        result = await self._request(server, "tools/call", {"name": tool_name, "arguments": sanitized_args})
        return _normalize_mcp_result(server_name, tool_name, sanitized_args, result)

    async def _request(self, server: MCPServer, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """处理相关逻辑。

        参数:
            server: 参与处理相关逻辑的 `server` 参数。
            method: 参与处理相关逻辑的 `method` 参数。
            params: 参与处理相关逻辑的 `params` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        if not server.process or not server.process.stdin:
            raise RuntimeError(f"MCP server {server.name} is not running")

        loop = asyncio.get_running_loop()
        async with server.write_lock:
            server.request_id += 1
            request_id = server.request_id
            request = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
            future: asyncio.Future[dict[str, Any]] = loop.create_future()
            server.pending_requests[request_id] = future
            try:
                server.process.stdin.write((json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8"))
                await server.process.stdin.drain()
            except Exception:
                server.pending_requests.pop(request_id, None)
                raise
        try:
            return await asyncio.wait_for(future, timeout=20)
        except Exception:
            server.pending_requests.pop(request_id, None)
            raise

    async def _stdout_reader(self, server: MCPServer) -> None:
        """处理`reader`。

        参数:
            server: 参与处理`reader`的 `server` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        try:
            if not server.process or not server.process.stdout:
                return
            while True:
                line = await server.process.stdout.readline()
                if not line:
                    raise RuntimeError(f"MCP server {server.name} closed stdout")
                text = line.decode("utf-8", errors="replace").strip()
                if not text:
                    continue
                try:
                    response = json.loads(text)
                except json.JSONDecodeError:
                    log = _classify_non_json_stdout(server.name, text)
                    log("Ignoring non-JSON MCP stdout server=%s line=%r", server.name, text[:300])
                    continue
                request_id = response.get("id")
                if request_id is None:
                    logger.debug("Ignoring MCP message without id server=%s message=%s", server.name, text[:300])
                    continue
                future = server.pending_requests.pop(int(request_id), None)
                if future is None:
                    logger.debug("Ignoring unrelated MCP message server=%s message=%s", server.name, text[:300])
                    continue
                if "error" in response:
                    future.set_exception(RuntimeError(response["error"]))
                else:
                    future.set_result(response.get("result", {}))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._fail_pending(server, exc)

    async def _stderr_reader(self, server: MCPServer) -> None:
        """处理`reader`。

        参数:
            server: 参与处理`reader`的 `server` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        try:
            if not server.process or not server.process.stderr:
                return
            while True:
                line = await server.process.stderr.readline()
                if not line:
                    return
                text = line.decode("utf-8", errors="replace").strip()
                if text:
                    log = _classify_stderr_log(server.name, text)
                    log("MCP stderr server=%s line=%r", server.name, text[:300])
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("MCP stderr reader failed server=%s", server.name)

    def _fail_pending(self, server: MCPServer, error: Exception) -> None:
        """处理`pending`。

        参数:
            server: 参与处理`pending`的 `server` 参数。
            error: 参与处理`pending`的 `error` 参数。

        返回:
            返回与本函数处理结果对应的数据。
        """
        for future in server.pending_requests.values():
            if not future.done():
                future.set_exception(error)
        server.pending_requests.clear()


def _expand_env(env: dict[str, str]) -> dict[str, str] | None:
    """展开`env`。

    参数:
        env: 参与展开`env`的 `env` 参数。

    返回:
        返回与本函数处理结果对应的数据。
    """
    expanded: dict[str, str] = {}
    for key, value in env.items():
        text = str(value)

        def replace(match: re.Match[str]) -> str:
            """把 `${ENV_NAME}` 形式的占位符替换成真实环境变量值。"""
            name = match.group(1)
            if name not in os.environ:
                raise KeyError(name)
            return os.environ[name]

        try:
            expanded[key] = ENV_PATTERN.sub(replace, text)
        except KeyError as exc:
            logger.warning("MCP env %s references missing environment variable %s", key, exc.args[0])
            return None
    return expanded


def _is_search_tool(server_name: str, tool_name: str, description: str) -> bool:
    """判断 MCP 工具是否属于默认可见的搜索类工具。

    参数:
        server_name: MCP server 名称。
        tool_name: 原始 MCP 工具名。
        description: 工具描述文本。

    返回:
        True 表示该工具适合默认暴露给模型。
    """
    text = f"{server_name} {tool_name} {description}".lower()
    if server_name == "duckduckgo":
        return tool_name in {"web-search", "web_search"}
    keywords = ("web_search", "search_web", "news_search")
    return any(keyword in text for keyword in keywords)


def _infer_tool_risk(tool_name: str, description: str) -> str:
    """根据工具名和描述推断工具风险等级。

    参数:
        tool_name: 原始 MCP 工具名。
        description: 工具描述文本。

    返回:
        read 或 side_effect。
    """
    text = f"{tool_name} {description}".lower()
    write_keywords = ("write", "create", "update", "delete", "remove", "ack", "send", "post", "publish")
    return "side_effect" if any(keyword in text for keyword in write_keywords) else "read"


def _safe_tool_name(name: str) -> str:
    """把 MCP 原始工具名转换成合法的 Python/OpenAI 工具名。

    参数:
        name: 可能包含连字符、冒号或其他符号的原始名称。

    返回:
        只包含字母、数字和下划线的安全名称。
    """
    safe = TOOL_NAME_PATTERN.sub("_", name)
    safe = re.sub(r"_+", "_", safe).strip("_")
    if not safe:
        return "mcp_tool"
    if safe[0].isdigit():
        safe = f"mcp_{safe}"
    return safe[:64]


def _normalize_mcp_schema(server_name: str, tool_name: str, schema: dict[str, Any]) -> dict[str, Any]:
    """按项目约定修正 MCP 工具的参数 schema。

    参数:
        server_name: MCP server 名称。
        tool_name: 原始 MCP 工具名。
        schema: server 返回的 inputSchema。

    返回:
        可直接提供给模型的 JSON Schema。
    """
    if server_name == "duckduckgo" and tool_name in {"web-search", "web_search"}:
        properties = dict(schema.get("properties", {}))
        properties["numResults"] = {
            "type": "integer",
            "description": "How many search results to return (1-5).",
            "default": 5,
            "minimum": 1,
            "maximum": 5,
        }
        properties["mode"] = {
            "type": "string",
            "description": "Search returns summary results only; use fetch_page/web_fetch to read a page in detail.",
            "enum": ["short"],
            "default": "short",
        }
        normalized = dict(schema)
        normalized["properties"] = properties
        return normalized
    return schema


def _sanitize_mcp_args(server_name: str, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """在调用 MCP server 前清洗和限制工具参数。

    参数:
        server_name: MCP server 名称。
        tool_name: 原始 MCP 工具名。
        args: 模型传入的原始参数。

    返回:
        清洗后的参数字典。
    """
    if server_name == "duckduckgo" and tool_name in {"web-search", "web_search"}:
        query = str(args.get("query", "")).strip()
        try:
            num_results = int(args.get("numResults", args.get("limit", 5)) or 5)
        except (TypeError, ValueError):
            num_results = 5
        num_results = max(1, min(num_results, 5))
        return {"query": query, "numResults": num_results, "mode": "short"}
    return args


def _classify_non_json_stdout(server_name: str, text: str):
    """给 MCP stdout 中的非 JSON 文本选择合适日志级别。

    参数:
        server_name: 参与处理`non`、`json`、`stdout`的 `server_name` 参数。
        text: 参与处理`non`、`json`、`stdout`的 `text` 参数。
    """
    if server_name == "duckduckgo":
        lowered = text.lower()
        if lowered.startswith("searching ") or lowered.startswith("found "):
            return logger.debug
    return logger.warning


def _classify_stderr_log(server_name: str, text: str):
    """给 MCP stderr 文本选择合适日志级别，过滤已知噪音。

    参数:
        server_name: 参与处理`stderr`、`log`的 `server_name` 参数。
        text: 参与处理`stderr`、`log`的 `text` 参数。
    """
    if server_name == "duckduckgo":
        lowered = text.lower()
        if "duckduckgo & ai search mcp" in lowered or "youtube.com/@oevortex" in lowered:
            return logger.info
        if "started and listening on stdio" in lowered:
            return logger.info
        if "http 202: failed to fetch search results" in lowered:
            return logger.info
        if "error handling web-search tool call" in lowered and "http 202" in lowered:
            return logger.info
    return logger.warning


def _is_transient_search_failure(server_name: str, tool_name: str, error: str) -> bool:
    """判断搜索工具失败是否属于可降级的临时故障。

    参数:
        server_name: MCP server 名称。
        tool_name: 原始 MCP 工具名。
        error: 捕获到的错误文本。

    返回:
        True 表示可返回 degraded 空搜索结果，而不是直接中断工具循环。
    """
    if _tool_kind(server_name, tool_name) != "search":
        return False
    lowered = error.lower()
    return any(
        token in lowered
        for token in (
            "http 202",
            "failed to fetch search results",
            "timed out",
            "timeout",
            "connection reset",
            "temporarily unavailable",
        )
    )


def _build_degraded_search_result(server_name: str, args: dict[str, Any], error: str) -> dict[str, Any]:
    """构建搜索后端临时不可用时的降级结果。

    参数:
        server_name: 参与构建`degraded`、`search`、结果的 `server_name` 参数。
        args: 参与构建`degraded`、`search`、结果的 `args` 参数。
        error: 参与构建`degraded`、`search`、结果的 `error` 参数。

    返回:
        返回与本函数处理结果对应的数据。
    """
    query = str(args.get("query", "")).strip()
    return {
        "query": query,
        "results": [],
        "provider": server_name,
        "error": error,
        "degraded": True,
    }


def _normalize_mcp_result(server_name: str, tool_name: str, args: dict[str, Any], result: Any) -> dict[str, Any]:
    """按工具类型把 MCP 返回值归一化成项目内部结构。

    参数:
        server_name: 参与归一化`mcp`、结果的 `server_name` 参数。
        tool_name: 参与归一化`mcp`、结果的 `tool_name` 参数。
        args: 参与归一化`mcp`、结果的 `args` 参数。
        result: 参与归一化`mcp`、结果的 `result` 参数。

    返回:
        返回与本函数处理结果对应的数据。
    """
    kind = _tool_kind(server_name, tool_name)
    if kind == "search":
        return _normalize_search_result(server_name, args, result)
    if kind == "fetch":
        return _normalize_fetch_result(server_name, args, result)
    if kind == "feed":
        return _normalize_feed_result(server_name, result)
    return result if isinstance(result, dict) else {"content": result}


def _tool_kind(server_name: str, tool_name: str) -> str:
    """判断 MCP 工具的业务类型：搜索、网页抓取、feed 或普通工具。

    参数:
        server_name: 参与处理`kind`的 `server_name` 参数。
        tool_name: 参与处理`kind`的 `tool_name` 参数。

    返回:
        返回与本函数处理结果对应的数据。
    """
    if server_name == "duckduckgo" and tool_name in {"web-search", "web_search"}:
        return "search"
    if server_name == "web_content" and tool_name == "fetch_page":
        return "fetch"
    if server_name == "feed_bridge" and tool_name == "get_proactive_events":
        return "feed"
    if server_name == "rss" and tool_name == "get_content":
        return "feed"
    return ""


def _normalize_search_result(server_name: str, args: dict[str, Any], result: Any) -> dict[str, Any]:
    """归一化`search`、结果。

    参数:
        server_name: 参与归一化`search`、结果的 `server_name` 参数。
        args: 参与归一化`search`、结果的 `args` 参数。
        result: 参与归一化`search`、结果的 `result` 参数。

    返回:
        返回与本函数处理结果对应的数据。
    """
    query = str(args.get("query", "")).strip()
    if isinstance(result, dict) and isinstance(result.get("results"), list):
        rows = [_normalize_search_row(item, server_name) for item in result.get("results", [])]
        return {"query": query or str(result.get("query", "")), "results": [row for row in rows if row], "provider": server_name}

    content_items = result.get("content", []) if isinstance(result, dict) else []
    if isinstance(content_items, str):
        text = content_items
    else:
        text = "\n".join(
            str(item.get("text", ""))
            for item in content_items
            if isinstance(item, dict) and item.get("type") == "text"
        )
    rows = _parse_search_text(text, server_name)
    normalized: dict[str, Any] = {"query": query, "results": rows, "provider": server_name}
    if text and not rows:
        normalized["raw_text"] = text[:4000]
    return normalized


def _normalize_search_row(item: Any, provider: str) -> dict[str, str] | None:
    """归一化`search`、`row`。

    参数:
        item: 参与归一化`search`、`row`的 `item` 参数。
        provider: 参与归一化`search`、`row`的 `provider` 参数。

    返回:
        返回与本函数处理结果对应的数据。
    """
    if not isinstance(item, dict):
        return None
    title = str(item.get("title") or item.get("name") or "").strip()
    url = str(item.get("url") or item.get("link") or item.get("href") or "").strip()
    snippet = str(item.get("snippet") or item.get("description") or item.get("summary") or "").strip()
    source = str(item.get("source") or provider).strip() or provider
    if not (title or url or snippet):
        return None
    return {"title": title, "url": url, "snippet": snippet, "source": source}


def _parse_search_text(text: str, provider: str) -> list[dict[str, str]]:
    """解析`search`、文本。

    参数:
        text: 参与解析`search`、文本的 `text` 参数。
        provider: 参与解析`search`、文本的 `provider` 参数。

    返回:
        返回与本函数处理结果对应的数据。
    """
    if not text.strip():
        return []

    matches = list(re.finditer(r"(?ms)^\s*(\d+)\.\s+\*\*(.*?)\*\*\s*\n(.*?)(?=^\s*\d+\.\s+\*\*|\Z)", text))
    if not matches:
        return []

    results: list[dict[str, str]] = []
    for match in matches:
        title = match.group(2).strip()
        body = match.group(3)
        data: dict[str, str] = {"title": title, "url": "", "snippet": "", "source": provider}
        for raw_line in body.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("URL:"):
                data["url"] = line.removeprefix("URL:").strip()
            elif line.startswith("Snippet:"):
                data["snippet"] = line.removeprefix("Snippet:").strip()
            elif line.startswith("Content:") and not data["snippet"]:
                data["snippet"] = line.removeprefix("Content:").strip()[:500]
        if data["title"] or data["url"] or data["snippet"]:
            results.append(data)
    return results


def _normalize_fetch_result(server_name: str, args: dict[str, Any], result: Any) -> dict[str, Any]:
    """归一化抓取结果、结果。

    参数:
        server_name: 参与归一化抓取结果、结果的 `server_name` 参数。
        args: 参与归一化抓取结果、结果的 `args` 参数。
        result: 参与归一化抓取结果、结果的 `result` 参数。

    返回:
        返回与本函数处理结果对应的数据。
    """
    if isinstance(result, dict):
        return {
            "url": str(result.get("url") or args.get("url") or ""),
            "title": str(result.get("title") or result.get("url") or args.get("url") or ""),
            "content": str(result.get("content") or result.get("text") or ""),
            "provider": server_name,
        }
    return {"url": str(args.get("url") or ""), "title": str(args.get("url") or ""), "content": str(result or ""), "provider": server_name}


def _normalize_feed_result(server_name: str, result: Any) -> dict[str, Any]:
    """归一化feed 数据、结果。

    参数:
        server_name: 参与归一化feed 数据、结果的 `server_name` 参数。
        result: 参与归一化feed 数据、结果的 `result` 参数。

    返回:
        返回与本函数处理结果对应的数据。
    """
    if isinstance(result, dict) and "events" in result:
        events = result.get("events", [])
    elif isinstance(result, dict):
        events = result.get("articles") or result.get("items") or result.get("content") or []
    elif isinstance(result, list):
        events = result
    else:
        events = []

    normalized_events = [_normalize_event(item, server_name) for item in _ensure_iterable(events)]
    payload = {"events": [item for item in normalized_events if item], "provider": server_name}
    if isinstance(result, dict) and "channel" in result:
        payload["channel"] = str(result.get("channel") or "")
    return payload


def _normalize_event(item: Any, provider: str) -> dict[str, str] | None:
    """归一化事件。

    参数:
        item: 参与归一化事件的 `item` 参数。
        provider: 参与归一化事件的 `provider` 参数。

    返回:
        返回与本函数处理结果对应的数据。
    """
    if not isinstance(item, dict):
        return None
    title = str(item.get("title") or item.get("name") or item.get("headline") or "").strip()
    url = str(item.get("url") or item.get("link") or item.get("href") or "").strip()
    event_id = str(item.get("event_id") or item.get("id") or item.get("guid") or url or title).strip()
    image_url = str(
        item.get("image_url")
        or item.get("image")
        or item.get("thumbnail")
        or item.get("thumbnail_url")
        or item.get("cover_image")
        or ""
    ).strip()
    return {
        "event_id": event_id,
        "title": title or "Untitled",
        "url": url,
        "source": str(item.get("source") or item.get("feedTitle") or provider).strip() or provider,
        "content": str(item.get("content") or item.get("description") or item.get("summary") or "").strip(),
        "published_at": str(item.get("published_at") or item.get("pubDate") or item.get("published") or item.get("updated") or "").strip(),
        "image_url": image_url,
    }


def _ensure_iterable(value: Any) -> Iterable[Any]:
    """校验`iterable`。

    参数:
        value: 参与校验`iterable`的 `value` 参数。

    返回:
        返回与本函数处理结果对应的数据。
    """
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return value
    return []
