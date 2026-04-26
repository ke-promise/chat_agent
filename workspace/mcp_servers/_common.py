"""本地 MCP 示例服务的公共辅助函数。"""
from __future__ import annotations

import html
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chat_agent.url_safety import ensure_public_http_url

for stream in (sys.stdin, sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

Handler = Callable[[dict[str, Any]], Any]


def respond(req: dict[str, Any], result: Any | None = None, error: str | None = None) -> None:
    """按 JSON-RPC 2.0 格式把结果写回标准输出。"""
    payload: dict[str, Any] = {"jsonrpc": "2.0", "id": req.get("id")}
    if error:
        payload["error"] = error
    else:
        payload["result"] = result if result is not None else {}
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def http_get(url: str, limit: int = 300_000) -> str:
    """抓取网页内容并限制最大读取字节数。

    参数:
        url: 目标 URL，会先经过公网安全校验。
        limit: 最多读取多少字节，避免 MCP server 被超大页面拖垮。
    """
    safe_url = ensure_public_http_url(url)
    req = urllib.request.Request(
        safe_url,
        headers={
            "User-Agent": "chat-agent-web-mcp/0.2",
            "Accept": "text/html,application/xhtml+xml,application/xml,text/xml;q=0.9,*/*;q=0.8",
        },
    )
    with urllib.request.urlopen(req, timeout=15) as response:
        body = response.read(limit)
        charset = response.headers.get_content_charset() or "utf-8"
        return body.decode(charset, errors="replace")


def compact_text(text: str, limit: int = 6000) -> str:
    """移除 HTML 标签并压缩空白，生成适合返回给模型的纯文本摘要。"""
    text = re.sub(r"<script\b.*?</script>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<style\b.*?</style>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def serve(server_name: str, version: str, tools: list[dict[str, Any]], handlers: dict[str, Handler]) -> None:
    """运行一个最小 MCP stdio 服务循环。

    参数:
        server_name: 对外暴露的服务名称。
        version: 当前服务版本号。
        tools: `tools/list` 接口返回的工具描述列表。
        handlers: 工具名到处理函数的映射表。
    """
    for line in sys.stdin:
        req = json.loads(line)
        method = req.get("method")
        params = req.get("params") or {}
        try:
            if method == "initialize":
                respond(req, {"serverInfo": {"name": server_name, "version": version}})
            elif method == "tools/list":
                respond(req, {"tools": tools})
            elif method == "tools/call":
                name = str(params.get("name") or "")
                args = params.get("arguments") or {}
                handler = handlers.get(name)
                if not handler:
                    respond(req, error=f"unknown tool: {name}")
                    continue
                respond(req, handler(args))
            else:
                respond(req, {})
        except Exception as exc:
            respond(req, error=str(exc))
