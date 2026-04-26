"""网页抓取 MCP 示例服务。"""
from __future__ import annotations

import re
from typing import Any

from _common import compact_text, http_get, serve


def fetch_page(url: str) -> dict[str, Any]:
    """抓取`page`。

    参数:
        url: 参与抓取`page`的 `url` 参数。

    返回:
        返回与本函数处理结果对应的数据。
    """
    raw = http_get(url)
    title_match = re.search(r"<title[^>]*>(.*?)</title>", raw, flags=re.I | re.S)
    title = compact_text(title_match.group(1), 300) if title_match else url
    return {"url": url, "title": title, "content": compact_text(raw, 8000)}


TOOLS = [
    {
        "name": "fetch_page",
        "description": "读取网页正文预览。参数 url 是公开的 http 或 https 链接。",
        "inputSchema": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    }
]


def main() -> None:
    """启动相关逻辑。

    返回:
        返回与本函数处理结果对应的数据。"""
    serve(
        "web_content",
        "0.2.0",
        TOOLS,
        {"fetch_page": lambda args: fetch_page(str(args["url"]))},
    )


if __name__ == "__main__":
    main()
