from __future__ import annotations

from chat_agent.reply_format import format_reply


def test_format_reply_cleans_news_markdown_and_spacing() -> None:
    raw = """'这是为您整理的今天（2026年4月22日 ）三条热点新闻：

1.  **国内油价迎年内首降** 📉
    国内成品油价格终结了此前的“六连涨”格局，迎来2026年首次下调。

2.  **国务院发文推动服务业扩能提 质** 📝
    明确提出到2030年，我国服务业总规模要迈上100万亿元台阶，并致 力于培育更多“中国服务”品牌。'"""

    formatted = format_reply(raw)

    assert not formatted.startswith("'")
    assert "**" not in formatted
    assert "提质" in formatted
    assert "致力" in formatted
    assert "\n\n1. 国内油价迎年内首降" in formatted
    assert "\n\n\n" not in formatted
