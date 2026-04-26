---
name: summarize
description: 总结长文本、网页、文件、RSS 条目、Telegram 图片说明或对话历史。用户要求总结、提炼要点、摘要、TL;DR 时使用。
metadata: {"chat_agent":{"always":false,"drift":true,"requires":{"bins":[],"env":[]}}}
---

# 总结与提炼

当前项目主要依靠 LLM、`web_fetch`、文件工具和 MCP 工具完成总结，不依赖外部 `summarize` CLI。

## 触发场景

- 用户发来长文本，希望总结重点。
- 用户发来 URL，希望了解网页内容。
- 用户要求“提炼要点”“TL;DR”“总结这篇文章/对话/资料”。
- Drift 空闲时整理网页、feed 或长期上下文。

## 工具策略

1. 如果输入是普通文本，直接总结。
2. 如果输入是 URL：
   - 先用 `tool_search(query="web fetch")` 找到网页工具。
   - 优先用 `web_fetch` 或 MCP 的 `web_fetch_page` 获取正文。
   - 抓取失败时说明限制，不要编造网页内容。
3. 如果是 workspace 文件：
   - 用 `read_file(path, max_chars)` 读取。
4. 如果是最新信息、新闻、价格、天气：
   - 先用搜索/MCP 工具获取来源，再总结。

## 输出格式

默认使用简洁中文：

```text
核心结论：...

要点：
1. ...
2. ...
3. ...

可继续追问：...
```

如果是主动 drift 输出，第一行必须是适合 Telegram 主动推送的短句，后续再写 Markdown 归档。

## 注意

- 区分“来自工具的内容”和“模型推断”。
- 不要输出大段原文。
- 来源不可靠或抓取失败时，要明确说明。
