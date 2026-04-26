---
name: feed-manage
description: 管理和查询主动信息源、RSS、网页 feed、proactive source。用户问订阅了什么、有哪些动态、信息来源、RSS 配置时使用。
metadata: {"chat_agent":{"always":false,"drift":false,"requires":{"bins":[],"env":[]}}}
---

# Feed 管理

当前项目通过 MCP 和 `workspace/proactive_sources.json` 接入主动信息源。不要使用 Akashic 原项目的 `mcp_feed__*` 工具名。

## 触发场景

- 用户问“你有什么信息来源”“你在监控什么”“订阅了什么”。
- 用户要查看最近 feed 内容、RSS 动态、主动推送来源。
- 用户要增加、删除、启用或禁用 proactive source。

## 工具策略

1. 先调用 `tool_search`，搜索关键词可以是 `feed`、`rss`、`proactive`、`source`。
2. 如果已连接 MCP，可能看到类似工具：
   - `web_poll_feeds`
   - `web_get_proactive_events`
   - `web_ack_events`
   - `rss_get_content`
3. 如果只是查看配置，可用文件工具读取：
   - `read_file("proactive_sources.json")`
   - `read_file("mcp_servers.json")`
4. 如果要修改配置，使用 `write_file` 更新 workspace 内对应 JSON，并提醒用户需要重启或执行 `/mcp_reload`。

## 查询最近内容

优先使用已解锁的 MCP 工具获取最新状态，不要凭历史记忆回答。

示例意图：

```text
tool_search(query="feed rss proactive source")
web_poll_feeds()
web_get_proactive_events()
```

具体工具名以当前 `tool_search` 返回为准。

## 管理 source

`workspace/proactive_sources.json` 的基本结构：

```json
{
  "sources": [
    {
      "server": "web",
      "channel": "content",
      "poll_tool": "poll_feeds",
      "get_tool": "get_proactive_events",
      "ack_tool": "ack_events",
      "enabled": true
    }
  ]
}
```

## 注意

- source 配置不保存 Telegram token 或 LLM API key。
- 读到空 feed 不代表错误，可能只是暂时没有新内容。
- 主动发送仍受 `daily_max`、cooldown、presence busy 限制。
