---
name: feed-manage
description: 管理和查询主动信息源、RSS、网页 feed、proactive source。用户问订阅了什么、有哪些动态、信息来源、RSS 配置时使用。
metadata: {"chat_agent":{"always":false,"drift":false,"triggers":["信息源","订阅","RSS","rss","feed","主动推送来源","监控什么","有哪些动态"],"requires":{"bins":[],"env":[],"tools":["tool_search","read_file"]}}}
---

# Feed 管理

当前项目通过 MCP 和 `workspace/proactive_sources.json` 接入主动信息源。不要使用 Akashic 原项目的 `mcp_feed__*` 工具名，也不要假设存在 `web_poll_feeds` 这类固定工具名。

## 触发场景

- 用户问“你有什么信息来源”“你在监控什么”“订阅了什么”。
- 用户要查看最近 feed 内容、RSS 动态、主动推送来源。
- 用户要增加、删除、启用或禁用 proactive source。

## 工具策略

1. 先调用 `tool_search`，搜索关键词可以是 `feed`、`rss`、`proactive`、`source`。
2. 如果已连接 MCP，当前项目常见工具名可能是：
   - `feed_bridge_poll_feeds`
   - `feed_bridge_get_proactive_events`
   - `feed_bridge_ack_events`
   - `rss_get_content`
3. 具体工具名以 `tool_search` 返回和当前可见工具为准，不要硬编码不存在的工具。
4. 默认文件工具只能访问 `[tools].file_workspace`，通常是 `workspace/files`；因此运行中的 Agent 通常不能直接读取或修改 `workspace/proactive_sources.json` 和 `workspace/mcp_servers.json`。
5. 如果文件工作区已被配置为允许访问对应 JSON，才使用 `read_file` / `write_file` 查看或修改；否则给用户明确的配置片段和需要修改的文件路径。

## 查询最近内容

优先使用已解锁的 MCP 工具获取最新状态，不要凭历史记忆回答。

示例意图：

```text
tool_search(query="feed rss proactive source")
feed_bridge_poll_feeds()
feed_bridge_get_proactive_events(channel="content")
```

如果 `feed_bridge_get_proactive_events` 返回空，说明当前可能没有新内容；不等于配置错误。

## 管理 source

`workspace/proactive_sources.json` 的基本结构：

```json
{
  "sources": [
    {
      "server": "feed_bridge",
      "channel": "content",
      "poll_tool": "poll_feeds",
      "get_tool": "get_proactive_events",
      "ack_tool": "ack_events",
      "enabled": true
    }
  ]
}
```

常见搭配：

- RSS/Atom：优先复用 `feed_bridge`。
- 第三方 `rss` MCP：可能使用 `rss_get_content`，通常没有 ACK。
- 自定义网页/API：先参考 `create-proactive-source` skill，再决定是否新增 MCP server。

## 注意

- source 配置不保存 Telegram token、QQ secret、LLM API key 或 cookie。
- 读到空 feed 不代表错误，可能只是暂时没有新内容。
- 修改 MCP 或 proactive source 配置后，提醒用户执行 `/mcp_reload` 或重启 agent。
- 主动发送仍受 daily cap、cooldown、quiet hours、presence busy、dedupe 等限制。
