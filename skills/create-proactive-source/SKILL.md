---
name: create-proactive-source
description: 创建或更新当前 chat_agent 项目的主动信息源 MCP server，并注册到 proactive_sources.json。当用户要新增主动推送的数据来源、RSS/feed 或网页监控时使用。
metadata: {"chat_agent":{"always":false,"drift":false,"triggers":["新增主动信息源","创建 proactive source","接入 RSS","接入 feed","主动推送来源","网页监控","订阅信息源"],"requires":{"bins":[],"env":[],"tools":["read_file","write_file","tool_search"]}}}
---

# 创建 Proactive 信息源

## 目标

把一个新的内容来源接入当前项目的 proactive 主动推送系统。产出通常包括：

- 一个位于 `workspace/mcp_servers/` 下的 MCP server 脚本
- `workspace/mcp_servers.json` 中的 server 启动配置
- `workspace/proactive_sources.json` 中的 source 条目
- 必要时更新 `workspace/rss_feeds.opml` 或 server 专用配置文件

当前项目的主动 feed 链路由 `chat_agent.proactive.feed.ProactiveFeedManager` 读取 `workspace/proactive_sources.json`，通过 MCP 调用 `poll_tool`、`get_tool` 和 `ack_tool`，再把事件转成 `ProactiveCandidate`。

## 触发场景

- 用户想订阅新的 RSS/Atom feed。
- 用户想监控某个网页、API、公告页或服务动态。
- 用户想把已有 MCP server 的内容接入主动推送。
- 用户想查看、修改、启用或禁用主动信息源配置。

## 当前项目约定

主要文件：

```text
workspace/mcp_servers.json              # MCP server 启动配置
workspace/proactive_sources.json        # proactive feed source 配置
workspace/mcp_servers/                  # 本项目自带或用户扩展的 MCP server 脚本
workspace/rss_feeds.opml                # RSS/Atom feed 列表，可选
```

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

字段含义：

- `server`：必须对应 `workspace/mcp_servers.json` 里的 server 名。
- `channel`：当前主要使用 `"content"`。
- `poll_tool`：可选，定时拉取或刷新外部数据。
- `get_tool`：必填，返回候选内容列表。
- `ack_tool`：可选，用于确认已处理事件，避免重复推送。
- `poll_args` / `get_args`：可选，固定传给工具的参数。
- `enabled`：是否启用。

## 返回数据协议

`get_tool` 返回值可以是 dict 或 list。当前 `ProactiveFeedManager` 会兼容这些形态：

```json
{"events": [...]}
```

```json
{"articles": [...]}
```

```json
{"items": [...]}
```

```json
[
  {
    "event_id": "stable-id",
    "title": "标题",
    "url": "https://example.com/item",
    "source": "来源名",
    "content": "摘要或正文片段",
    "published_at": "2026-04-29T12:00:00Z",
    "image_url": "https://example.com/cover.jpg"
  }
]
```

推荐每条事件包含：

- `event_id`：稳定唯一 ID；没有时系统会退回用 `id`、`guid`、`url` 或 `title`。
- `title`：标题。
- `url`：原文链接，用于去重和展示。
- `source` 或 `feedTitle`：来源名。
- `content` / `description` / `summary`：摘要文本。
- `published_at` / `pubDate` / `published` / `updated`：发布时间。
- `image_url` / `image` / `thumbnail`：可选封面图。

## 创建流程

1. 先读取现有配置：
   - 默认文件工具通常只能访问 `workspace/files`，不能直接读取 `workspace/mcp_servers.json`、`workspace/proactive_sources.json` 或 `workspace/rss_feeds.opml`
   - 如果当前环境的文件工具已允许访问这些配置，再读取对应 JSON/OPML
   - 否则请用户贴出配置内容，或给出需要手工写入的配置片段
2. 判断是否可以复用现有 server：
   - RSS/Atom 优先复用 `feed_bridge`
   - 纯网页正文抓取优先考虑已有 `web_content`
   - 只有协议特殊、需要登录或需要自定义解析时，才新增 MCP server
3. 如果只是新增 RSS/Atom：
   - 更新 `workspace/rss_feeds.opml` 或 `feed_bridge` 的环境配置
   - 确认 `workspace/proactive_sources.json` 已有 `feed_bridge` source
4. 如果需要新增 MCP server：
   - 在 `workspace/mcp_servers/<name>_mcp_server.py` 创建脚本
   - 实现 `tools/list` 和 `tools/call`
   - 至少提供 `get_tool`；需要刷新时提供 `poll_tool`；需要去重 ACK 时提供 `ack_tool`
   - 在 `workspace/mcp_servers.json` 注册启动命令
   - 在 `workspace/proactive_sources.json` 添加 source 条目
5. 修改 JSON 时保持原有 source，不要覆盖用户已有配置；如果当前 Agent 没有写入这些路径的权限，就输出明确的 JSON patch/片段让用户或开发侧应用。

## MCP Server 最小形态

优先参考 `workspace/mcp_servers/feed_bridge_mcp_server.py` 和 `workspace/mcp_servers/web_content_mcp_server.py` 的 stdio JSON-RPC 写法。

最小工具集合：

```text
tools/list
tools/call
```

推荐工具：

```text
poll_feeds              # 可选，刷新外部数据
get_proactive_events    # 必填，返回 events/articles/items/list
ack_events              # 可选，确认已处理 event_id
```

`get_proactive_events(channel="content")` 可以返回：

```json
{
  "channel": "content",
  "events": [
    {
      "event_id": "example-001",
      "title": "Example update",
      "url": "https://example.com/update",
      "source": "Example",
      "content": "一小段摘要",
      "published_at": "2026-04-29T12:00:00Z"
    }
  ]
}
```

`ack_events(event_ids)` 如果实现，应把 ACK 状态持久化到 `workspace/` 下的 JSON 文件，避免 agent 重启后重复推送。

## 本地验证

完成配置或 server 变更后，至少验证这些点：

1. JSON 文件可解析：
   - `workspace/mcp_servers.json`
   - `workspace/proactive_sources.json`
2. 新增 server 命令能启动，并能响应 `tools/list`。
3. `get_tool` 返回 dict/list，且事件能被 `ProactiveFeedManager._extract_events()` 识别。
4. 每条事件至少有可读的 `title` 或 `content`。
5. 如实现 ACK，调用后同一个 `event_id` 不应继续返回。
6. 修改后提醒用户执行 `/mcp_reload` 或重启 agent，让 MCP 配置重新加载。

## 注意

- 不要使用旧 Akashic 路径，例如 `$HOME/.akashic/workspace/...`。
- 不要引用 `proactive_v2.*`，当前项目模块是 `chat_agent.proactive.*`。
- 不要把 API key、token 或私密 cookie 写进 skill 或提交到仓库。
- 不要为了简单 RSS 新建 server；优先复用 `feed_bridge`。
- 不要让 MCP server 只在内存里记 ACK；stdio server 可能重启，状态会丢失。
- 不要假设 `read_file` / `write_file` 能访问 `workspace/` 根目录配置；默认只能访问 `workspace/files`。
- 主动发送仍受 daily cap、cooldown、quiet hours、presence busy、dedupe 等限制；source 有内容不代表一定立刻推送。
