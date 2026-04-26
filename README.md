# Telegram 个人智能体

这是一个 Telegram-only 的个人智能体项目。它参考 Akashic Agent 的运行时分层，但不引入 QQ、Discord、NapCat 或复杂多通道抽象。当前能力包括：文本对话、Telegram 图片输入、主/次 LLM、工具循环、SKILL.md 技能说明书、长期记忆、提醒、MCP 工具接入、主动 feed source、drift 空闲任务、HyDE/query rewrite 和 SQLite trace。

## 架构

```text
Telegram Update
  -> channels/telegram.py
  -> InboundMessage / Attachment
  -> loop.py
  -> context.py
  -> reasoner.py
  -> skills catalog / active skills
  -> tools/registry.py
  -> memory/store.py + observe/trace.py
  -> OutboundMessage
  -> Telegram send

proactive/loop.py
  -> due reminders
  -> MCP feed source
  -> context fallback
  -> Telegram send
```

## 安装

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
cd package
npm install
cd ..
python main.py init
copy config.example.toml config.toml
```

`python main.py init` 会创建：

- `workspace/`
- `logs/`
- `workspace/mcp_servers.json`
- `workspace/proactive_sources.json`
- `workspace/drift_tasks.json`
- `workspace/skills/`
- `workspace/drift/skills/`

## 配置主/次 LLM

```toml
[llm.main]
model = "qwen-vl-plus"
api_key = "${QWEN_API_KEY}"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
timeout_seconds = 60
max_tokens = 4096
enable_vision = true

[llm.fast]
model = "qwen-flash"
api_key = "${QWEN_API_KEY}"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
timeout_seconds = 20
max_tokens = 1024
enable_vision = false
```

`main` 用于普通对话、图片理解、工具循环最终推理。`fast` 用于 query rewrite、HyDE、主动 feed relevance judge 等轻量任务。两者可以用不同模型、不同 base_url 和不同 key。

环境变量：

```powershell
$env:QWEN_API_KEY="your-api-key"
$env:TELEGRAM_BOT_TOKEN="your-telegram-token"
```

缺少 `${ENV_NAME}` 时会在启动时报清楚。日志会对 token/key 做脱敏。

## Telegram 图片

配置：

```toml
[telegram]
download_images = true
image_max_mb = 10
```

发送图片时，bot 会生成 `Attachment`，把图片 URL 或本地路径注入上下文。如果 `[llm.main].enable_vision = true`，图片会以 OpenAI-compatible multimodal content 传给主模型。如果主模型不支持 vision，会友好提示“当前主模型不支持图片理解”。

## 工具循环

支持 OpenAI-compatible `tool_calls`，也保留文本协议：

```text
<tool_call name="memorize">
{"content":"用户喜欢简洁回答","tags":["preference"],"type":"preference"}
</tool_call>
```

内置工具：

- `memorize`
- `recall_memory`
- `create_reminder`
- `list_reminders`
- `cancel_reminder`
- `web_fetch`
- `tool_search`

默认只暴露 always-on 工具。需要更多工具时，模型可以先调用 `tool_search` 解锁。工具循环有最大迭代次数和重复调用防护。

## Skills 说明书

`skills` 是 Akashic 风格的能力说明书系统。它和 tools 的区别是：

```text
tools = Agent 能调用的动作，比如保存记忆、创建提醒、读取网页。
skills = Agent 做事前看的说明书 / SOP，比如天气查询该怎么问、总结该保留什么。
```

skill 是一个目录里的 `SKILL.md`：

```text
skills/weather/SKILL.md
workspace/skills/my-skill/SKILL.md
workspace/drift/skills/my-drift-skill/SKILL.md
```

格式：

```markdown
---
name: weather
description: 查询天气。当用户问天气、温度、预报时使用。
metadata: {"chat_agent":{"always":false,"requires":{"bins":["curl"],"env":[]}}}
---

# Weather

这里写给 Agent 看的操作说明。
```

配置：

```toml
[skills]
enabled = true
builtin_dir = "skills"
workspace_dir = "workspace/skills"
inject_catalog = true
max_catalog_chars = 4000
```

每轮 prompt 会注入 skills catalog 摘要，但不会把所有 `SKILL.md` 全量塞进去。完整 skill 只会在这些情况下进入上下文：

- `metadata.chat_agent.always = true`
- 用户显式提到 `@weather`
- 用户显式提到 `skill:weather`
- 用户文本包含 skill name

workspace skill 优先级高于内置 skill，同名会覆盖。`always=true` 会增加每轮 prompt 长度，建议只给非常稳定、必须常驻的规则使用。

依赖检查：

- `requires.bins`：检查本机命令是否存在。
- `requires.env`：检查环境变量是否存在。

Telegram 命令：

- `/skills`：查看所有 skill、来源和可用状态。

内置工具：

- `list_skills`
- `read_skill`
- `create_skill`
- `update_skill`

这些工具只允许写入 `workspace/skills/`，不会修改项目内置 `skills/`。

## 记忆检索增强

配置：

```toml
[memory]
query_rewrite_enabled = true
hyde_enabled = true
```

检索流程：

1. fast LLM 判断是否需要检索。
2. fast LLM 改写检索 query。
3. HyDE 生成假想记忆。
4. 分别用 raw query / rewritten query / HyDE query 检索 SQLite。
5. 合并去重后注入 prompt。

fast LLM 失败时会回退到原始 keyword/LIKE 检索，不阻断主回复。

## MCP 工具

`workspace/mcp_servers.json` 示例：

```json
{
  "servers": {
    "duckduckgo": {
      "enabled": true,
      "command": ["node", "package\\bin\\cli.js"],
      "env": {}
    },
    "rss": {
      "enabled": false,
      "command": ["npx.cmd", "-y", "mcp_rss"],
      "env": {
        "OPML_FILE_PATH": "workspace\\rss_feeds.opml",
        "DB_HOST": "localhost",
        "DB_PORT": "3306",
        "DB_USERNAME": "root",
        "DB_PASSWORD": "123456",
        "DB_DATABASE": "mcp_rss",
        "RSS_UPDATE_INTERVAL": "1"
      }
    },
    "web_content": {
      "enabled": true,
      "command": [".venv\\Scripts\\python.exe", "workspace\\mcp_servers\\web_content_mcp_server.py"],
      "env": {
        "PYTHONUTF8": "1"
      }
    },
    "feed_bridge": {
      "enabled": true,
      "command": [".venv\\Scripts\\python.exe", "workspace\\mcp_servers\\feed_bridge_mcp_server.py"],
      "env": {
        "PYTHONUTF8": "1",
        "WEB_FEEDS": "[{\"name\":\"python-insider\",\"url\":\"https://blog.python.org/rss.xml\"}]"
      }
    }
  }
}
```

当前实现是最小 stdio JSON-RPC MCP client，支持：

- `initialize`
- `tools/list`
- `tools/call`

启动失败只记录 warning，不会让 bot 崩溃。MCP 工具会注册进 `ToolRegistry`，source 为 `mcp:<server>`。

DuckDuckGo Search 使用仓库内固定的 npm MCP 包 `@oevortex/ddg_search@1.2.2`，启动命令固定为 `node package\bin\cli.js`。它不需要 API key，但首次使用前需要先在 `package/` 目录执行一次 `npm install`，避免运行时 `npx` 冷启动。

这里要区分两类能力：

- `search` 负责发现候选链接。
- builtin `web_fetch` 和 MCP `fetch_page` 负责读取单个 URL 的正文预览。

RSS 示例使用第三方 `mcp_rss`，它依赖 OPML 文件和 MySQL。`python main.py init` 会创建 `workspace/rss_feeds.opml`。准备好 MySQL 后，把 `workspace/mcp_servers.json` 里的 `"rss"` 改成 `"enabled": true`，再执行 `/mcp_reload`。

Telegram 命令：

- `/mcp`：查看已连接 server 和工具数。
- `/mcp_reload`：重新加载 MCP server。

## 主动 Feed Source

`workspace/proactive_sources.json` 示例：

```json
{
  "sources": [
    {
      "server": "rss",
      "channel": "content",
      "poll_tool": null,
      "get_tool": "get_content",
      "get_args": {
        "status": "normal",
        "limit": 10
      },
      "ack_tool": null,
      "enabled": false
    },
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

tick 流程：

```text
tick
  -> due reminders 优先
  -> collect feed candidates
  -> run drift preparation
  -> optional fallback candidate
  -> 统一预算 / 去重 / quiet hours / source cap 过滤
  -> 统一排序，最多发送 1 条
  -> ack、写审计和 trace
```

`/proactive_status` 可以查看最近 tick、内容数、发送数和 seen item 数。

## Drift 空闲任务

Drift 现在是后台 preparation layer，不再默认承担“主动发新闻”的职责。它会在后台整理记忆、准备 follow-up 草稿、维护内部观察笔记；如果某次结果被显式标记为 `shareable=true`，才会被提升为普通主动候选，并与 feed / fallback 一起进入统一排序。

配置示例：

```toml
[proactive.loop]
enabled = true
tick_interval_seconds = 60
target_chat_id = ""

[proactive.budget]
daily_max = 6
min_interval_minutes = 90
quiet_hours_start = ""
quiet_hours_end = ""

[proactive.feed]
enabled = true
sources_path = "workspace/proactive_sources.json"
daily_cap = 3

[proactive.drift]
enabled = false
tasks_path = "workspace/drift_tasks.json"
output_dir = "workspace/drift_runs"
run_cooldown_minutes = 180
daily_run_cap = 3
promotion_enabled = true
daily_cap = 2

[proactive.fallback]
enabled = false
probability = 0.03
daily_cap = 2
```

`python main.py init` 会创建 `workspace/drift_tasks.json`，格式如下：

```json
{
  "tasks": [
    {
      "id": "memory_review",
      "title": "整理近期记忆线索",
      "prompt": "阅读近期摘要、长期记忆和待提醒，输出一份后台整理笔记。默认不要直接打扰用户，只有内容高价值、高置信且适合马上说出口时，才在 candidate 元数据中把 shareable 设为 true。",
      "enabled": true
    }
  ]
}
```

运行结果会保存到 `workspace/drift_runs/`，并写入 SQLite 的 `drift_runs` 表。是否真正发给用户，不再由 drift 自己决定，而是由 `ProactiveLoop` 把它当成普通候选统一评估。

drift 现在也会走工具循环。也就是说，后台任务可以通过 `tool_search` 解锁 MCP 工具，再调用 Brave Search 或 RSS 工具获取外部信息。

Drift 也可以使用 skill：

```toml
[proactive.drift.skills]
enabled = true
workspace_dir = "workspace/drift/skills"
include_builtin = true
```

开启后，drift 会优先扫描 `workspace/drift/skills/`，并可使用内置 skills 中标记 `metadata.chat_agent.drift=true` 的说明书；如果没有可用 drift skill，会回退到原来的 `workspace/drift_tasks.json`。

## 常用命令

- `/start`
- `/help`
- `/status`
- `/memory`
- `/forget <id>`
- `/mcp`
- `/mcp_reload`
- `/skills`
- `/proactive_status`

## 示例

```text
你好
记住：我喜欢简洁回答
你记得我喜欢什么？
1分钟后提醒我喝水
```

也可以直接发送图片，并配 caption：

```text
帮我看看这张图里有什么
```

## 测试

```powershell
pytest
python -m compileall chat_agent main.py tests
```

测试不会调用真实 LLM、Telegram 或外部 MCP server。

## 常见问题

- 主模型不支持图片：确认 `[llm.main].enable_vision = true`，并使用视觉模型。
- API key 401：检查环境变量、base_url、模型供应商配置。
- Telegram 文件下载失败：检查 bot token、网络和 `image_max_mb`。
- MCP server 启动失败：看日志里的 warning，确认 command 路径和 env。
- feed poll 空：确认 MCP server 暴露了 `poll_tool/get_tool`，并且 `proactive_sources.json` 配置正确。
- 主动消息不发：检查 `proactive.loop.target_chat_id`、`proactive.budget.daily_max`、`proactive.budget.min_interval_minutes`、`seen_items` 去重和 source cap。

## 当前补充能力

本版本进一步补齐了更接近长期运行个人智能体的几个基础能力：

- 用户画像：`user_profiles` 表会保存用户名称、偏好、禁忌和回答风格等轻量画像；`ContextBuilder` 会把画像注入上下文。
- 回答后异步记忆写入：普通回复发送前不会被记忆抽取阻塞；本轮提交后会启动后台任务，用规则抽取稳定偏好和事实。
- 文件工具：`list_files`、`read_file`、`write_file` 只能访问 `[tools].file_workspace`，默认是 `workspace/files`，不能逃逸到项目其他目录。
- 消息推送工具：`send_message` 允许模型在工具循环中向当前 chat 或 `proactive.loop.target_chat_id` 发送消息；默认需要通过 `tool_search` 解锁。
- 独立观测库：新增 `[observe].database_path`，默认写到 `observe/observe.db`；消息 trace、MCP 工具日志和 proactive tick 会进入观测库，同时 proactive tick 也保留在主库供 `/status` 使用。

配置示例：

```toml
[tools]
file_workspace = "workspace/files"

[observe]
database_path = "observe/observe.db"
```

## Embedding 与向量记忆

向量记忆分两个阶段实现：

第一阶段已经可用：`provider = "sqlite_json"`。记忆仍保存在 SQLite 的 `memories` 表中，
embedding 会写入 `memory_embeddings` 表，检索时会把 LIKE、query rewrite、HyDE 和向量相似度结果合并。
这个阶段不需要部署外部向量数据库，适合个人助手早期使用。

```toml
[embedding]
enabled = true
provider = "sqlite_json"
model = "text-embedding-v4"
api_key = "${QWEN_API_KEY}"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
timeout_seconds = 30
dimension = 1024
top_k = 5
min_score = 0.2
```

第二阶段已经接入 Chroma：`provider = "chroma"`。记忆正文仍以 SQLite 为准，
Chroma 只保存 `memory_id`、`chat_id` 和 embedding，用于高效向量相似度检索。
如果你没有单独启动 Chroma server，请不要把 `provider` 改成 `chroma`；直接使用 `sqlite_json` 即可。

先启动 Chroma server：

```powershell
chroma run --host localhost --port 8000
```

然后修改配置：

```toml
[embedding]
enabled = true
provider = "chroma"
model = "text-embedding-v4"
api_key = "${QWEN_API_KEY}"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
timeout_seconds = 30
dimension = 1024
top_k = 5
min_score = 0.2
external_url = "http://localhost:8000"
external_api_key = ""
collection = "chat_agent_memories"
```

Chroma 依赖 `chromadb`。更新依赖后运行：

```powershell
pip install -e ".[dev]"
```

## 记忆持久化链路

当前项目的记忆已经不是单一 SQLite 表，而是三层落盘：

1. 原始对话层：`AgentLoop._commit()` 每轮都会把 `user` 和 `assistant` 两条消息写入 `session_messages`，这是后续整理长期记忆的原料。
2. 语义记忆层：显式 `memorize`、轻量抽取和 consolidation 都会写入 `memories`；开启 embedding 后会同步写入 SQLite JSON 向量或 Chroma。
3. 文本档案层：`workspace/memory/` 下维护 `HISTORY.md`、`PENDING.md`、`MEMORY.md`、`RECENT_CONTEXT.md`、`SELF.md`、`NOW.md`。

整理流程是：回复结束后触发 post-turn consolidation，旧会话窗口会写入 `HISTORY.md` / `PENDING.md` / `memories`，成功后才推进 `last_consolidated` 检查点。后台 `MemoryOptimizerLoop` 会定期把 `PENDING.md` 合并进 `MEMORY.md`，并使用快照回滚避免合并失败时丢失 pending 内容。
