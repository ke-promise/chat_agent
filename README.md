# 个人智能体

一个可在 Telegram 和 QQ 官方 Bot 中运行的个人智能体项目。它围绕 OpenAI-compatible LLM、长期记忆、提醒、MCP 工具、主动消息、`SKILL.md` 说明书和本地表情包库构建，目标是把“陪伴型个人助手”的完整链路在本地跑通，并保持结构清晰、可扩展、便于继续实验。

## 项目特点

- 支持 Telegram long polling 和 QQ 官方 Bot Webhook 两种渠道。
- 支持文本和图片输入；主模型开启视觉能力后可读取图片附件。
- 维护会话历史、长期记忆、候选记忆、用户画像、会话摘要和 Markdown 审计档案。
- 内置工具循环，支持记忆、提醒、网页抓取、文件读写、skills、表情包和主动发送。
- 支持 MCP server 接入，当前模板覆盖 DuckDuckGo 搜索、网页正文抓取、RSS/feed 桥接等能力。
- 支持 `SKILL.md` 说明书机制，让模型按 SOP 工作，而不是把所有规则硬塞进 prompt。
- 支持主动触达：到期提醒、feed 候选、drift 后台整理、轻量 fallback 问候。
- 支持本地表情包目录、自动挂图和用户图片收录。
- 具备独立 observe 数据库，记录消息 trace、MCP 调用、主动 tick、候选审计等运行信息。

## 当前能力

- 被动对话：处理 Telegram/QQ 文本和图片消息，支持白名单控制。
- 长期记忆：支持显式“记住：...”、自然语言偏好抽取、候选记忆晋升、记忆纠正、回忆查询、后台 consolidation。
- 检索增强：支持 SQLite FTS5 BM25、可选 embedding、可选 Chroma、RRF 融合和 HTTP reranker。
- 提醒系统：支持自然语言提醒，例如“1 分钟后提醒我喝水”。
- 工具循环：支持 OpenAI-compatible `tool_calls`，也兼容文本协议 `<tool_call ...>...</tool_call>`。
- Skills：支持内置 skills、`workspace/skills/` 覆盖、依赖检查、catalog 注入和按需全文注入。
- MCP：可加载 `duckduckgo`、`web_content`、`feed_bridge`、`rss` 等 server，并按 allowlist 注册工具。
- 主动消息：后台循环优先发送到期提醒，再评估 feed、drift、deferred candidate 和 fallback 候选。
- 表情包系统：支持 `list_memes`、`send_meme`、自动挂图，以及把收到的图片收录到本地分类。

## 目录结构

```text
.
├─ chat_agent/                 核心 Python 代码
│  ├─ agent/                   OpenAI-compatible LLM provider
│  ├─ channels/                Telegram / QQ 通道
│  ├─ memory/                  记忆、摘要、向量检索、reranker、档案导出
│  ├─ mcp/                     MCP client/registry
│  ├─ observe/                 trace 记录
│  ├─ proactive/               主动消息、feed、drift
│  ├─ tools/                   内置工具注册
│  ├─ context.py               上下文拼装
│  ├─ loop.py                  被动消息主循环
│  ├─ reasoner.py              模型推理与工具协调
│  ├─ skills.py                SKILL.md 加载与注入
│  └─ memes.py                 表情包目录与自动挂图
├─ skills/                     项目内置技能说明书
├─ workspace/                  运行时工作目录
│  ├─ mcp_servers/             本地 MCP 示例 server
│  ├─ skills/                  用户自定义 skills
│  ├─ drift/skills/            drift 专用 skills
│  └─ memory/                  记忆审计 Markdown 输出
├─ package/                    DuckDuckGo/IAsk/Monica 搜索 MCP Node 包
├─ tests/                      单元测试
├─ main.py                     CLI 入口
├─ config.example.toml         配置模板
├─ shujuliu-xiangjie.md        全链路数据流说明
└─ README.md
```

## 快速开始

### 1. 安装依赖

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
cd package
npm install
cd ..
```

### 2. 初始化项目目录

```powershell
python main.py init
Copy-Item config.example.toml config.toml
```

`python main.py init` 会准备：

- `workspace/`
- `logs/`
- `observe/`
- `workspace/mcp_servers.json`
- `workspace/proactive_sources.json`
- `workspace/drift_tasks.json`
- `workspace/rss_feeds.opml`
- `workspace/skills/`
- `workspace/drift/skills/`
- `workspace/memory/`

### 3. 配置环境变量

默认模板使用 DashScope 的 OpenAI-compatible 接口，也可以换成其他兼容服务。

```powershell
$env:QWEN_API_KEY="your-api-key"
$env:TELEGRAM_BOT_TOKEN="your-telegram-bot-token"
# 使用 QQ 官方 Bot 时改用：
$env:QQ_BOT_APP_ID="your-qq-bot-app-id"
$env:QQ_BOT_APP_SECRET="your-qq-bot-app-secret"
```

### 4. 修改 `config.toml`

最少需要确认：

- `[llm.main]`：主对话模型，负责正常回复、图片理解、最终工具推理。
- `[llm.fast]`：轻量模型，负责 query rewrite、HyDE、主动消息改写、drift 和 tie-break。
- `[channel]`：`type = "telegram"` 或 `type = "qq"`。
- `[telegram]`：Telegram token、用户名白名单、图片下载策略。
- `[qq]`：QQ app 凭证、Webhook 监听地址、签名校验、白名单、图片下载策略。
- `[memory]`：主业务 SQLite，默认 `workspace/agent.sqlite3`。
- `[observe]`：观测 SQLite，默认 `observe/observe.db`。

### 5. 启动

```powershell
python main.py
```

使用其他配置文件：

```powershell
python main.py --config path\to\config.toml
```

## 配置概览

### LLM

`[llm.main]` 和 `[llm.fast]` 都是 OpenAI-compatible 配置。

- `main` 适合主回复、工具循环和视觉输入。
- `fast` 适合低成本判断、检索增强、主动系统润色和 drift。
- 两者可以指向同一个模型，但通常成本和延迟会更高。

### Telegram

`[telegram]` 重点字段：

- `token`
- `allow_from`
- `unauthorized_reply`
- `download_images`
- `image_max_mb`

当 `download_images = true` 时，收到的图片会下载到 `workspace/attachments/`，并作为附件注入上下文。

### QQ 官方 Bot

把 `[channel].type` 改成 `"qq"` 后，程序会启动本地 Webhook 服务。

- `app_id` / `app_secret`：QQ 开放平台机器人凭证。
- `sandbox`：是否使用沙箱 API。
- `host` / `port` / `path`：Webhook 监听地址，例如 `http://your-domain:8080/qqbot`。
- `verify_signature`：默认校验 QQ 回调签名。
- `allow_from`：QQ openid 白名单，留空表示允许所有来源。
- `download_images` / `image_max_mb`：是否下载 QQ 图片附件，以及单张图片大小限制。
- `max_text_chars`：QQ 文本回复的最大字符数，默认 `1800`。

主动推送的 `proactive.loop.target_chat_id` 需要带场景前缀，例如 `c2c:<user_openid>`、`group:<group_openid>`、`channel:<channel_id>` 或 `dm:<guild_id>`。

### 记忆与检索

`[memory]` 控制：

- 会话历史窗口
- 长期记忆召回数量
- BM25 / vector / RRF 参数
- 摘要阈值
- query rewrite
- HyDE

`[embedding]` 当前支持：

- `provider = "sqlite_json"`：向量存在 SQLite，搜索时在本地计算余弦相似度。
- `provider = "chroma"`：向量检索接到外部 Chroma，记忆正文仍以 SQLite 为准。

如果开启 Chroma，需要先启动服务：

```powershell
chroma run --host localhost --port 8000
```

`[reranker]` 可选接入 HTTP rerank 服务，用于对 RRF 候选再精排。

### 工具

`[tools]` 控制工具循环和文件工作区。默认文件工作区是 `workspace/files`。

- 默认可见：`memorize`、`recall_memory`、`create_reminder`、`list_reminders`、`tool_search`、`list_memes`
- 启用 skills 后默认可见：`list_skills`、`read_skill`
- 可通过 `tool_search` 发现：`cancel_reminder`、`web_fetch`、`list_files`、`read_file`、`send_meme`
- 默认隐藏：`write_file`、`send_message`、`send_emoji`、`create_skill`、`update_skill`

隐藏工具可以通过 `[tools].extra_model_tools` 主动暴露给模型。MCP 工具注册后也可按注册名加入这里，例如 `web_content_fetch_page`。

### MCP

`[mcp]` 默认从 `workspace/mcp_servers.json` 加载 server 定义，并用 `allowed_servers` / `allowed_tools` 控制可用范围。

模板包含：

- `duckduckgo`：依赖 `package/` 下 Node 包，提供搜索工具。
- `web_content`：本仓库自带网页正文预览 MCP server。
- `feed_bridge`：本仓库自带 RSS/Atom 到 proactive 事件桥接 server。
- `rss`：第三方 `mcp_rss` 示例，依赖 MySQL，默认关闭。

注册到模型侧时，MCP 工具名会从 `<server>:<tool>` 规整为 `<server>_<tool>`，例如 `duckduckgo:web-search` 会注册为 `duckduckgo_web_search`。

### Skills

`[skills]` 控制 `SKILL.md` 说明书系统。来源有两类：

- 项目内置：`skills/<name>/SKILL.md`
- 工作区覆盖：`workspace/skills/<name>/SKILL.md`

同名时，`workspace` 版本优先。完整 `SKILL.md` 不会每轮全量注入；系统会先注入 catalog 摘要，只有 always skill 或被用户显式触发的 skill 才会全文注入。

### 主动系统

主动系统相关配置位于：

- `[proactive.loop]`
- `[proactive.budget]`
- `[proactive.feed]`
- `[proactive.drift]`
- `[proactive.drift.skills]`
- `[proactive.fallback]`
- `[proactive.presence]`
- `[scheduler]`

想让 bot 主动发消息，至少需要：

- `[proactive.loop].enabled = true`
- 设置 `[proactive.loop].target_chat_id`
- `[scheduler].enabled = true`

主动循环会优先发送到期提醒；非提醒候选会经过预算、静默时段、忙碌状态、去重、source cap、语义相似度过滤和候选打分。

## 运行机制

### 被动对话链路

```text
Telegram Update / QQ Webhook event
  -> channels/telegram.py 或 channels/qq.py
  -> InboundMessage
  -> AgentLoop
  -> ContextBuilder
  -> Reasoner
  -> ToolRegistry / MCP
  -> SQLiteStore + TraceRecorder
  -> OutboundMessage
  -> 当前通道发送
```

### 主动消息链路

```text
ProactiveLoop tick
  -> due reminders
  -> feed candidates
  -> drift result
  -> deferred candidates
  -> fallback candidate
  -> budget / quiet-hours / dedupe / busy filters
  -> choose winner
  -> 当前通道发送
```

## 记忆系统

当前项目的记忆由多条链路共同组成：

- 会话历史：每轮 `user` 和 `assistant` 消息都会进入 `session_messages`。
- 长期记忆：显式记忆、推断记忆、候选晋升记忆和纠错后的记忆都会进入 `memories`。
- 候选记忆：低置信线索先进入 `memory_candidates`，重复证据足够后自动晋升。
- 用户画像：抽取姓名、喜好、禁忌、回复风格等轻量信息。
- 会话摘要：达到阈值后刷新 `conversation_summaries`。
- 向量索引：embedding 开启后，记忆会同步进入 `memory_embeddings` 或 Chroma。
- 审计档案：consolidation 会在 `workspace/memory/<chat_id>/` 导出 Markdown 快照。

默认能处理这些自然语言场景：

```text
记住：我喜欢简洁回答
你记得我喜欢什么？
不是这个，我喜欢的是……
对，就是这个
```

## 表情包系统

表情包目录默认位于 `workspace/files/memes/`。

- 显式发送：模型可通过 `list_memes` 和 `send_meme` 选择并发送。
- 自动挂图：系统根据消息内容、情绪信号、冷却时间、来源策略和信息密度决定是否自动附图。
- 图片收录：用户发图后，可用文字把图片收录成本地表情包。

示例：

```text
存成表情包：开心
加入表情包 可爱
```

收录后的素材会进入本地目录，并维护 `manifest.json` 元数据。

## MCP 与主动 Feed

默认相关文件：

- `workspace/mcp_servers.json`
- `workspace/proactive_sources.json`
- `workspace/rss_feeds.opml`

`feed_bridge` 暴露：

- `poll_feeds`
- `get_proactive_events`
- `ack_events`

`web_content` 暴露：

- `fetch_page`

`duckduckgo` Node 包暴露：

- `web-search`
- `iask-search`
- `monica-search`

## 常用命令

Telegram 和 QQ 文本命令都支持下面这些常用项：

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

文本对话：

```text
你好
记住：我喜欢简洁回答
你记得我喜欢什么？
1分钟后提醒我喝水
```

图片理解：

```text
帮我看看这张图里有什么
```

表情包收录：

```text
存成表情包：抱抱
```

## 测试

```powershell
pytest
python -m compileall chat_agent main.py tests
```

当前测试不会调用真实的 LLM、Telegram Bot API 或外部 MCP server。最近一次本地验证为 `131 passed`。

## 运行产物

默认运行后会看到这些本地数据：

- `workspace/agent.sqlite3`：业务主库，保存消息、记忆、提醒、主动候选等。
- `observe/observe.db`：观测库，保存 trace、MCP 调用日志、proactive tick 等。
- `logs/app.log`：应用日志。
- `workspace/attachments/`：下载的 Telegram/QQ 图片附件。
- `workspace/files/`：工具可访问的文件工作区，也是表情包目录根。
- `workspace/memory/`：长期运行时整理出的 Markdown 审计档案。
- `workspace/drift_runs/`：drift 后台任务产物。

## 常见问题

- 启动时报 `Config error`：通常是 `config.toml` 缺字段，或 `${ENV_NAME}` 对应环境变量未设置。
- 图片无法理解：检查 `[llm.main].enable_vision = true`，并确认主模型支持视觉输入。
- DuckDuckGo MCP 用不了：先进入 `package/` 执行 `npm install`。
- 主动消息不发送：重点检查 `proactive.loop.target_chat_id`、预算配置、静默时段和 `presence.skip_when_busy`。
- Chroma 检索无效：确认 Chroma 服务已启动；如果不需要外部服务，切回 `provider = "sqlite_json"`。
- QQ 收不到图片附件：检查 `[qq].download_images`，以及 QQ 平台对应场景是否允许本地图片上传或 URL 图片发送。

## 适合继续扩展的方向

- 增加更多内置 skills 和 drift 专用 skills。
- 接入更多 MCP server。
- 为主动消息增加更强的排序策略和用户反馈回路。
- 补充更细的记忆治理、合并、删除和审计工具。
- 为表情包系统增加管理命令和审核流程。
